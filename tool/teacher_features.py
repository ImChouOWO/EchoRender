"""Capture stages from the pinned functional Teacher, without copying its graph.

E1/E2/E3: last window output before fused pooling/projection; ViT: last global
block. The existing backend performs every operation. Captures are detached BCHW.
"""
import torch
import mlxdlss.model as backend
from mlxdlss.model import NeuralRenderingModel


class FeatureRecordingBackend(NeuralRenderingModel):
    def __init__(self, weights):
        super().__init__(weights)
        self._captures = None
        self._graph_enabled = False
        self._graph = None
        self._graph_biases = None
        self._window_names = {
            self.topology.encoder_stage(stage)[-1]: f'e{stage}'
            for stage in (1, 2, 3)
        }

    def _capture(self, name, value):
        if self._captures is not None:
            self._captures[name] = value.detach().permute(0, 3, 1, 2).contiguous()

    def _window(self, value, index, *, head_count, publish=True):
        if getattr(self, '_graph_biases', None) is None:
            result = super()._window(value, index, head_count=head_count, publish=publish)
        else:
            result = self._window_cached(value, index, head_count=head_count, publish=publish)
        if index in self._window_names:
            self._capture(self._window_names[index], result)
        return result

    def _global(self, value, index):
        result = super()._global(value, index)
        if index == self.topology.global_stage[-1]:
            self._capture('vit', result)
        return result

    def _eager_with_features(self, value):
        if self._captures is not None:
            raise RuntimeError('Teacher feature extraction cannot be nested')
        self._captures = {}
        try:
            head = self(value)
            if set(self._captures) != {'e1', 'e2', 'e3', 'vit'}:
                raise RuntimeError('Teacher did not execute all selected stages')
            return head, self._captures
        finally:
            self._captures = None


    def enable_cuda_graph(self):
        self._graph_enabled = True

    def _clear_graph(self):
        graph = getattr(self, '_graph', None)
        if graph is not None:
            graph.reset()
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self._graph_key = None
        self._graph_biases = None

    def _apply(self, fn, recurse=True):
        self._clear_graph()
        return super()._apply(fn, recurse=recurse)

    def _split_attention_bias(self, index):
        name = f'block{index}.layer2.attn_bias'
        if getattr(self, '_graph_biases', None) is not None:
            return self._graph_biases[name]
        return super()._split_attention_bias(index)

    def _prepare_graph_biases(self):
        self._graph_biases = {}
        for name in self._weight_attributes:
            if name.endswith('.attn_bias'):
                bias = self.weight(name)
                index = int(name.split('.')[0][5:])
                if backend.uses_fragment_swizzle(index, bias.shape[0]):
                    bias = backend.recover_attention_bias_layout(bias)
                self._graph_biases[name] = bias

    def _capture_graph(self, value):
        # The reference capture is retained only for exact first-input validation.
        reference_head, reference_features = self._eager_with_features(value)
        self._prepare_graph_biases()
        self._graph_input = value.clone()
        with torch.cuda.device(value.device), torch.inference_mode():
            side = torch.cuda.Stream(device=value.device)
            side.wait_stream(torch.cuda.current_stream(value.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    warmup = self._eager_with_features(self._graph_input)
                del warmup
            torch.cuda.current_stream(value.device).wait_stream(side)
            torch.cuda.synchronize(value.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._eager_with_features(self._graph_input)
            graph.replay()
            torch.cuda.synchronize(value.device)
            if not torch.equal(reference_head, output[0]) or any(
                    not torch.equal(reference_features[name], output[1][name]) for name in reference_features):
                raise RuntimeError('CUDA Graph output differs from reference Teacher')
            self._graph = graph
            self._graph_output = output
            self._graph_key = (tuple(value.shape), value.dtype, value.device)
        print(f'[Teacher]: CUDA Graph ready for {tuple(value.shape)}; exact-output check passed.', flush=True)

    def forward_with_features(self, value):
        if (not getattr(self, '_graph_enabled', False) or value.device.type != 'cuda'
                or value.shape[0] != 1):
            return self._eager_with_features(value)
        key = (tuple(value.shape), value.dtype, value.device)
        if self._graph is None or self._graph_key != key:
            self._clear_graph()
            try:
                self._capture_graph(value)
            except torch.OutOfMemoryError:
                self._clear_graph()
                raise
            except RuntimeError as error:
                self._clear_graph()
                self._graph_enabled = False
                print(f'[Teacher]: CUDA Graph unavailable; using reference inference: {error}', flush=True)
                return self._eager_with_features(value)
        self._graph_input.copy_(value)
        self._graph.replay()
        return self._graph_output

    def _window_cached(
        self,
        value: torch.Tensor,
        index: int,
        *,
        head_count: int,
        publish: bool = True,
    ) -> torch.Tensor:
        prefix = f"block{index}.layer0"
        origin = self.topology.window_origin(index)
        attention_bias = self._graph_biases[f"{prefix}.attn_bias"]
        if f"{prefix}.ffn_expand_weight" in self._weight_attributes:
            output = backend.branched_window_block(
                value,
                expansion_weight=self.weight(f"{prefix}.ffn_expand_weight"),
                branch_projection_weight=self.weight(
                    f"{prefix}.ffn_branch_projection_weight"
                ),
                output_projection_weight=self.weight(
                    f"{prefix}.ffn_output_projection_weight"
                ),
                feed_forward_cosine=self.weight(f"{prefix}.ffn_cos_skip"),
                qkv_weight=self.weight(f"{prefix}.qkv_weight"),
                attention_scale=self.weight(f"{prefix}.attn_scale"),
                attention_bias=attention_bias,
                attention_projection_weight=self.weight(f"{prefix}.projection_weight"),
                attention_cosine=self.weight(f"{prefix}.attn_cos_skip"),
                head_count=head_count,
                window_size=8,
                window_origin=origin,
            )
        else:
            output = backend.window_block(
                value,
                expansion_weight=self.weight(f"{prefix}.weight1"),
                feed_forward_projection_weight=self.weight(f"{prefix}.weight2"),
                feed_forward_cosine=self.weight(f"{prefix}.ffn_cos_skip"),
                qkv_weight=self.weight(f"{prefix}.qkv_weight"),
                attention_scale=self.weight(f"{prefix}.attn_scale"),
                attention_bias=attention_bias,
                attention_projection_weight=self.weight(f"{prefix}.projection_weight"),
                attention_cosine=self.weight(f"{prefix}.attn_cos_skip"),
                head_count=head_count,
                window_size=8,
                window_origin=origin,
            )
        # The pre block is pooled before its publication and the post block
        # feeds the head; ``publish=False`` hands the half-precision output to
        # a fused transition.
        return output if index in (0, self.topology.post) or not publish else backend.e4m3_round_trip(output)


class TeacherFeatureCache:
    """Clean targets shared by all epochs; a process lock protects the directory."""
    def __init__(self, directory):
        from pathlib import Path
        self.directory = Path(directory)
        self.lock = None
        self.benchmark = None
        self.pipeline = None
        self._progress = None
        from threading import RLock
        self._progress_lock = RLock()

    def open(self):
        import fcntl
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory.parent / f'.{self.directory.name}.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError(f'Teacher feature cache is in use: {self.directory}') from None
        try:
            self.validate_files()
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def validate_files(self):
        import re
        paths = list(self.directory.iterdir())
        if any(not path.is_file() or not re.fullmatch(r'(?:(?:train|val)-[0-9a-f]{64}\.pt(?:\.tmp)?|manifest\.json(?:\.tmp)?)', path.name)
               for path in paths):
            raise ValueError(f'Unexpected files in feature cache; refusing to remove them: {self.directory}')
        return paths

    def clear(self):
        for path in self.validate_files():
            path.unlink()
        self.benchmark = None
        self._progress = None

    def close(self, *, delete=True):
        if self.lock is not None:
            try:
                if delete:
                    self.clear()
            finally:
                self.lock.close()
                self.lock = None

    def fingerprint(self, teacher_digest, train_loader, val_loader, config, device):
        import hashlib
        import json
        import inspect
        from pathlib import Path
        import torch
        from . import teacher_pipeline
        def samples(loader):
            dataset = getattr(loader, 'dataset', None)
            if dataset is not None and hasattr(dataset, 'samples'):
                return [(scene, str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                        for scene, path in dataset.samples]
            result = []
            for batch in loader:
                for sample_id, image in zip(batch['sample_ids'], batch['images']):
                    result.append((sample_id, list(image.shape), str(image.dtype),
                                   hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
            return result
        from mlxdlss import model as backend, features as preparation_backend
        from model import teacher as teacher_wrapper
        payload = {'version': 1, 'teacher': teacher_digest,
                   'crop_size': config['training']['crop_size'], 'dataset': config['dataset'],
                   'train': samples(train_loader), 'val': samples(val_loader),
                   'benchmark': config['benchmark'], 'device': str(device), 'torch': torch.__version__,
                   'preparation_backend': hashlib.sha256(Path(preparation_backend.__file__).read_bytes()).hexdigest(),
                   'teacher_wrapper': hashlib.sha256(Path(teacher_wrapper.__file__).read_bytes()).hexdigest(),
                   'preprocessing': hashlib.sha256(Path(teacher_pipeline.__file__).read_bytes()).hexdigest(),
                   'capture': hashlib.sha256(inspect.getsource(FeatureRecordingBackend).encode()).hexdigest(),
                   'backend': hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest()}
        if device.type == 'cuda':
            payload['gpu'] = str(torch.cuda.get_device_properties(device).uuid)
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload['capture'] = 'c9a0c25a43a460839ea3bc2ecab6b56de32cad73113f5396ece004fbe2516d93'
        self._previous_capture_fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload['capture'] = 'a80bd129c1d83250b706a0d9ba1706b70691bf3c85a0d4da2ff1eca5d0e00c33'
        self._legacy_fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return fingerprint

    def read_manifest(self):
        import json
        try:
            payload = json.loads((self.directory / 'manifest.json').read_text())
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    def compatible(self, payload, fingerprint):
        return payload.get('fingerprint') in (fingerprint, getattr(self, '_legacy_fingerprint', fingerprint),
                                             getattr(self, '_previous_capture_fingerprint', fingerprint))

    def reuse(self, fingerprint):
        payload = self.read_manifest()
        self.reason = 'cache index missing or unreadable'
        if not self.compatible(payload, fingerprint):
            if payload:
                self.reason = 'Teacher/data/preprocessing signature changed'
            return False
        if not payload.get('complete', True):
            self.reason = 'cache extraction was interrupted'
            return False
        if not isinstance(payload.get('files'), dict) or not payload['files']:
            return False
        for name, size in payload['files'].items():
            path = self.directory / name
            if path.parent != self.directory or not path.is_file() or path.stat().st_size != size:
                self.reason = f'missing or incomplete target: {name}'
                return False
        self.benchmark = payload.get('benchmark')
        return True

    def prepare(self, fingerprint, blend_scale):
        payload = self.read_manifest()
        if not self.compatible(payload, fingerprint) or not isinstance(payload.get('files'), dict):
            self.clear()
            payload = {'files': {}}
        self.benchmark = payload.get('benchmark')
        self.pipeline = payload.get('pipeline')
        self._progress = {'fingerprint': fingerprint, 'blend_scale': blend_scale, 'files': dict(payload['files'])}
        self.write_manifest(complete=False)

    def write_manifest(self, *, complete):
        import json
        with self._progress_lock:
            payload = {**self._progress, 'files': dict(self._progress['files']), 'complete': complete,
                       'benchmark': self.benchmark, 'pipeline': self.pipeline}
            temporary = self.directory / 'manifest.json.tmp'
            temporary.write_text(json.dumps(payload, sort_keys=True))
            temporary.replace(self.directory / 'manifest.json')

    def finished(self, split, sample_id):
        if self._progress is None:
            return False
        path = self.path(split, sample_id)
        size = self._progress['files'].get(path.name)
        return size is not None and path.is_file() and path.stat().st_size == size

    def record_file(self, path):
        if self._progress is not None:
            with self._progress_lock:
                self._progress['files'][path.name] = path.stat().st_size
                self.write_manifest(complete=False)

    def seal(self, fingerprint, *, blend_scale=None):
        self._progress = {'fingerprint': fingerprint, 'blend_scale': blend_scale,
                          'files': {path.name: path.stat().st_size for path in self.directory.glob('*.pt')}}
        self.write_manifest(complete=True)

    def path(self, split, sample_id):
        import hashlib
        if split not in ('train', 'val'):
            raise ValueError('Feature cache split must be train or val')
        digest = hashlib.sha256(sample_id.encode()).hexdigest()
        return self.directory / f'{split}-{digest}.pt'

    def load(self, split, sample_id, device):
        import torch
        payload = torch.load(self.path(split, sample_id), map_location=device, weights_only=True)
        return payload['head'], payload['features'], payload['rgb']

    def build(self, teacher, train_loader, val_loader, config, *, monitor=None):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        from time import perf_counter
        import torch
        from tqdm.auto import tqdm
        from .teacher_pipeline import compose_rgb_tensor
        from .monitoring import measure_forward
        if self._progress is None:
            self.clear()
        model = getattr(teacher, 'model', None)
        if isinstance(model, FeatureRecordingBackend) and teacher.device.type == 'cuda':
            model.enable_cuda_graph()
        plan = None
        pending = deque()

        def write(path, payload):
            temporary = path.with_suffix('.pt.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)
            self.record_file(path)

        def infer(prepared):
            inputs = prepared.features.to(teacher.device)
            head, features = teacher(inputs, return_features=True)
            rgb = compose_rgb_tensor(head, prepared, blend_scale=teacher.blend_scale)
            payload = {'head': head.detach().cpu(),
                       'features': {name: value.detach().cpu() for name, value in features.items()},
                       'rgb': rgb.detach().cpu()}
            return inputs, payload

        with torch.inference_mode():
            for split, loader in (('train', train_loader), ('val', val_loader)):
                source, total = cache_source(loader)
                items = list(source)
                missing = [item for item in items if not self.finished(split, cache_item_id(item))]
                reused = total - len(missing)
                tqdm.write(f'[Cache {split}]: reuse={reused}/{total}, extract={len(missing)}')
                if not missing and split == 'val' and config['benchmark']['enabled'] and self.benchmark is None:
                    missing = items[:1]
                    reused = total - len(missing)
                if not missing:
                    continue
                source = iter(missing)
                first = next(source)
                bar = tqdm(total=total, initial=reused, desc=f'Teacher cache [{split}]', unit='img',
                           disable=not config['logging']['progress'], dynamic_ncols=True)
                try:
                    if plan is None:
                        started = perf_counter()
                        sample_id, prepared = prepare_cache_item(first, config['training']['crop_size'])
                        prepare_seconds = perf_counter() - started
                        started = perf_counter()
                        inputs, payload = infer(prepared)
                        infer_seconds = perf_counter() - started
                        if isinstance(model, FeatureRecordingBackend) and model._graph is not None:
                            started = perf_counter()
                            model.forward_with_features(inputs)
                            torch.cuda.synchronize(inputs.device)
                            infer_seconds = perf_counter() - started
                        started = perf_counter()
                        write(self.path(split, sample_id), payload)
                        write_seconds = perf_counter() - started
                        sample_bytes = prepared.features.numel() * prepared.features.element_size() + prepared.rgb.nbytes
                        target_bytes = sum(value.numel() * value.element_size() for value in
                                           [payload['head'], payload['rgb'], *payload['features'].values()])
                        plan = cache_worker_plan(prepare_seconds, infer_seconds, write_seconds,
                                                 sample_bytes, target_bytes, total)
                        self.pipeline = {**plan, 'sample_prepare_ms': prepare_seconds * 1000,
                                         'sample_infer_ms': infer_seconds * 1000,
                                         'sample_write_ms': write_seconds * 1000}
                        tqdm.write(f'[Cache workers]: read/prepare={plan["readers"]}, write={plan["writers"]}, '
                                   f'prefetch={plan["prefetch"]}, pending_writes={plan["pending_writes"]}')
                        if split == 'val' and self.benchmark is None and config['benchmark']['enabled']:
                            timing = config['benchmark']
                            self.benchmark = {
                                'device': str(inputs.device), 'precision': 'float32', 'batch_size': 1,
                                'network_bhwc': list(inputs.shape), 'warmup': timing['warmup'],
                                'repeats': timing['repeats'], 'scope': 'forward_only_without_feature_capture',
                                **{f'teacher_{key}': value for key, value in measure_forward(
                                    teacher, inputs, warmup=timing['warmup'], repeats=timing['repeats']).items()},
                            }
                        bar.update(1)
                        if monitor is not None:
                            bar.set_postfix({'GPU': monitor.display()})
                        del inputs, payload, prepared
                    else:
                        # Return the first item to this split without decoding it on the main thread.
                        from itertools import chain
                        source = chain([first], source)
                    with ThreadPoolExecutor(max_workers=plan['readers'], thread_name_prefix='cache-read') as readers, \
                         ThreadPoolExecutor(max_workers=plan['writers'], thread_name_prefix='cache-write') as writers:
                        prepared_items = bounded_cache_prepare(readers, source, config['training']['crop_size'], plan['prefetch'])
                        try:
                            for sample_id, prepared in prepared_items:
                                inputs, payload = infer(prepared)
                                path = self.path(split, sample_id)
                                # Duplicate IDs must never race on their atomic temporary file.
                                for pending_path, future in pending:
                                    if pending_path == path:
                                        future.result()
                                pending.append((path, writers.submit(write, path, payload)))
                                if len(pending) >= plan['pending_writes']:
                                    pending.popleft()[1].result()
                                if split == 'val' and self.benchmark is None and config['benchmark']['enabled']:
                                    timing = config['benchmark']
                                    self.benchmark = {
                                        'device': str(inputs.device), 'precision': 'float32', 'batch_size': 1,
                                        'network_bhwc': list(inputs.shape), 'warmup': timing['warmup'],
                                        'repeats': timing['repeats'], 'scope': 'forward_only_without_feature_capture',
                                        **{f'teacher_{key}': value for key, value in measure_forward(
                                            teacher, inputs, warmup=timing['warmup'], repeats=timing['repeats']).items()},
                                    }
                                del inputs, payload, prepared
                                bar.update(1)
                                if monitor is not None:
                                    bar.set_postfix({'GPU': monitor.display()})
                            while pending:
                                pending.popleft()[1].result()
                        finally:
                            prepared_items.close()
                            # Executors finish before an error can release the cache directory lock.
                finally:
                    bar.close()


def cache_item_id(item):
    source, value = item
    if isinstance(source, str):
        return source
    if hasattr(source, 'samples') and hasattr(source, 'source') and hasattr(source, 'render_pass'):
        scene, path = source.samples[value]
        return f'{source.source}/{source.render_pass}/{scene}/{path.stem}'
    return source[value]['sample_id']


def cache_source(loader):
    dataset = getattr(loader, 'dataset', None)
    if dataset is not None:
        return iter((dataset, index) for index in range(len(dataset))), len(dataset)
    def samples():
        for batch in loader:
            for sample_id, image in zip(batch['sample_ids'], batch['images']):
                yield sample_id, image
    return samples(), sum(len(batch['images']) for batch in loader)


def prepare_cache_item(item, crop_size):
    import torch
    from .distillation import prepare_sample
    source, value = item
    with torch.inference_mode():
        if isinstance(source, str):
            sample_id, image = source, value
        else:
            sample = source[value]
            sample_id, image = sample['sample_id'], sample['image']
        return sample_id, prepare_sample(image, crop_size)


def cache_worker_plan(prepare_seconds, infer_seconds, write_seconds, sample_bytes, target_bytes, total):
    import math
    import os
    from pathlib import Path
    cpus = len(os.sched_getaffinity(0))
    # Respect inherited cgroup CPU limits as well as process affinity.
    group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
    for directory in (group, *group.parents):
        if directory == Path('/sys'):
            break
        limit = directory / 'cpu.max'
        if limit.is_file():
            quota, period = limit.read_text().split()
            if quota != 'max':
                cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    available = memory['MemAvailable']
    for name in ('memory.max', 'memory.high'):
        limit = group / name
        current = group / 'memory.current'
        if limit.is_file() and current.is_file() and limit.read_text().strip() != 'max':
            available = min(available, max(0, int(limit.read_text()) - int(current.read_text())))
    budget = min(256 * 2**20, max(1, available // 32))
    reader_cap = max(1, min(8, max(1, cpus - 1), total, budget // max(4 * sample_bytes, 1)))
    writer_cap = max(1, min(2, max(1, cpus - 1), budget // max(2 * target_bytes, 1)))
    # Choose capacity for ~25% headroom over measured GPU consumption, not maximum CPU usage.
    readers = min(reader_cap, max(1, math.ceil(1.25 * prepare_seconds / max(infer_seconds, 1e-6))))
    writers = min(writer_cap, max(1, math.ceil(1.25 * write_seconds / max(infer_seconds, 1e-6))))
    return {'readers': readers, 'writers': writers,
            'prefetch': max(1, min(2 * readers, budget // max(4 * sample_bytes, 1))),
            'pending_writes': max(1, min(2 * writers, budget // max(2 * target_bytes, 1))),
            'cpu_limit': cpus, 'buffer_budget_mib': round(budget / 2**20, 2)}


def bounded_cache_prepare(pool, source, crop_size, capacity):
    from collections import deque
    from itertools import islice
    pending = deque(pool.submit(prepare_cache_item, item, crop_size) for item in islice(source, capacity))
    try:
        while pending:
            result = pending.popleft().result()
            yield result
            item = next(source, None)
            if item is not None:
                pending.append(pool.submit(prepare_cache_item, item, crop_size))
    finally:
        for future in pending:
            future.cancel()


def ask_delete_cache(directory):
    while True:
        try:
            answer = input('[Done]: Delete cached features? [Y/N]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\n[Kept]: Cached features retained.')
            return False
        if answer in ('y', 'n'):
            print('[Delete]: Cached features will be deleted.' if answer == 'y' else '[Kept]: Cached features retained for reuse.')
            return answer == 'y'
        print('Please enter Y or N.')
