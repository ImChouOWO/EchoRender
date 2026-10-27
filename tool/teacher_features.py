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
    """Profile/clip/frame targets; exact signatures only, never reuse old image targets."""
    def __init__(self, directory, *, index_name='manifest.json', split_override=None):
        from pathlib import Path
        from threading import RLock
        self.directory = Path(directory)
        self.index_name, self.split_override = index_name, split_override
        self.lock = None
        self.benchmark = None
        self.pipeline = None
        self._progress = None
        self._mutex = RLock()

    def open(self):
        import fcntl
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory.parent / f'.{self.directory.name}.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError('Teacher cache is already in use') from None

    def read_manifest(self):
        import json
        try:
            return json.loads((self.directory / self.index_name).read_text())
        except (OSError, ValueError):
            return {}

    def fingerprint(self, teacher_digest, train_loader, val_loader, config, device):
        import hashlib, json, inspect
        from pathlib import Path
        from . import temporal, dataloader
        from mlxdlss import model as backend, features
        from model import teacher as wrapper
        import teacher_infer
        files = []
        for split, loader in (('train', train_loader), ('val', val_loader)):
            for scene, paths in loader.dataset.clips:
                for path in paths:
                    for item in (path, *(loader.dataset.root / loader.dataset.source / name / scene / (path.stem + suffix)
                        for name, suffix in (('flow', '.flo'), ('invalid', '.png'), ('occlusions', '.png')))):
                        if item.is_file():
                            stat = item.stat()
                            files.append((split, str(item), stat.st_size, stat.st_mtime_ns))
        payload = {'format': 'sequence-targets-v3', 'teacher': teacher_digest,
            'rendering': config['rendering'], 'crop': config['training']['crop_size'],
            'data': files, 'dataset': config['dataset'], 'device': str(device), 'torch': torch.__version__,
            'benchmark': config['benchmark'],
            'source': {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (temporal, dataloader, backend, features, wrapper, teacher_infer)},
            'capture': hashlib.sha256(inspect.getsource(FeatureRecordingBackend).encode()).hexdigest()}
        if device.type == 'cuda':
            payload['gpu_uuid'] = str(torch.cuda.get_device_properties(device).uuid)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def reuse(self, fingerprint):
        manifest = self.read_manifest()
        self.reason = 'new sequence/profile signature or incomplete extraction'
        if manifest.get('fingerprint') != fingerprint or not manifest.get('complete'):
            return False
        files = manifest.get('files', {})
        if not files or any(not self.valid_file(name, size) for name, size in files.items()):
            self.reason = 'missing or incomplete sequence target'
            return False
        self.benchmark = manifest.get('benchmark')
        return True

    def valid_file(self, name, size):
        from pathlib import Path
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 6:
            return False
        path = self.directory / relative
        return path.is_file() and not path.is_symlink() and path.stat().st_size == size

    def prepare(self, fingerprint, blend_scale):
        manifest = self.read_manifest()
        compatible = manifest.get('fingerprint') == fingerprint
        self._progress = {'format': 'sequence-targets-v3', 'fingerprint': fingerprint,
            'blend_scale': blend_scale, 'files': manifest.get('files', {}) if compatible else {}}
        self.benchmark = manifest.get('benchmark') if compatible else None
        self.write_manifest(complete=False)

    def write_manifest(self, *, complete):
        import json
        with self._mutex:
            temporary = self.directory / (self.index_name + '.tmp')
            temporary.write_text(json.dumps({**self._progress, 'complete': complete,
                'benchmark': self.benchmark, 'pipeline': self.pipeline}, sort_keys=True))
            temporary.replace(self.directory / self.index_name)

    def path(self, split, sample_id):
        from .temporal import PROFILES
        profile, clip_id, position, frame_id = sample_id.split(':')
        source, render_pass, scene, frame = frame_id.split('/')
        if profile not in PROFILES or split not in ('train', 'val') or render_pass != 'clean':
            raise ValueError('Invalid cache target identity')
        if any(part in ('', '.', '..') or '\\' in part for part in (scene, frame)):
            raise ValueError('Invalid cache path')
        return self.directory / profile / (self.split_override or split) / source / scene / frame / 'targets.pt'

    def finished(self, split, key):
        path = self.path(split, key)
        name = path.relative_to(self.directory).as_posix()
        size = self._progress['files'].get(name)
        return size is not None and self.valid_file(name, size)

    def load_payload(self, split, key):
        payload = torch.load(self.path(split, key), map_location='cpu', weights_only=True)
        profile, clip_id, position, sample_id = key.split(':')
        expected = (profile, clip_id, int(position), sample_id)
        actual = tuple(payload.get(k) for k in ('profile', 'clip_id', 'frame_index', 'sample_id'))
        if actual != expected:
            raise ValueError('Cached target identity differs from the requested clip/frame/profile')
        return payload

    def store(self, split, key, payload):
        path = self.path(split, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile, clip_id, position, frame_id = key.split(':')
        payload = {**payload, 'profile': profile, 'clip_id': clip_id, 'frame_index': int(position), 'sample_id': frame_id}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        with self._mutex:
            self._progress['files'][path.relative_to(self.directory).as_posix()] = path.stat().st_size
            self.write_manifest(complete=False)

    def seal(self, fingerprint, *, blend_scale=None):
        self.write_manifest(complete=True)

    def build(self, teacher, train_loader, val_loader, config, *, monitor=None):
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        from itertools import islice
        from time import perf_counter
        from tqdm.auto import tqdm
        from .temporal import crop_rgb, reset_history, teacher_target
        from .distillation import make_frame, cache_key
        from .monitoring import measure_forward
        if teacher.device.type == 'cuda' and hasattr(getattr(teacher, 'model', None), 'enable_cuda_graph'):
            teacher.model.enable_cuda_graph()
        pending = deque()
        plan = None
        writers = None
        for split, loader in (('train', train_loader), ('val', val_loader)):
            dataset = loader.dataset
            indices = []
            for i in range(len(dataset)):
                identity = dataset.metadata(i)
                if not all(self.finished(split, cache_key(identity, j)) for j in range(len(identity['sample_ids']))):
                    indices.append(i)
            tqdm.write(f'[Cache {split}]: profiles={len(dataset.profiles)} | reuse clips={len(dataset) - len(indices)}/{len(dataset)} | extract={len(indices)}')
            if not indices:
                continue
            def extract(sample, read_seconds=0.):
                nonlocal plan
                history = previous = None
                for position, image in enumerate(sample['images']):
                    rgb = crop_rgb(image, config['training']['crop_size'])
                    if reset_history(previous, rgb, config):
                        history = None
                    key = cache_key(sample, position)
                    if self.finished(split, key):
                        payload = self.load_payload(split, key)
                    else:
                        started = perf_counter()
                        frame = make_frame(sample, position, config, history, rgb=rgb)
                        prepared_seconds = perf_counter() - started + read_seconds / len(sample['images'])
                        started = perf_counter()
                        payload = teacher_target(teacher, frame)
                        infer_seconds = perf_counter() - started
                        payload['history'] = torch.from_numpy(payload['history'].copy())
                        if plan is None:
                            started = perf_counter()
                            self.store(split, key, payload)
                            write_seconds = perf_counter() - started
                            target_bytes = sum(v.numel() * v.element_size() for v in
                                (payload['head'], payload['rgb'], *payload['features'].values()))
                            plan = cache_worker_plan(prepared_seconds, infer_seconds, write_seconds,
                                frame.inputs.numel() * 4, target_bytes, len(dataset))
                            plan['prefetch'] = min(plan['prefetch'], 2)
                            self.pipeline = plan
                            tqdm.write(f'[Cache workers]: read/prepare={plan["readers"]}, write={plan["writers"]}, prefetch={plan["prefetch"]}, pending_writes={plan["pending_writes"]}')
                        elif writers is None:
                            self.store(split, key, payload)
                        else:
                            pending.append(writers.submit(self.store, split, key, payload))
                            if len(pending) >= plan['pending_writes']:
                                pending.popleft().result()
                        if self.benchmark is None and config['benchmark']['enabled']:
                            timing = config['benchmark']
                            self.benchmark = {f'teacher_{k}': v for k, v in measure_forward(teacher,
                                frame.inputs.to(teacher.device), warmup=timing['warmup'], repeats=timing['repeats']).items()}
                    history = payload['history'].numpy() if config['rendering']['temporal'] else None
                    previous = rgb
            groups = {}
            for index in indices:
                groups.setdefault(dataset.items[index][0], []).append(index)
            first_group, *remaining = groups.values()
            writers = None
            # Decode one physical clip once, then reuse its RGB/flow for every profile.
            with torch.inference_mode():
                started = perf_counter()
                first_sample = dataset[first_group[0]]
                extract(first_sample, perf_counter() - started)
                with ThreadPoolExecutor(max_workers=plan['readers'] if plan else 1) as readers, \
                     ThreadPoolExecutor(max_workers=plan['writers'] if plan else 1) as writers:
                    source = iter(remaining)
                    capacity = max(1, min((plan or {}).get('prefetch', 1), 2))
                    ready = deque((group, readers.submit(dataset.__getitem__, group[0]))
                                  for group in islice(source, capacity))
                    bar = tqdm(total=len(indices), initial=1, desc=f'Teacher cache [{split}]', unit='clip',
                               disable=not config['logging']['progress'], dynamic_ncols=True)
                    try:
                        for index in first_group[1:]:
                            extract({**first_sample, **dataset.metadata(index)})
                            bar.update(1)
                        del first_sample
                        while ready:
                            group, future = ready.popleft()
                            sample = future.result()
                            following = next(source, None)
                            if following is not None:
                                ready.append((following, readers.submit(dataset.__getitem__, following[0])))
                            for index in group:
                                extract({**sample, **dataset.metadata(index)})
                                bar.update(1)
                                if monitor is not None:
                                    bar.set_postfix({'GPU': monitor.display()})
                        while pending:
                            pending.popleft().result()
                    finally:
                        bar.close()

    def close(self, *, delete=True):
        if self.lock is not None:
            try:
                if delete:
                    import re
                    paths = list(self.directory.rglob('*'))
                    for path in paths:
                        if path.is_symlink():
                            raise ValueError('Unexpected cache symlink; refusing deletion')
                        if path.is_file() and path.name not in ('targets.pt', 'targets.pt.tmp', 'manifest.json', 'manifest.json.tmp', 'evaluation.json', 'evaluation.json.tmp') and not re.fullmatch(r'(?:train|val)-[0-9a-f]{64}\.pt(?:\.tmp)?', path.name):
                            raise ValueError('Unexpected cache file; refusing deletion')
                    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
                        path.unlink() if path.is_file() else path.rmdir()
            finally:
                self.lock.close()
                self.lock = None


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
