"""Teacher feature distillation: online extraction or precomputed cache.

No teacher architecture is defined here. Pass an existing torch module, or a
custom extractor returning {layer_name: tensor}. Inputs must already be prepared
exactly as required by the teacher (noise/history/controls/padding included).
"""
from pathlib import Path
import hashlib
import json
import os
import uuid
from collections.abc import Mapping
import torch
from torch import nn


def _digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, Mapping):
            for key in sorted(x):
                h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(str(type(x)).encode())
            for item in x:
                visit(item)
        else:
            raise TypeError('Cache inputs support tensors and nested tensor containers only')
    visit(value)
    return h.hexdigest()


def file_sha256(path):
    """Use the actual checkpoint digest as teacher_id; reads once during setup."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class TorchFeatureExtractor:
    """Capture named module outputs, detached. Hooks are removed after every call.

Repeated module calls are rejected rather than silently choosing a feature.
A selector may convert a module's tuple/dict output into the intended tensor.
"""
    def __init__(self, teacher, layers, selectors=None, runner=None):
        if not isinstance(teacher, nn.Module):
            raise TypeError('Pass an nn.Module teacher, or use a custom extractor')
        self.teacher = teacher
        self.layers = tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError('layers must be nonempty and unique')
        self.modules = dict(teacher.named_modules())
        missing = set(self.layers) - self.modules.keys()
        if missing:
            raise ValueError(f'Unknown teacher layers: {sorted(missing)}. Inspect list_layers().')
        self.selectors = selectors or {}
        self.runner = runner or teacher

    @staticmethod
    def list_layers(teacher):
        return tuple(name for name, _ in teacher.named_modules() if name)

    def __call__(self, inputs):
        features, handles = {}, []
        # Restore every submodule's previous train/eval state, even on errors.
        states = [(module, module.training) for module in self.teacher.modules()]
        def hook(name):
            def capture(module, args, output):
                if name in features:
                    raise RuntimeError(f'Layer {name} executed multiple times; use a custom extractor')
                if name in self.selectors:
                    output = self.selectors[name](output)
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f'Layer {name} output is not a tensor; supply a selector')
                features[name] = output.detach().clone()
            return capture
        try:
            for name in self.layers:
                handles.append(self.modules[name].register_forward_hook(hook(name)))
            self.teacher.eval()
            with torch.no_grad():
                self.runner(inputs)
            missing = set(self.layers) - features.keys()
            if missing:
                raise RuntimeError(f'Selected layers were not executed: {sorted(missing)}')
            return features
        finally:
            for handle in handles:
                handle.remove()
            for module, training in states:
                module.training = training


class TeacherFeatures:
    """Modes: online (run teacher) and cached (read only, fail on cache miss).

precompute() writes caches explicitly; cached training never runs the teacher.
teacher_id must identify checkpoint AND preprocessing/backend configuration.
Cache items may be samples or fixed batches. Their identity and inputs must match.
"""
    VERSION = 1

    def __init__(self, extractor, layers, *, mode='online', cache_dir=None,
                 teacher_id, output_device=None):
        self.extractor = extractor
        self.layers = tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError('layers must be nonempty and unique')
        if not isinstance(teacher_id, str) or not teacher_id:
            raise ValueError('Supply a nonempty teacher_id identifying weights and preprocessing')
        self.output_device = output_device
        self.spec = {'version': self.VERSION, 'teacher_id': teacher_id, 'layers': list(self.layers)}
        namespace = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()
        self.cache_dir = Path(cache_dir) / namespace if cache_dir else None
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in ('online', 'cached'):
            raise ValueError('mode must be online or cached')
        if mode == 'cached' and self.cache_dir is None:
            raise ValueError('cached mode requires cache_dir')
        if mode == 'online' and self.extractor is None:
            raise ValueError('online mode requires an extractor')
        self.mode = mode
        return self

    def _path(self, sample_id):
        if self.cache_dir is None:
            raise ValueError('Set cache_dir before precomputing')
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError('sample_id must be a nonempty stable string')
        return self.cache_dir / (hashlib.sha256(sample_id.encode()).hexdigest() + '.pt')

    def _extract(self, inputs):
        if self.extractor is None:
            raise ValueError('Precomputation requires an extractor')
        with torch.no_grad():
            values = self.extractor(inputs)
        if not isinstance(values, Mapping) or set(values) != set(self.layers):
            raise ValueError('Extractor must return exactly the selected layer names')
        if any(not isinstance(v, torch.Tensor) for v in values.values()):
            raise TypeError('Features must be tensors; convert NumPy arrays in the custom extractor')
        return {name: values[name].detach().clone() for name in self.layers}

    def _deliver(self, values):
        return {k: v.detach().to(self.output_device) if self.output_device is not None else v.detach()
                for k, v in values.items()}

    def get(self, inputs, *, sample_id=None):
        if self.mode == 'online':
            return self._deliver(self._extract(inputs))
        path = self._path(sample_id)
        if not path.is_file():
            raise FileNotFoundError(f'No cached features for {sample_id}; run precompute first')
        item = torch.load(path, map_location='cpu', weights_only=True)
        if item.get('spec') != self.spec or item.get('sample_id') != sample_id:
            raise ValueError('Cache metadata does not match this request')
        if item.get('input_digest') != _digest(inputs):
            raise ValueError('Input changed (augmentation/noise/history/controls); regenerate cache or use online mode')
        values = item['features']
        if set(values) != set(self.layers) or any(not isinstance(v, torch.Tensor) for v in values.values()):
            raise ValueError('Invalid cached feature tensors')
        return self._deliver(values)

    def precompute(self, samples, *, overwrite=False):
        """samples yields (stable_id, prepared_teacher_input). Returns written count.

Existing matching entries are reused; changed inputs fail unless overwrite=True.
Store CPU tensors at their original dtype with atomic replacement.
"""
        count = 0
        for sample_id, inputs in samples:
            path = self._path(sample_id)
            digest = _digest(inputs)
            if path.exists() and not overwrite:
                item = torch.load(path, map_location='cpu', weights_only=True)
                if item.get('spec') != self.spec or item.get('sample_id') != sample_id or item.get('input_digest') != digest:
                    raise ValueError(f'Existing cache differs for {sample_id}; use overwrite=True')
                continue
            features = {k: v.cpu() for k, v in self._extract(inputs).items()}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
            try:
                torch.save({'spec': self.spec, 'sample_id': sample_id,
                            'input_digest': digest, 'features': features}, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            count += 1
        return count
