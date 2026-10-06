"""Safe YOLO-style DAG builder; no eval and no arbitrary YAML imports."""
from pathlib import Path
from copy import deepcopy
import yaml
from torch import nn
from .card.card import MODULES

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'model.yaml'


class Model(nn.Module):
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()
        if isinstance(config, dict):
            cfg = deepcopy(config)
        else:
            with Path(config).open(encoding='utf-8-sig') as f:
                cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError('Model config must be a mapping')
        self.config = cfg
        self.input_channels = int(cfg.get('input_channels', 16))
        self.layout = cfg.get('input_layout', 'BHWC')
        if self.input_channels < 1 or self.layout not in ('BHWC', 'BCHW'):
            raise ValueError('Invalid input channels or layout')
        self.layers, self.sources, self.channels = nn.ModuleList(), [], []
        rows = cfg.get('backbone', []) + cfg.get('head', [])
        if not rows:
            raise ValueError('Model needs at least one layer')
        for i, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ValueError(f'Layer {i}: expected [from, repeats, module, args]')
            source, repeats, name, args = row
            if name not in MODULES or not isinstance(args, list):
                raise ValueError(f'Layer {i}: unknown module or invalid args: {name}')
            if type(repeats) is not int or repeats < 1:
                raise ValueError(f'Layer {i}: repeats must be a positive integer')
            refs = source if isinstance(source, list) else [source]
            if not refs or any(type(r) is not int or r >= i or (r < -1 and i + r < 0) for r in refs):
                raise ValueError(f'Layer {i}: invalid source {source}')
            cs = [self.input_channels if i == 0 and r == -1 else self.channels[i - 1 if r == -1 else r] for r in refs]
            if name in ('Add', 'Concat', 'Upsample'):
                if not isinstance(source, list) or len(refs) < 2 or repeats != 1:
                    raise ValueError(f'Layer {i}: merge/resize needs multiple inputs and repeats=1')
                if name == 'Upsample':
                    if len(cs) != 2 or len(args) != 1:
                        raise ValueError('Upsample expects two sources and [channels]')
                    c2 = int(args[0])
                    layer = MODULES[name](cs[0], c2)
                else:
                    if args or (name == 'Add' and len(set(cs)) != 1):
                        raise ValueError(f'Layer {i}: invalid merge channels/args')
                    c2 = cs[0] if name == 'Add' else sum(cs)
                    layer = MODULES[name]()
            else:
                if isinstance(source, list) or not args:
                    raise ValueError(f'Layer {i}: module needs one source and channel args')
                c2 = int(args[0])
                if c2 < 1:
                    raise ValueError('Channels must be positive')
                layers = [MODULES[name](cs[0] if n == 0 else c2, c2, *args[1:]) for n in range(repeats)]
                layer = layers[0] if repeats == 1 else nn.Sequential(*layers)
            self.layers.append(layer)
            self.sources.append(source)
            self.channels.append(c2)
        if self.channels[-1] != cfg.get('output_channels', 4):
            raise ValueError('Final channels do not match output_channels')

    def forward(self, x, return_features=False):
        axis = -1 if self.layout == 'BHWC' else 1
        if x.ndim != 4 or x.shape[axis] != self.input_channels or min(x.shape[1:3] if axis == -1 else x.shape[2:]) < 1:
            raise ValueError(f'Expected {self.layout} tensor with {self.input_channels} input channels')
        if self.layout == 'BHWC':
            x = x.permute(0, 3, 1, 2).contiguous()
        outputs = []
        for layer, source in zip(self.layers, self.sources):
            def resolve(r):
                return (outputs[-1] if outputs else x) if r == -1 else outputs[r]
            inputs = [resolve(r) for r in source] if isinstance(source, list) else resolve(source)
            outputs.append(layer(inputs))
        y = outputs[-1]
        if self.layout == 'BHWC':
            y = y.permute(0, 2, 3, 1).contiguous()
        return (y, outputs) if return_features else y
