"""Sixteen-channel RGB feature Student with an independent RGB motion branch."""
from .model import Model, DEFAULT_CONFIG
import torch
from .motion import RaftSmallMotion
from .card.card import InputAdapter

STAGES = ('InputAdapter', 'E0', 'E1', 'E2', 'E3', 'E4', 'VIT', 'D1', 'D2', 'D3')
HEAD = 'OutputHead'
COMPOSITION = 'RGB composition'
DEFAULT_STAGES = (COMPOSITION, HEAD, 'E2', 'VIT', 'D1', 'D3')


class EchoRenderStudent(Model):
    def __init__(self, config=DEFAULT_CONFIG, *, adapter_size=None, distillation_layers=None, precision=None):
        super().__init__(config)
        if self.input_channels != 16 or self.channels[-1] != 4:
            raise ValueError('Student requires 16 prepared channels and a four-channel head')
        stages = self.config['stages']
        if (not isinstance(stages, dict) or set(stages) != set((*STAGES, HEAD))
                or any(type(i) is not int or not 0 <= i < len(self.layers) for i in stages.values())
                or len(set(stages.values())) != len(stages)
                or stages.get('InputAdapter') != 0 or stages.get(HEAD) != len(self.layers) - 1
                or any(stages[name] == len(self.layers) - 1 for name in STAGES)):
            raise ValueError('Model stages require unique InputAdapter/E0-E4/VIT/D1-D3/OutputHead indices')
        size = adapter_size if adapter_size is not None else self.config.get('adapter_size', 320)
        selected = list(distillation_layers if distillation_layers is not None else
                        self.config.get('distillation_layers', DEFAULT_STAGES))
        if type(size) is not int or size < 1:
            raise ValueError('adapter_size must be a positive integer')
        if not selected or len(selected) != len(set(selected)) or any(n not in (*stages, COMPOSITION) for n in selected):
            raise ValueError('distillation_layers must contain unique model stage names')
        if not isinstance(self.layers[0], InputAdapter) or not self.layers[0].has_attention:
            raise ValueError('Student must start with a Swin InputAdapter')
        self.adapter_size = self.layers[0].size = size
        self.feature_layers = {name: stages[name] for name in selected if name in STAGES}
        self.precision = precision if precision is not None else self.config.get('precision', 'float32')
        if self.precision not in ('float32', 'bf16'):
            raise ValueError('precision must be float32 or bf16')
        self.config.update(adapter_size=size, distillation_layers=selected, precision=self.precision)
        self.motion_mode = self.config['motion']
        if self.motion_mode not in ('raw', 'raft_small'):
            raise ValueError('Student motion must be raw or raft_small')
        self.motion = RaftSmallMotion(size) if self.motion_mode == 'raft_small' else None

    def autocast(self):
        device = next(self.parameters()).device
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=self.precision == 'bf16' and device.type == 'cuda')

    def forward(self, x, return_features=False):
        with self.autocast():
            result = super().forward(x, return_features=return_features)
        if not return_features:
            return result
        head, outputs = result
        return head, {name: outputs[index] for name, index in self.feature_layers.items()}
