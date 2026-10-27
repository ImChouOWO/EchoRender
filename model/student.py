"""RGB-only Student with external render controls and detached RGB history."""
import torch
from torch import nn
from .model import Model, DEFAULT_CONFIG

CONTROL_DIM = 16

class EchoRenderStudent(Model):
    FEATURE_LAYERS = {'e1': 3, 'e2': 5, 'e3': 7, 'vit': 11}

    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__(config)
        if self.input_channels != 3 or self.channels[-1] != 4:
            raise ValueError('Student requires RGB input and a four-channel head')
        self.control_layers = nn.ModuleDict({str(i): nn.Linear(CONTROL_DIM, 2 * c)
            for i, c in enumerate(self.channels) if i in (0, 1, 3, 5, 7, 9, 11, 14, 17, 20, 23, 26)})
        self.history_projection = nn.Conv2d(3, self.channels[0], 1, bias=False)
        for layer in self.control_layers.values():
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.history_projection.weight, std=.01)

    def forward(self, x, *, controls, history=None, return_features=False):
        if controls.shape != (x.shape[0], CONTROL_DIM):
            raise ValueError(f'controls must be [batch,{CONTROL_DIM}]')
        rgb = x.permute(0, 3, 1, 2) if self.layout == 'BHWC' else x
        history = torch.zeros_like(rgb) if history is None else history
        if history.shape != rgb.shape:
            raise ValueError('History must be aligned BCHW RGB at the network extent')
        def condition(index, value):
            if index == 0:
                value = value + self.history_projection(history)
            key = str(index)
            if key in self.control_layers:
                scale, bias = self.control_layers[key](controls).chunk(2, dim=1)
                value = value * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
            return value
        result = super().forward(x, return_features=return_features, condition=condition)
        if not return_features:
            return result
        head, outputs = result
        return head, {name: outputs[index] for name, index in self.FEATURE_LAYERS.items()}
