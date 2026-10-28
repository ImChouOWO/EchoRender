"""RGB/history/style/tone Student with an independent RGB motion branch."""
from .model import Model, DEFAULT_CONFIG
from .motion import FeatureMotion


class EchoRenderStudent(Model):
    FEATURE_LAYERS = {'e1': 3, 'e2': 5, 'e3': 7, 'vit': 11}

    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__(config)
        if self.input_channels != 8 or self.channels[-1] != 4:
            raise ValueError('Student requires 8 prepared channels and a four-channel head')
        self.motion_mode = self.config['motion']
        if self.motion_mode not in ('raw', 'feat'):
            raise ValueError('Student motion must be raw or feat')
        self.motion = FeatureMotion() if self.motion_mode == 'feat' else None

    def forward(self, x, return_features=False):
        result = super().forward(x, return_features=return_features)
        if not return_features:
            return result
        head, outputs = result
        return head, {name: outputs[index] for name, index in self.FEATURE_LAYERS.items()}
