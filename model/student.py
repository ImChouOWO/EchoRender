"""Configurable student model; initialize independently from teacher weights."""
from .model import Model

class EchoRenderStudent(Model):
    FEATURE_LAYERS = {'e1': 3, 'e2': 5, 'e3': 7, 'vit': 11}

    def forward(self, x, return_features=False):
        result = super().forward(x, return_features=return_features)
        if not return_features:
            return result
        head, outputs = result
        return head, {name: outputs[index] for name, index in self.FEATURE_LAYERS.items()}

Student = EchoRenderStudent
