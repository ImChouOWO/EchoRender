"""Recovered teacher adapter; safetensors alone does not define a forward graph."""
from torch import nn

class EchoRenderTeacher(nn.Module):
    def __init__(self, weights, device='cpu'):
        super().__init__()
        try:
            from mlxdlss.model import load_model
        except ImportError as exc:
            raise ImportError('Install the recovered MLX-DLSS implementation exposing mlxdlss.model.load_model. Use dlssnr-logical.safetensors, not packed weights.') from exc
        self.model = load_model(str(weights)).to(device).eval()
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Teacher stays in evaluation mode even inside a training container.
        super().train(False)
        return self

    def forward(self, x):
        return self.model(x)

Teacher = EchoRenderTeacher
