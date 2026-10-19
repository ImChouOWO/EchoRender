"""Frozen recovered Teacher backed by the pinned MLX-DLSS PyTorch graph.

Input/output: BHWC float32, 16 prepared channels -> four raw head channels.
Image preparation and RGB composition live in tool.teacher_pipeline.
No NVIDIA kernel-level numeric equivalence is claimed.
"""
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def resolve_device(device="auto"):
    """Select CUDA when available; allow an explicit CPU/CUDA device."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Teacher supports 'auto', 'cpu', or 'cuda[:index]'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this environment")
    return device


class EchoRenderTeacher(nn.Module):
    def __init__(self, weights, device="auto"):
        super().__init__()
        self.weights_path = Path(weights).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Teacher weights not found: {self.weights_path}")
        target_device = resolve_device(device)
        with safe_open(str(self.weights_path), framework="pt", device="cpu") as source:
            self.metadata = source.metadata() or {}
            self.tensor_count = len(source.keys())
        if self.metadata.get("fully_logical") != "true":
            raise ValueError(
                "Teacher requires decoded logical weights (fully_logical=true). "
                "Use dlssnr-logical.safetensors, not dlssnr-packed.safetensors."
            )
        try:
            from mlxdlss.model import NeuralRenderingModel
            from mlxdlss.pipeline import load_weights
        except ImportError as exc:
            raise ImportError("Teacher backend is missing; run 'uv sync --locked'") from exc
        # Validate the complete tensor names/shapes and checkpoint format.
        self.model = NeuralRenderingModel(load_weights(self.weights_path))
        self.model.to(device=target_device, dtype=torch.float32)
        self.model.requires_grad_(False)
        self.eval()

    @property
    def device(self):
        return next(self.model.buffers()).device

    @property
    def blend_scale(self):
        value = self.model.blend_scale
        if value is None:
            raise ValueError("Logical checkpoint does not contain the head blend_scale")
        return value

    def train(self, mode=True):
        # Stay frozen/eval when a containing distillation model calls train().
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[-1] != 16
                or x.shape[0] < 1 or min(x.shape[1:3]) < 320
                or any(size % 64 for size in x.shape[1:3])):
            raise ValueError("Expected BHWC input [B,H,W,16], H/W >= 320 and multiples of 64")
        if not x.is_floating_point() or not torch.isfinite(x).all().item():
            raise ValueError("Teacher input must contain finite floating-point values")
        # The recovered graph owns its FP16/E4M3 rounding points.
        with torch.autocast(device_type=self.device.type, enabled=False):
            head = self.model(x.to(device=self.device, dtype=torch.float32))
        if head.shape != (*x.shape[:-1], 4) or not torch.isfinite(head).all().item():
            raise RuntimeError("Teacher returned an invalid or non-finite four-channel head")
        return head


Teacher = EchoRenderTeacher
