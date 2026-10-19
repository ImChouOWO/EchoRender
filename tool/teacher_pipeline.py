"""Stateless, single-image Teacher preparation and head-to-RGB composition.

Feature layout/noise/geometry come from the pinned MLX-DLSS reference. Each call
starts from the current image in channels 7:10; no previous image is retained.
Only the standard style is exposed, so no additional style grade is required.
"""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
import torch
from mlxdlss.features import NetworkGeometry, PROFILES, half, make_features

from model.teacher import EchoRenderTeacher


@dataclass(frozen=True)
class PreparedImage:
    rgb: np.ndarray
    features: torch.Tensor
    geometry: NetworkGeometry


@dataclass(frozen=True)
class TeacherResult:
    image: np.ndarray
    raw_head: torch.Tensor
    network_extent: tuple[int, int]  # height, width
    timings: dict[str, float]


def load_rgb(path):
    """Read an image as oriented HWC RGB float32 in [0,1]."""
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255.0


def prepare_image(rgb, *, noise_index=0):
    """Prepare one image on CPU using the backend's matching reflected geometry."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Image must be nonempty HWC RGB")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Image must contain finite RGB values in [0,1]")
    if type(noise_index) is not int or noise_index < 0:
        raise ValueError("noise_index must be a nonnegative integer")
    rgb = np.array(rgb, copy=True, order="C")
    height, width = rgb.shape[:2]
    geometry = NetworkGeometry.vendor_aligned(width, height)
    features = make_features(rgb, geometry=geometry, frame_index=noise_index, **PROFILES["standard"])
    return PreparedImage(rgb, torch.from_numpy(features).unsqueeze(0), geometry)


def compose_image(head, prepared, *, blend_scale, intensity=1.0):
    """Compose all four head channels with the current-image history condition.

    predicted = clamp(colour + 0.25 * half(head.rgb), 0, 1)
    alpha = sigmoid(half(head.a)) * half(checkpoint blend_scale)
    composed = predicted + alpha * (decoded input history - predicted)
    RGB = clamp(colour + intensity * (composed - colour), 0, 1)

    The gate remains part of this independent image mapping; it does not create
    cross-image state. This is the first-frame gate contract, not temporal training.
    """
    if not np.isfinite(intensity) or not 0 <= intensity <= 1:
        raise ValueError("intensity must be finite and within [0,1]")
    if not np.isfinite(blend_scale) or not 0 <= blend_scale <= 1:
        raise ValueError("blend_scale must be finite and within [0,1]")
    if not isinstance(head, torch.Tensor) or tuple(head.shape) != (*prepared.features.shape[:-1], 4):
        raise ValueError("Head must be [1,network_height,network_width,4]")
    raw = head.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    if not np.isfinite(raw).all():
        raise ValueError("Head contains non-finite values")
    raw = prepared.geometry.crop(raw)
    history = prepared.geometry.crop(prepared.features[0, ..., 7:10].numpy()) * 8.0 + 0.5
    predicted = np.clip(prepared.rgb + half(raw[..., :3]) * np.float32(0.25), 0, 1)
    with np.errstate(over="ignore"):
        alpha = np.clip(1.0 / (1.0 + np.exp(-half(raw[..., 3:4]))) * half(blend_scale), 0, 1)
    composed = predicted + alpha * (history - predicted)
    output = np.clip(prepared.rgb + np.float32(intensity) * (composed - prepared.rgb), 0, 1)
    if not np.isfinite(output).all():
        raise RuntimeError("RGB composition produced non-finite values")
    return output.astype(np.float32)


def save_rgb(rgb, path):
    """Write a lossless 8-bit RGB PNG without resizing."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Output must use .png for lossless RGB output")
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Output must be finite HWC RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def compose_rgb_tensor(head, prepared, *, blend_scale):
    """Differentiable counterpart of compose_image, returning valid BCHW RGB.

    Forward retains the backend's FP16 rounding; backward uses a straight-through
    estimator for that rounding only. Clamp/sigmoid keep their actual gradients.
    """
    h, w = prepared.rgb.shape[:2]
    raw = head[:, :h, :w].float()
    def half_ste(value):
        return value + (value.half().float() - value).detach()
    colour = torch.as_tensor(prepared.rgb, device=head.device).unsqueeze(0)
    history = prepared.features[:, :h, :w, 7:10].to(head.device) * 8 + 0.5
    predicted = (colour + 0.25 * half_ste(raw[..., :3])).clamp(0, 1)
    scale = torch.as_tensor(blend_scale, device=head.device).half().float()
    alpha = torch.sigmoid(half_ste(raw[..., 3:])) * scale
    output = (predicted + alpha * (history - predicted)).clamp(0, 1)
    return output.permute(0, 3, 1, 2).contiguous()


class TeacherPipeline:
    def __init__(self, teacher: EchoRenderTeacher):
        self.teacher = teacher

    def enhance(self, rgb, *, noise_index=0, intensity=1.0):
        if not np.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("intensity must be finite and within [0,1]")
        started = perf_counter()
        prepared = prepare_image(rgb, noise_index=noise_index)
        prepared_at = perf_counter()
        head = self.teacher(prepared.features).detach().cpu()
        # CPU transfer synchronizes GPU work before timing and composition.
        inferred_at = perf_counter()
        image = compose_image(head, prepared, blend_scale=self.teacher.blend_scale, intensity=intensity)
        return TeacherResult(
            image, head, tuple(prepared.features.shape[1:3]),
            {"prepare": prepared_at - started, "inference": inferred_at - prepared_at,
             "compose": perf_counter() - inferred_at},
        )

    def enhance_file(self, path, **kwargs):
        return self.enhance(load_rgb(path), **kwargs)
