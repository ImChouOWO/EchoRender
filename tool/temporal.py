"""SR degradation synthesis and history-aware tiled inference."""

import math
import torch
from torch.nn import functional as F

FORMAT = "echorender-sr-v2"


def augmentation_ratio(epoch, epochs, config):
    if not config["enabled"]:
        return 0.0
    if epoch >= math.ceil(epochs * config["stop_fraction"]):
        return 0.0
    peak_epoch = max(1, math.ceil(epochs * config["peak_fraction"]))
    amount = min(1.0, (epoch - 1) / max(1, peak_epoch - 1))
    return (
        config["start_ratio"] + (config["max_ratio"] - config["start_ratio"]) * amount
    )


def gaussian_blur(image, sigma):
    if sigma <= 0:
        return image
    radius = max(1, math.ceil(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    horizontal = kernel[None, None, None, :].repeat(3, 1, 1, 1)
    vertical = kernel[None, None, :, None].repeat(3, 1, 1, 1)
    image = F.conv2d(
        F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=3
    )
    return F.conv2d(
        F.pad(image, (0, 0, radius, radius), mode="replicate"), vertical, groups=3
    )


def synthesize(lr, hr, strength, config, *, mode="combined", generator=None):
    """Blur clean HR; add LR noise and map that same realization to HR targets.

    The SR stage must preserve observable LR noise, not invent independent HR
    randomness destroyed by downsampling. Intermediate targets remain unclipped.
    """
    if strength == 0 or mode == "original":
        return lr, {"sr": hr, "deblur": hr, "output": hr}
    if mode not in ("noise", "blur", "combined"):
        raise ValueError("Degradation must be original, noise, blur or combined")
    scale = hr.shape[-1] / lr.shape[-1]
    sigma = (
        config["blur_sigma"] * strength * scale if mode in ("blur", "combined") else 0.0
    )
    std = config["noise_std"] * strength if mode in ("noise", "combined") else 0.0
    noise_lr = (
        torch.randn(lr.shape, device=lr.device, dtype=lr.dtype, generator=generator)
        * std
    )
    noise_hr = F.interpolate(
        noise_lr, size=hr.shape[-2:], mode="bilinear", align_corners=False
    )
    blurred = gaussian_blur(hr, sigma)
    base_lr = (
        F.interpolate(
            blurred,
            size=lr.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        if sigma
        else lr
    )
    return (base_lr + noise_lr).clamp(0, 1), {
        "sr": blurred + noise_hr,
        "deblur": hr + noise_hr,
        "output": hr,
    }


def autocast(model, precision):
    device = next(model.parameters()).device
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


def tiled_forward(model, rgb, previous, *, tile_size, overlap):
    h, w = rgb.shape[-2:]
    scale = model.scale
    if max(h, w) <= tile_size:
        return model(rgb, previous)
    step = tile_size - overlap

    def starts(length):
        return sorted(
            set(
                [
                    *range(0, max(1, length - tile_size + 1), step),
                    max(0, length - tile_size),
                ]
            )
        )

    output = {}
    weights = rgb.new_zeros(1, 1, h * scale, w * scale)
    for y in starts(h):
        for x in starts(w):
            block = rgb[..., y : y + tile_size, x : x + tile_size]
            history = (
                None
                if previous is None
                else previous[..., y : y + tile_size, x : x + tile_size]
            )
            values = model(block, history)
            bh, bw = values["output"].shape[-2:]
            sy, sx = y * scale, x * scale
            # Triangular overlap weighting reduces tile seams without changing pixel geometry.
            wy = torch.hann_window(
                bh + 2, periodic=False, device=rgb.device, dtype=torch.float32
            )[1:-1]
            wx = torch.hann_window(
                bw + 2, periodic=False, device=rgb.device, dtype=torch.float32
            )[1:-1]
            window = (wy[:, None] * wx[None, :]).clamp_min(1e-5)[None, None]
            weights[..., sy : sy + bh, sx : sx + bw] += window
            for key, value in values.items():
                if key not in output:
                    output[key] = value.new_zeros(
                        value.shape[0], 3, h * scale, w * scale, dtype=torch.float32
                    )
                output[key][..., sy : sy + bh, sx : sx + bw] += value.float() * window
    return {key: value / weights for key, value in output.items()}
