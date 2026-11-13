"""Established full-reference RGB/Y PSNR, SSIM and pretrained RGB LPIPS."""

import os
from pathlib import Path
import torch
from torch import nn
from pytorch_msssim import ssim


class ImageMetrics(nn.Module):
    def __init__(self, config):
        super().__init__()
        import lpips

        os.environ.setdefault(
            "TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache/torch")
        )

        self.perceptual = (
            lpips.LPIPS(net=config["lpips_net"], verbose=False)
            .eval()
            .requires_grad_(False)
        )
        self.crop = config["crop_border"]
        self.y = config["test_y_channel"]

    @torch.no_grad()
    def forward(self, prediction, target):
        prediction = prediction.float().clamp(0, 1)
        target = target.float().clamp(0, 1)
        if self.crop:
            c = self.crop
            prediction = prediction[..., c:-c, c:-c]
            target = target[..., c:-c, c:-c]
        if min(target.shape[-2:]) < 64:
            raise ValueError(
                "Metric evaluation requires at least 64x64 after border cropping"
            )
        p, t = prediction, target
        if self.y:
            coefficients = p.new_tensor([65.481, 128.553, 24.966])[None, :, None, None]
            p = (p * coefficients).sum(1, keepdim=True) / 255 + 16 / 255
            t = (t * coefficients).sum(1, keepdim=True) / 255 + 16 / 255
        mse = (p - t).square().flatten(1).mean(1)
        psnr = -10 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))
        structural = ssim(p, t, data_range=1, win_size=11, size_average=False)
        perceptual = (
            self.perceptual(prediction * 2 - 1, target * 2 - 1).flatten(1).mean(1)
        )
        return [
            {
                key: float(value[i])
                for key, value in [
                    ("psnr", psnr),
                    ("ssim", structural),
                    ("lpips", perceptual),
                ]
            }
            for i in range(len(prediction))
        ]
