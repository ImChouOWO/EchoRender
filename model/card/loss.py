"""Task losses: SR L1, deblur Charbonnier/edge, and final denoising L1."""

import torch
from torch import nn
from torch.nn import functional as F


def charbonnier(x, y, epsilon=1e-3):
    return torch.sqrt((x - y).square() + epsilon**2).mean()


def laplacian(image):
    kernel = image.new_tensor([0.05, 0.25, 0.4, 0.25, 0.05])
    kernel = (kernel[:, None] * kernel[None, :])[None, None].repeat(3, 1, 1, 1)

    def blur(x):
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), kernel, groups=3)

    low = blur(image)[..., ::2, ::2]
    up = torch.zeros_like(image)
    up[..., ::2, ::2] = low * 4
    return image - blur(up)


class RestorationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, predictions, targets):
        # Losses use unclamped FP32 RGB so out-of-range predictions retain gradients.
        p = {k: v.float() for k, v in predictions.items()}
        t = {k: v.float() for k, v in targets.items()}
        sr = F.l1_loss(p["sr"], t["sr"])
        deblur = charbonnier(p["deblur"], t["deblur"], self.config["epsilon"])
        edge = charbonnier(
            laplacian(p["deblur"]), laplacian(t["deblur"]), self.config["epsilon"]
        )
        denoise = F.l1_loss(p["output"], t["output"])
        total = (
            self.config["sr"] * sr
            + self.config["deblur"] * (deblur + self.config["edge"] * edge)
            + self.config["denoise"] * denoise
        )
        return total, {"sr": sr, "deblur": deblur, "edge": edge, "denoise": denoise}
