"""Independent RGB-pair motion branch; backward flow uses normalized image units."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def raw_motion(previous, current):
    """Estimate current-to-previous flow outside the network, without dataset guides."""
    import cv2
    if previous.shape != current.shape or current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError('Motion requires matching HWC RGB frames')
    cv2.setNumThreads(1)
    height, width = current.shape[:2]
    scale = min(1., 512 / max(height, width))
    size = max(8, round(width * scale)), max(8, round(height * scale))
    def gray(rgb):
        image = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), size)
    before, now = gray(previous), gray(current)
    def estimate(a, b):
        return cv2.calcOpticalFlowFarneback(a, b, None, .5, 4, 21, 5, 7, 1.5, 0)
    backward, forward = estimate(now, before), estimate(before, now)
    yy, xx = np.indices(now.shape, dtype=np.float32)
    x, y = xx + backward[..., 0], yy + backward[..., 1]
    reverse = cv2.remap(forward, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    error = np.square(backward + reverse).sum(-1)
    threshold = .01 * (np.square(backward).sum(-1) + np.square(reverse).sum(-1)) + .5
    confidence = ((error <= threshold) & (x >= 0) & (x <= size[0] - 1)
                  & (y >= 0) & (y <= size[1] - 1)).astype(np.float32)
    backward /= np.array(size, np.float32)
    motion = cv2.resize(backward, (width, height), interpolation=cv2.INTER_LINEAR)
    confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    return motion, confidence[..., None]


class FeatureMotion(nn.Module):
    """RGB pair -> 32/64-channel features -> two offsets and one validity logit; no skips."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.decoder = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(32, 3, 3, padding=1))
        # Zero initial displacement; RGB supervision and flow loss train the entire branch.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, previous, current):
        if previous.shape != current.shape or current.ndim != 4 or current.shape[1] != 3:
            raise ValueError('Feature motion requires matching BCHW RGB frames')
        field = self.decoder(self.encoder(torch.cat((previous, current), dim=1)))
        height, width = field.shape[-2:]
        field = F.interpolate(field, current.shape[-2:], mode='bilinear', align_corners=False)
        units = field.new_tensor([width, height]).view(1, 2, 1, 1)
        return field[:, :2] / units, field[:, 2:3]


def warp_history(history, motion, confidence):
    """Differentiable backward sampling, with invalid and out-of-frame pixels rejected."""
    height, width = history.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=history.device),
                           torch.arange(width, device=history.device), indexing='ij')
    base = torch.stack(((xx + .5) / width, (yy + .5) / height), dim=-1).to(history)
    coordinates = base[None] + motion.permute(0, 2, 3, 1)
    valid = ((coordinates[..., 0] >= .5 / width) & (coordinates[..., 0] <= 1 - .5 / width)
             & (coordinates[..., 1] >= .5 / height) & (coordinates[..., 1] <= 1 - .5 / height))
    aligned = F.grid_sample(history, 2 * coordinates - 1, mode='bilinear',
                            padding_mode='border', align_corners=False)
    return aligned, confidence * valid[:, None]
