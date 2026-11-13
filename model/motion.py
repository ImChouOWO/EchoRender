"""RGB-only backward motion and historical RGB alignment; no dataset flow input."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def warp(history, flow):
    b, _, h, w = history.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=history.device),
        torch.arange(w, device=history.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), 0).to(flow)[None] + flow
    valid = (
        (grid[:, 0] >= 0)
        & (grid[:, 0] <= w - 1)
        & (grid[:, 1] >= 0)
        & (grid[:, 1] <= h - 1)
    )[:, None]
    grid = torch.stack(
        (2 * (grid[:, 0] + 0.5) / w - 1, 2 * (grid[:, 1] + 0.5) / h - 1), -1
    )
    return (
        F.grid_sample(history, grid, align_corners=False, padding_mode="border") * valid
    )


class MotionAlignment(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("raw", "feat", "raft_small"):
            raise ValueError("motion must be raw, feat or raft_small")
        self.mode = mode
        if mode == "raft_small":
            import os
            from pathlib import Path
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            os.environ.setdefault(
                "TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache/torch")
            )
            self.estimator = (
                raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False)
                .eval()
                .requires_grad_(False)
            )
        if mode == "feat":
            self.predictor = nn.Sequential(
                nn.Conv2d(6, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 2, 3, padding=1),
            )
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, current, previous):
        if previous is None:
            return torch.zeros_like(current)
        if previous.shape != current.shape:
            raise ValueError("History must have the current LR RGB dimensions")
        if self.mode == "feat":
            flow = self.predictor(torch.cat((current, previous), 1))
        elif self.mode == "raft_small":
            h, w = current.shape[-2:]
            extent = (max(128, (h + 7) // 8 * 8), max(128, (w + 7) // 8 * 8))
            self.estimator.eval()
            with (
                torch.no_grad(),
                torch.autocast(device_type=current.device.type, enabled=False),
            ):
                now = F.interpolate(
                    current.float(), extent, mode="bilinear", align_corners=False
                )
                before = F.interpolate(
                    previous.float(), extent, mode="bilinear", align_corners=False
                )
                field = self.estimator(
                    now * 2 - 1, before * 2 - 1, num_flow_updates=12
                )[-1]
                self.estimator.corr_block.corr_pyramid = []
                flow = F.interpolate(
                    field, (h, w), mode="bilinear", align_corners=False
                )
                flow = (
                    flow
                    * flow.new_tensor([w / extent[1], h / extent[0]])[
                        None, :, None, None
                    ]
                )
        else:
            cv2.setNumThreads(1)
            arrays = []
            for now, before in zip(
                current.detach().float().cpu(), previous.detach().float().cpu()
            ):

                def gray(t):
                    return cv2.cvtColor(
                        np.uint8(t.permute(1, 2, 0).numpy().clip(0, 1) * 255),
                        cv2.COLOR_RGB2GRAY,
                    )

                arrays.append(
                    cv2.calcOpticalFlowFarneback(
                        gray(now), gray(before), None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                )
            flow = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(current)
        return warp(previous, flow)
