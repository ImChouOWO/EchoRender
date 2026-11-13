"""RGB-only sequential SR, deblurring and denoising network."""

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from .model import Model, DEFAULT_CONFIG
from .motion import MotionAlignment
from .card.restoration import ReconstructionHead
from .card.mamba import SSMStage


class EchoRenderSR(nn.Module):
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()
        import yaml

        self.config = (
            deepcopy(config)
            if isinstance(config, dict)
            else yaml.safe_load(Path(config).read_text())
        )
        if self.config["input_channels"] != 6:
            raise ValueError(
                "SR backbone requires current RGB + aligned historical RGB"
            )
        self.backbone = Model(self.config)
        self.motion = MotionAlignment(self.config["motion"])
        head = self.config["reconstruction"]
        self.head = ReconstructionHead(self.backbone.channels[-1], **head)
        self.scale = head["scale"]
        for module in self.modules():
            if isinstance(module, SSMStage):
                module.gradient_checkpointing = self.config["gradient_checkpointing"]

    def forward(self, rgb, previous=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected BCHW RGB input")
        aligned = self.motion(rgb, previous)
        prepared = torch.cat((rgb, aligned), 1)
        features = self.backbone(prepared)
        return self.head(features, rgb)
