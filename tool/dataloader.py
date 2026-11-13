"""Aligned REDS LR/HR clips. Random training patches cover the whole image."""

from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def read_rgb(path):
    with Image.open(path) as image:
        return (
            torch.from_numpy(np.array(image.convert("RGB"), copy=True))
            .permute(2, 0, 1)
            .float()
            / 255
        )


class REDSDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.epoch = 0
        d = config["dataset"]
        self.scale = d["scale"]
        self.patch = d["lr_patch_size"]
        self.seed = config["loader"]["seed"]
        self.max_frames = d["max_frames"]
        self.min_frames = d["min_frames"]
        settings = d["splits"].get(split)
        self.root = Path(d["root"])
        if settings is None:
            raise ValueError(
                f"No paired {split} split configured; official REDS test HR is not public"
            )
        source = settings["source"]
        hr_root = self.root / source / f"{source}_sharp"
        lr_root = self.root / source / f"{source}_sharp_bicubic" / f"X{self.scale}"
        # Official REDS extraction may put X4 above scenes or directly contain scenes.
        if not lr_root.is_dir():
            lr_root = lr_root.parent
        scenes = settings["scenes"] or sorted(
            p.name for p in hr_root.iterdir() if p.is_dir()
        )
        self.clips = []
        for scene in scenes:
            pairs = []
            for hr in sorted((hr_root / scene).glob("*.png")):
                lr = lr_root / scene / hr.name
                if not lr.is_file():
                    raise FileNotFoundError(f"Missing paired LR: {lr}")
                pairs.append((lr, hr))
            for i in range(0, len(pairs), self.max_frames):
                self.clips.append((scene, pairs[i : i + self.max_frames]))
        if not self.clips:
            raise ValueError(f"No paired REDS images found for {split}")
        selected = settings["scenes"]
        if selected is not None and any(
            not any(scene == item[0] for item in self.clips) for scene in selected
        ):
            raise ValueError(f"An explicitly selected {split} scene is empty")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        scene, pairs = self.clips[index]
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        if self.split == "train":
            length = rng.randint(min(self.min_frames, len(pairs)), len(pairs))
            offset = rng.randint(0, len(pairs) - length)
            pairs = pairs[offset : offset + length]
        lrs, hrs = [], []
        crop = None
        flip = rng.random() < 0.5 if self.split == "train" else False
        for lr_path, hr_path in pairs:
            lr, hr = read_rgb(lr_path), read_rgb(hr_path)
            h, w = lr.shape[-2:]
            if hr.shape[-2:] != (h * self.scale, w * self.scale):
                raise ValueError(f"Invalid x{self.scale} pair: {lr_path}, {hr_path}")
            if self.split == "train" and self.patch:
                if min(h, w) < self.patch:
                    raise ValueError("lr_patch_size exceeds an input dimension")
                if crop is None:
                    crop = (
                        rng.randint(0, h - self.patch),
                        rng.randint(0, w - self.patch),
                    )
                y, x = crop
                size = self.patch
                scale = self.scale
                lr = lr[:, y : y + size, x : x + size]
                hr = hr[
                    :, y * scale : (y + size) * scale, x * scale : (x + size) * scale
                ]
            if flip:
                lr = lr.flip(-1)
                hr = hr.flip(-1)
            lrs.append(lr)
            hrs.append(hr)
        return {
            "lr": torch.stack(lrs),
            "hr": torch.stack(hrs),
            "names": [f"{scene}/{p[1].stem}" for p in pairs],
        }


def build_dataloader(config, split="train", generator=None):
    dataset = REDSDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config["loader"]["batch_size"] if split == "train" else 1,
        shuffle=config["loader"]["train_shuffle"] if split == "train" else False,
        num_workers=config["loader"]["num_workers"],
        collate_fn=list,
        pin_memory=True,
        generator=generator,
    )
