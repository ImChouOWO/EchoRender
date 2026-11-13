"""YOLO-style fixed-width progress with task losses and GPU telemetry."""

import torch
import sys
from tqdm.auto import tqdm

BAR_FORMAT = "{desc} {bar:10} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def progress_bar(loader=None, *, enabled, validation=False, total=None):
    columns = (
        ("Epoch", "GPU_mem", "GPU", "Loss", "PSNR", "SSIM", "LPIPS", "Images")
        if validation
        else ("Epoch", "GPU_mem", "GPU", "SR", "Deblur", "Denoise", "LR", "Clips")
    )
    if enabled:
        print("".join(f"{name:>11}" for name in columns), flush=True)
    return tqdm(
        loader,
        total=total,
        bar_format=BAR_FORMAT,
        dynamic_ncols=True,
        disable=not enabled,
        file=sys.stdout,
    )


def progress_row(epoch, epochs, device, monitor, values):
    device = torch.device(device)
    memory = (
        f"{torch.cuda.memory_reserved(device) / 2**30:.2f}G"
        if device.type == "cuda"
        else "0G"
    )
    return "".join(
        f"{str(value):>11}"
        for value in [f"{epoch}/{epochs}", memory, monitor.display(), *values]
    )


def highlight_best(message):
    import os, sys

    return (
        f"\033[1;93m{message}\033[0m"
        if sys.stdout.isatty() and "NO_COLOR" not in os.environ
        else message
    )
