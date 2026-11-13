"""One degradation per full-resolution comparison, with separate error heatmaps."""

from pathlib import Path
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from .temporal import synthesize, tiled_forward, autocast
from .reporting import atomic_json


def image_array(tensor):
    return np.uint8(
        tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        * 255
        + 0.5
    )


def save_comparison(path, images, labels, title):
    arrays = [image_array(x) for x in images]
    h, w = arrays[0].shape[:2]
    top = 90
    gap = 16
    canvas = Image.new(
        "RGB", (w * len(arrays) + gap * (len(arrays) + 1), h + top + gap), "white"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default(size=24)
    draw.text((gap, 12), title, font=font, fill="black")
    for i, (array, label) in enumerate(zip(arrays, labels)):
        x = gap + (w + gap) * i
        draw.text((x, 50), label, font=font, fill="black")
        canvas.paste(Image.fromarray(array), (x, top))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_heatmap(path, prediction, target):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    p, t = prediction.detach().float(), target.detach().float()
    rgb = (p.clamp(0, 1) - t).abs().mean(1)[0].cpu().numpy()

    def gradient(x):
        gray = x.mean(1, keepdim=True)
        dx = F.pad(gray[..., 1:] - gray[..., :-1], (0, 1))
        dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-12)

    edge = (gradient(p) - gradient(t)).abs()[0, 0].cpu().numpy()
    arrays = []
    # Fixed scales make heatmaps comparable across strengths; these are diagnostics, not metrics.
    for values in (rgb, edge):
        color = np.uint8(colormaps["inferno"]((values / 0.2).clip(0, 1))[..., :3] * 255)
        arrays.append(torch.from_numpy(color.copy()).permute(2, 0, 1).float() / 255)
    save_comparison(
        path,
        arrays,
        ["RGB absolute error", "Gradient magnitude error"],
        "Error maps | fixed range 0.00-0.20 | darker is closer to GT",
    )


@torch.inference_mode()
def export_predictions(model, loader, config, directory, device, metrics):
    model.eval()
    directory = Path(directory)
    samples = []
    rng = random.Random(config["loader"]["seed"])
    indices = rng.sample(
        range(len(loader.dataset)),
        min(config["reporting"]["samples_per_mode"], len(loader.dataset)),
    )
    for i in indices:
        clip = loader.dataset[i]
        position = rng.randrange(len(clip["lr"]))
        samples.append((clip, position))
    conditions = [("original", 0.0)] + [
        (mode, float(s))
        for mode in ("noise", "blur", "combined")
        for s in config["reporting"]["strengths"]
    ]
    records = []
    bar = tqdm(
        total=len(samples) * len(conditions),
        desc="Export predictions",
        dynamic_ncols=True,
        disable=not config["logging"]["progress"],
    )
    try:
        for mode, strength in conditions:
            for clip, position in samples:
                generator = torch.Generator(device=device).manual_seed(
                    config["loader"]["seed"] + position
                )
                lr, hr = (
                    clip["lr"][: position + 1].to(device),
                    clip["hr"][: position + 1].to(device),
                )
                inputs, _ = synthesize(
                    lr,
                    hr,
                    strength,
                    config["augmentation"],
                    mode=mode,
                    generator=generator,
                )
                previous = None
                durations = []
                for frame in inputs:
                    rgb = frame[None]
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    with autocast(model, config["training"]["precision"]):
                        prediction = tiled_forward(
                            model,
                            rgb,
                            previous,
                            tile_size=config["validation"]["tile_size"],
                            overlap=config["validation"]["tile_overlap"],
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    durations.append(time.perf_counter() - start)
                    previous = rgb
                name = clip["names"][position].replace("/", "_")
                label = f"{mode}_{strength:.2f}"
                target = hr[position : position + 1]
                up = F.interpolate(
                    inputs[position : position + 1],
                    scale_factor=model.scale,
                    mode="bicubic",
                    align_corners=False,
                )
                scores = metrics(prediction["output"], target)[0]
                title = f"{mode.title()} {strength:.0%} | total infer {sum(durations):.3f} s | avg infer {sum(durations) / len(durations):.3f} s |"
                save_comparison(
                    directory / "pred" / mode / name / f"{label}.png",
                    [up, prediction["output"], target],
                    ["Input (bicubic)", "Prediction", "Ground truth"],
                    title,
                )
                save_comparison(
                    directory / "preview" / mode / name / f"{label}.png",
                    [up, target],
                    ["Input (bicubic)", "Ground truth"],
                    title,
                )
                save_heatmap(
                    directory / "heatmaps" / mode / name / f"{label}.png",
                    prediction["output"],
                    target,
                )
                records.append(
                    {
                        "image": name,
                        "mode": mode,
                        "strength": strength,
                        "history_frames": position + 1,
                        "total_infer_seconds": sum(durations),
                        "avg_infer_seconds": sum(durations) / len(durations),
                        **scores,
                    }
                )
                bar.update(1)
    finally:
        bar.close()
    atomic_json(directory / "predictions.json", records)
    return records
