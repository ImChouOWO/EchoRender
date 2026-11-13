"""RGB-only single-image or ordered-frame SR inference."""

import argparse
from pathlib import Path
import torch
from PIL import Image
from train import resolve_device
from model.student import EchoRenderSR
from tool.resume import read_checkpoint
from tool.training_config import project_path, reserve_output_dir, PROJECT_ROOT
from tool.dataloader import read_rgb
from tool.temporal import tiled_forward, autocast
from tool.prediction_report import image_array


@torch.inference_mode()
def run_inference(
    weights, input_image, *, device="cuda:1", project="sr_inference", max_frames=10
):
    if not 1 <= max_frames <= 10:
        raise ValueError("max_frames must be between 1 and 10")
    if Path(project).name != project or project in (".", ".."):
        raise ValueError("Invalid project name")
    payload = read_checkpoint(project_path(weights))
    config = payload["train_config"]
    device = resolve_device(device)
    model = EchoRenderSR(payload["model_config"]).to(device).eval()
    model.load_state_dict(payload["model"])
    source = project_path(input_image)
    paths = (
        [source]
        if source.is_file()
        else sorted(
            p
            for p in source.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        )[:max_frames]
    )
    if not paths:
        raise ValueError("No input images found")
    output = reserve_output_dir(PROJECT_ROOT / "output/infer" / project)
    previous = None
    for path in paths:
        rgb = read_rgb(path)[None].to(device)
        if previous is not None and previous.shape != rgb.shape:
            previous = None
        with autocast(model, config["training"]["precision"]):
            predictions = tiled_forward(
                model,
                rgb,
                previous,
                tile_size=config["validation"]["tile_size"],
                overlap=config["validation"]["tile_overlap"],
            )
        Image.fromarray(image_array(predictions["output"])).save(
            output / f"{path.stem}.png"
        )
        previous = rgb
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--project", default="sr_inference")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()
    print(
        run_inference(
            args.weights,
            args.input,
            device=args.device,
            project=args.project,
            max_frames=args.max_frames,
        )
    )


if __name__ == "__main__":
    main()
