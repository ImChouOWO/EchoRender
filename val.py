"""Evaluate current SR weights and export degradation comparisons."""

import argparse
from copy import deepcopy
from pathlib import Path
import re
import torch
from train import resolve_device
from model.student import EchoRenderSR
from model.card.loss import RestorationLoss
from tool.resume import read_checkpoint
from tool.training_config import project_path, reserve_output_dir, PROJECT_ROOT
from tool.dataloader import build_dataloader
from tool.metrics import ImageMetrics
from tool.trainer import evaluate
from tool.prediction_report import export_predictions
from tool.reporting import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--project", default="validation")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.project):
            raise ValueError("Invalid project name")
        payload = read_checkpoint(project_path(args.weights))
        cfg = deepcopy(payload["train_config"])
        if args.max_images is not None:
            if args.max_images < 1:
                raise ValueError("--max-images must be positive")
            cfg["validation"]["max_images"] = args.max_images
        device = resolve_device(args.device or cfg["runtime"]["device"])
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        model = EchoRenderSR(payload["model_config"]).to(device)
        model.load_state_dict(payload["model"])
        loader = build_dataloader(cfg, args.split)
        metrics = ImageMetrics(cfg["validation"]).to(device)
        result = evaluate(
            model,
            loader,
            RestorationLoss(cfg["loss"]),
            metrics,
            cfg,
            device,
            payload["epoch"],
        )
        directory = reserve_output_dir(PROJECT_ROOT / "output/val" / args.project)
        atomic_json(
            directory / "validation.json",
            {
                "weights": str(args.weights),
                "split": args.split,
                "protocol": cfg["validation"],
                **result,
            },
        )
        export_predictions(model, loader, cfg, directory, device, metrics)
        print(
            f"Results: {directory}\nPSNR={result['psnr']:.3f} | SSIM={result['ssim']:.4f} | LPIPS={result['lpips']:.4f}"
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"Validation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
