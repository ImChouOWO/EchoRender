"""Train RGB super-resolution with supervised deblurring and denoising stages."""

import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
from model.student import EchoRenderSR
from model.card.loss import RestorationLoss
from tool.training_config import (
    load_training_config,
    DEFAULT_TRAIN_CONFIG,
    reserve_output_dir,
    project_path,
    PROJECT_ROOT,
)
from tool.dataloader import build_dataloader
from tool.metrics import ImageMetrics
from tool.trainer import fit
from tool.resume import read_checkpoint, validate_resume


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda":
        if (
            not torch.cuda.is_available()
            or (device.index or 0) >= torch.cuda.device_count()
        ):
            raise ValueError(f"Unavailable device: {device}")
        torch.cuda.set_device(device)
    return device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_training_config(args.config)
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache/torch"))
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        seed = cfg["loader"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = resolve_device(cfg["runtime"]["device"])
        generator = torch.Generator().manual_seed(seed)
        train_loader = build_dataloader(cfg, "train", generator)
        val_loader = build_dataloader(cfg, "val")
        model = EchoRenderSR(cfg["model"]["config"])
        if model.scale != cfg["dataset"]["scale"]:
            raise ValueError("Model and dataset scale must match")
        saved = None
        if args.resume:
            if cfg["model"]["weights"]:
                raise ValueError("Use --resume or model.weights, not both")
            path = project_path(args.resume)
            if not path.is_relative_to(PROJECT_ROOT / "output"):
                raise ValueError("Resume checkpoint must be inside output")
            saved = read_checkpoint(path)
            validate_resume(saved, cfg, model.config)
            output = path.parent
        else:
            if cfg["model"]["weights"]:
                model.load_state_dict(
                    read_checkpoint(cfg["model"]["weights"])["model"], strict=True
                )
            output = reserve_output_dir(cfg["runtime"]["output_dir"])
        cfg["runtime"]["output_dir"] = str(output)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), **cfg["training"]["optimizer"])
        criterion = RestorationLoss(cfg["loss"])
        metrics = ImageMetrics(cfg["validation"])
        print(
            f"Output: {output}\nTraining on {device}; train={len(train_loader.dataset)} clips, val={len(val_loader.dataset)} clips",
            flush=True,
        )
        print(
            f"REDS x{model.scale} | RGB + aligned history | Masked Attention SR -> NAFNet -> Restormer",
            flush=True,
        )
        print(
            f"Batch: effective={cfg['loader']['batch_size']} clips, micro={cfg['training']['micro_batch_size']} | precision={cfg['training']['precision']}",
            flush=True,
        )
        result = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            metrics,
            cfg,
            device,
            generator,
            saved,
        )
        if cfg["reporting"]["enabled"]:
            from tool.prediction_report import export_predictions

            model.load_state_dict(read_checkpoint(result["best_checkpoint"])["model"])
            export_predictions(
                model,
                val_loader,
                cfg,
                Path(result["report_dir"]) / "val",
                device,
                metrics,
            )
            if cfg["dataset"]["splits"].get("test"):
                export_predictions(
                    model,
                    build_dataloader(cfg, "test"),
                    cfg,
                    Path(result["report_dir"]) / "test",
                    device,
                    metrics,
                )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
