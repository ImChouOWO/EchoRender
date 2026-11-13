"""Strict SR configuration. Paths resolve from the project root."""

from copy import deepcopy
from pathlib import Path
import math
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "config/train.yaml"


def project_path(path):
    return (PROJECT_ROOT / Path(path).expanduser()).resolve()


def reserve_output_dir(path):
    path = Path(path)
    for index in range(1, 100000):
        candidate = path if index == 1 else path.with_name(f"{path.name}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            pass
    raise RuntimeError("Unable to reserve an output directory")


def load_training_config(path=DEFAULT_TRAIN_CONFIG):
    cfg = yaml.safe_load(project_path(path).read_text())
    expected = {
        "model",
        "runtime",
        "training",
        "loss",
        "schedule",
        "augmentation",
        "dataset",
        "loader",
        "validation",
        "reporting",
        "logging",
    }
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise ValueError(
            "Use the current SR config schema; legacy Teacher/distillation settings are unsupported"
        )
    cfg = deepcopy(cfg)
    allowed = {
        "model": {"config", "weights"},
        "runtime": {"device", "cpu_threads", "output_dir"},
        "training": {
            "epochs",
            "micro_batch_size",
            "max_steps",
            "precision",
            "optimizer",
            "grad_clip",
            "early_stopping",
        },
        "loss": {"sr", "deblur", "denoise", "edge", "epsilon"},
        "schedule": {"kind", "eta_min"},
        "augmentation": {
            "enabled",
            "start_ratio",
            "max_ratio",
            "peak_fraction",
            "stop_fraction",
            "blur_sigma",
            "noise_std",
        },
        "dataset": {
            "root",
            "scale",
            "lr_patch_size",
            "max_frames",
            "min_frames",
            "splits",
        },
        "loader": {"batch_size", "num_workers", "seed", "train_shuffle"},
        "validation": {
            "tile_size",
            "tile_overlap",
            "crop_border",
            "test_y_channel",
            "lpips_net",
            "max_images",
        },
        "reporting": {"enabled", "best_metric", "samples_per_mode", "strengths"},
        "logging": {"progress", "gpu_poll_seconds", "tensorboard"},
    }
    for section, keys in allowed.items():
        if set(cfg[section]) != keys:
            raise ValueError(f"Invalid {section} keys: expected {sorted(keys)}")
    for section, key in [
        ("model", "config"),
        ("runtime", "output_dir"),
        ("dataset", "root"),
    ]:
        cfg[section][key] = str(project_path(cfg[section][key]))
    if cfg["model"]["weights"]:
        cfg["model"]["weights"] = str(project_path(cfg["model"]["weights"]))
    out = Path(cfg["runtime"]["output_dir"])
    if not out.is_relative_to(PROJECT_ROOT / "output"):
        raise ValueError("runtime.output_dir must be inside EchoRender/output")
    if (
        cfg["dataset"]["scale"] not in (2, 4, 8)
        or not 1 <= cfg["dataset"]["min_frames"] <= cfg["dataset"]["max_frames"] <= 10
    ):
        raise ValueError("Invalid SR scale or frame count (1..10)")
    for section, key in [
        ("training", "epochs"),
        ("training", "micro_batch_size"),
        ("loader", "batch_size"),
        ("runtime", "cpu_threads"),
    ]:
        if type(cfg[section][key]) is not int or cfg[section][key] < 1:
            raise ValueError(f"{section}.{key} must be positive")
    if cfg["training"]["precision"] not in ("float32", "bf16"):
        raise ValueError("precision must be float32 or bf16")
    if (
        cfg["schedule"]["kind"] != "cosine"
        or not 0 < cfg["schedule"]["eta_min"] <= cfg["training"]["optimizer"]["lr"]
    ):
        raise ValueError("Expected cosine schedule with a positive eta_min <= LR")
    a = cfg["augmentation"]
    if (
        not 0 <= a["start_ratio"] <= a["max_ratio"] <= 0.25
        or not 0 < a["peak_fraction"] < a["stop_fraction"] <= 1
    ):
        raise ValueError("Invalid augmentation cap or epoch schedule")
    if not 0 < a["noise_std"] <= 1 or not 0 < a["blur_sigma"] <= 10:
        raise ValueError("Invalid noise or blur maximum")
    v = cfg["validation"]
    if not 0 <= v["tile_overlap"] < v["tile_size"] or v["crop_border"] < 0:
        raise ValueError("Invalid evaluation tile or crop")
    if cfg["reporting"]["best_metric"] not in ("psnr", "ssim", "lpips", "loss"):
        raise ValueError("Unknown best metric")
    if (
        any(
            not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0
            for x in cfg["loss"].values()
        )
        or cfg["loss"]["epsilon"] == 0
    ):
        raise ValueError("Loss weights must be finite/nonnegative and epsilon positive")
    optimizer = cfg["training"]["optimizer"]
    if (
        set(optimizer) != {"lr", "momentum", "weight_decay"}
        or optimizer["lr"] <= 0
        or optimizer["momentum"] < 0
        or optimizer["weight_decay"] < 0
    ):
        raise ValueError("Expected valid SGD lr, momentum and weight_decay")
    if cfg["loader"]["num_workers"] < 0 or cfg["dataset"]["lr_patch_size"] < 1:
        raise ValueError("num_workers must be nonnegative and lr_patch_size positive")
    if cfg["training"]["max_steps"] is not None and (
        type(cfg["training"]["max_steps"]) is not int
        or cfg["training"]["max_steps"] < 1
    ):
        raise ValueError("max_steps must be null or a positive integer")
    if cfg["validation"]["max_images"] is not None and (
        type(cfg["validation"]["max_images"]) is not int
        or cfg["validation"]["max_images"] < 1
    ):
        raise ValueError("max_images must be null or a positive integer")
    early = cfg["training"]["early_stopping"]
    if (
        set(early) != {"patience", "min_delta"}
        or type(early["patience"]) is not int
        or early["patience"] < 0
        or early["min_delta"] < 0
    ):
        raise ValueError("Invalid early stopping settings")
    reporting = cfg["reporting"]
    if (
        not reporting["strengths"]
        or any(not 0 < s <= 1 for s in reporting["strengths"])
        or reporting["samples_per_mode"] < 1
    ):
        raise ValueError(
            "Reporting requires positive sample count and strengths in (0,1]"
        )
    splits = cfg["dataset"]["splits"]
    if set(splits) != {"train", "val", "test"} or any(
        not isinstance(splits[name], dict) for name in ("train", "val")
    ):
        raise ValueError("Configure paired train/val splits; test may be null")
    return cfg
