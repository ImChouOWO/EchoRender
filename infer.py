"""Run the recovered Teacher on one image, without a DataLoader or training loop."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from model.teacher import EchoRenderTeacher
from tool.teacher_pipeline import TeacherPipeline, load_rgb, save_rgb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image")
    parser.add_argument("--weights", type=Path, required=True, help="Decoded logical safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Output RGB PNG")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--noise-index", type=int, default=0, help="Deterministic noise index; no history is retained")
    parser.add_argument("--intensity", type=float, default=1.0, help="Effect blend, within [0,1]")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--report", type=Path, help="Optional JSON timings and image statistics")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs")
    args = parser.parse_args(argv)
    try:
        if args.cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        if args.noise_index < 0 or not np.isfinite(args.intensity) or not 0 <= args.intensity <= 1:
            raise ValueError("noise-index must be nonnegative and intensity must be within [0,1]")
        if args.output.suffix.lower() != ".png":
            raise ValueError("output must use .png")
        outputs = [args.output] + ([args.report] if args.report else [])
        protected = {args.input.resolve(), args.weights.resolve()}
        if any(path.resolve() in protected for path in outputs):
            raise ValueError("Output/report paths must differ from the input image and weights")
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ValueError("Output image and report must use different paths")
        for path in outputs:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {path}; choose a new path or use --overwrite")
        torch.set_num_threads(args.cpu_threads)
        image = load_rgb(args.input)
        started = perf_counter()
        print("Loading decoded Teacher weights...", flush=True)
        teacher = EchoRenderTeacher(args.weights, device=args.device)
        load_seconds = perf_counter() - started
        print(f"Running Teacher on {teacher.device}: {image.shape[1]}x{image.shape[0]}...", flush=True)
        result = TeacherPipeline(teacher).enhance(image, noise_index=args.noise_index, intensity=args.intensity)
        save_rgb(result.image, args.output)
        difference = np.abs(result.image - image)
        report = {
            "backend": "MLX-DLSS PyTorch recovered reference",
            "backend_revision": "7870371e8bf6c03967213f1a5d40db6c5a3bee38",
            "weights_format": teacher.metadata.get("format"),
            "weight_tensors": teacher.tensor_count,
            "device": str(teacher.device), "torch": torch.__version__,
            "input": str(args.input), "output": str(args.output),
            "image_size": [image.shape[1], image.shape[0]],
            "network_extent_hw": list(result.network_extent),
            "noise_index": args.noise_index, "intensity": args.intensity,
            "finite": bool(np.isfinite(result.image).all()),
            "output_min": float(result.image.min()), "output_max": float(result.image.max()),
            "mean_absolute_change": float(difference.mean()),
            "max_absolute_change": float(difference.max()),
            "seconds": {"load_weights": load_seconds, **result.timings},
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Teacher inference failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
