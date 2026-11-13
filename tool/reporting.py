"""Persist standard metrics, task losses and training curves."""

import csv
import json
from pathlib import Path
import numpy as np

DIRECTIONS = {"loss": -1, "psnr": 1, "ssim": 1, "lpips": -1}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    temporary.replace(path)


def is_better(values, best, metric):
    return best is None or DIRECTIONS[metric] * (values[metric] - best[metric]) > 0


def write_training_report(history, directory, best_epoch, *, print_table=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    statistics = {}
    selected = next((row for row in history if row["epoch"] == best_epoch), history[-1])
    for key in DIRECTIONS:
        values = [row["validation"][key] for row in history]
        statistics[key] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "best": min(values) if DIRECTIONS[key] < 0 else max(values),
            "at_best_checkpoint": selected["validation"][key],
        }
    summary = {"best_epoch": best_epoch, "statistics": statistics, "history": history}
    atomic_json(directory / "summary.json", summary)
    with (directory / "metrics.csv").open("w", newline="") as f:
        rows = [
            {
                "epoch": r["epoch"],
                "train_loss": r["train"]["loss"],
                **r["validation"],
                "lr": r["lr"],
            }
            for r in history
        ]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    epochs = [row["epoch"] for row in history]
    for ax, key in zip(axes.flat, DIRECTIONS):
        ax.plot(epochs, [row["validation"][key] for row in history], label="Validation")
        if key == "loss":
            ax.plot(epochs, [row["train"]["loss"] for row in history], label="Training")
        ax.set(xlabel="Epoch", title=key.upper())
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(directory / "curves.png", dpi=160)
    plt.close(fig)
    if print_table:
        print(f"{'Metric':<10}{'Min':>12}{'Max':>12}{'Mean':>12}{'Best':>12}")
        for key, stats in statistics.items():
            print(
                f"{key.upper():<10}"
                + "".join(
                    f"{stats[name]:>12.5f}" for name in ("min", "max", "mean", "best")
                )
            )
    return summary
