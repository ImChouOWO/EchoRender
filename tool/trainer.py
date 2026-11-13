"""SR training with clip microbatches, exact per-update cosine and resume."""

import math
from pathlib import Path
import time
import torch
from .temporal import FORMAT, augmentation_ratio, synthesize, autocast, tiled_forward
from .monitoring import GPUMonitor
from .progress import progress_bar, progress_row, highlight_best
from .reporting import DIRECTIONS, is_better, write_training_report, atomic_json
from .resume import capture_rng, restore_rng, save_checkpoint, training_lock


def learning_rate(step, total, initial, minimum):
    if total == 1:
        return initial
    return (
        minimum + (initial - minimum) * (1 + math.cos(math.pi * step / (total - 1))) / 2
    )


@torch.inference_mode()
def evaluate(model, loader, criterion, metrics, config, device, epoch=0):
    model.eval()
    totals = {}
    count = 0
    monitor = GPUMonitor(device, config["logging"]["gpu_poll_seconds"])
    available = (
        sum(len(pairs) for _, pairs in loader.dataset.clips)
        if hasattr(loader.dataset, "clips")
        else None
    )
    maximum = config["validation"]["max_images"]
    expected = (
        min(available, maximum) if available is not None and maximum else available
    )
    bar = progress_bar(
        enabled=config["logging"]["progress"], validation=True, total=expected
    )
    started = time.perf_counter()
    try:
        for batch in loader:
            for clip in batch:
                previous = None
                for lr, hr in zip(clip["lr"], clip["hr"]):
                    if (
                        config["validation"]["max_images"]
                        and count >= config["validation"]["max_images"]
                    ):
                        break
                    lr, hr = lr[None].to(device), hr[None].to(device)
                    with autocast(model, config["training"]["precision"]):
                        prediction = tiled_forward(
                            model,
                            lr,
                            previous,
                            tile_size=config["validation"]["tile_size"],
                            overlap=config["validation"]["tile_overlap"],
                        )
                    loss, _ = criterion(
                        prediction, {k: hr for k in ("sr", "deblur", "output")}
                    )
                    scores = metrics(prediction["output"], hr)[0]
                    scores["loss"] = float(loss)
                    for key, value in scores.items():
                        totals[key] = totals.get(key, 0) + value
                    count += 1
                    previous = lr.detach()
                    bar.update(1)
                    avg = {k: v / count for k, v in totals.items()}
                    bar.set_description_str(
                        progress_row(
                            epoch,
                            config["training"]["epochs"],
                            device,
                            monitor,
                            [
                                f"{avg['loss']:.5f}",
                                f"{avg['psnr']:.2f}",
                                f"{avg['ssim']:.4f}",
                                f"{avg['lpips']:.4f}",
                                count,
                            ],
                        )
                    )
                if (
                    config["validation"]["max_images"]
                    and count >= config["validation"]["max_images"]
                ):
                    break
            if (
                config["validation"]["max_images"]
                and count >= config["validation"]["max_images"]
            ):
                break
    finally:
        bar.close()
        monitor.close()
    if not count:
        raise ValueError("Validation produced no images")
    return {
        **{k: v / count for k, v in totals.items()},
        "images": count,
        "seconds": time.perf_counter() - started,
    }


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    metrics,
    config,
    device,
    generator,
    resume=None,
):
    output = Path(config["runtime"]["output_dir"])
    model.to(device)
    criterion.to(device)
    metrics.to(device)
    epochs = config["training"]["epochs"]
    initial = config["training"]["optimizer"]["lr"]
    total = min(epochs * len(train_loader), config["training"]["max_steps"] or math.inf)
    step = 0
    history = []
    best = None
    best_epoch = None
    early = {"value": None, "bad_epochs": 0}
    start = 1
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        step = resume["step"]
        history = resume["history"]
        best = resume["best"]
        best_epoch = resume["best_epoch"]
        early = resume["early"]
        start = resume["epoch"] + 1
        if total != resume["total_steps"]:
            raise ValueError("Resume schedule length changed")
        restore_rng(resume["rng"])
        generator.set_state(resume["loader_rng"])
        if not (output / "best.pt").is_file():
            raise ValueError("Resume requires the original best.pt alongside last.pt")
    metric = config["reporting"]["best_metric"]
    early_cfg = config["training"]["early_stopping"]
    report = output.parent / "val" / output.name
    writer = None
    if config["logging"]["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            str(output / "tensorboard"), purge_step=start if resume else None
        )
    monitor = GPUMonitor(device, config["logging"]["gpu_poll_seconds"])
    try:
        with training_lock(output):
            for epoch in range(start, epochs + 1):
                if step >= total:
                    break
                train_loader.dataset.epoch = epoch
                model.train()
                sums = {}
                clips = 0
                started = time.perf_counter()
                ratio = augmentation_ratio(epoch, epochs, config["augmentation"])
                bar = progress_bar(
                    enabled=config["logging"]["progress"],
                    total=min(len(train_loader), total - step),
                )
                for batch in train_loader:
                    if step >= total:
                        break
                    lr = learning_rate(
                        step, total, initial, config["schedule"]["eta_min"]
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    # Select at most floor(batch * ratio); never augment an entire small batch.
                    selected = set(
                        torch.randperm(len(batch))[: int(len(batch) * ratio)].tolist()
                    )
                    micro = config["training"]["micro_batch_size"]
                    batch_values = {}
                    for offset in range(0, len(batch), micro):
                        subset = batch[offset : offset + micro]
                        prepared = []
                        for j, clip in enumerate(subset):
                            inputs, targets = synthesize(
                                clip["lr"].to(device),
                                clip["hr"].to(device),
                                ratio if offset + j in selected else 0.0,
                                config["augmentation"],
                            )
                            prepared.append((inputs, targets))
                        for t in range(max(len(item[0]) for item in prepared)):
                            valid = [
                                i for i, item in enumerate(prepared) if t < len(item[0])
                            ]
                            rgb = torch.stack([prepared[i][0][t] for i in valid])
                            previous = (
                                None
                                if t == 0
                                else torch.stack(
                                    [prepared[i][0][t - 1] for i in valid]
                                ).detach()
                            )
                            with autocast(model, config["training"]["precision"]):
                                predictions = model(rgb, previous)
                            # Each clip contributes equally, irrespective of its sampled frame count.
                            frame_loss = rgb.new_zeros(())
                            for k, i in enumerate(valid):
                                targets = {
                                    name: value[t : t + 1]
                                    for name, value in prepared[i][1].items()
                                }
                                loss, parts = criterion(
                                    {
                                        name: value[k : k + 1]
                                        for name, value in predictions.items()
                                    },
                                    targets,
                                )
                                factor = 1 / (len(batch) * len(prepared[i][0]))
                                frame_loss = frame_loss + loss * factor
                                for name, value in {"loss": loss, **parts}.items():
                                    batch_values[name] = (
                                        batch_values.get(name, 0)
                                        + float(value.detach()) * factor
                                    )
                            frame_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["grad_clip"]
                    ) if config["training"]["grad_clip"] else None
                    optimizer.step()
                    step += 1
                    clips += len(batch)
                    bar.update(1)
                    for name, value in batch_values.items():
                        sums[name] = sums.get(name, 0) + value * len(batch)
                    bar.set_description_str(
                        progress_row(
                            epoch,
                            epochs,
                            device,
                            monitor,
                            [
                                f"{batch_values['sr']:.4f}",
                                f"{batch_values['deblur']:.4f}",
                                f"{batch_values['denoise']:.4f}",
                                f"{lr:.2e}",
                                clips,
                            ],
                        )
                    )
                    if step >= total:
                        break
                bar.close()
                train_values = {k: v / clips for k, v in sums.items()}
                validation = evaluate(
                    model, val_loader, criterion, metrics, config, device, epoch
                )
                row = {
                    "format": FORMAT,
                    "epoch": epoch,
                    "step": step,
                    "train": train_values,
                    "validation": validation,
                    "lr": lr,
                    "augmentation_ratio": ratio,
                    "seconds": time.perf_counter() - started,
                }
                history.append(row)
                improved = is_better(validation, best, metric)
                if improved:
                    changes = (
                        "initial checkpoint"
                        if best is None
                        else " | ".join(
                            f"{k.upper()} {DIRECTIONS[k] * (validation[k] - best[k]) / max(abs(best[k]), 1e-12) * 100:+.2f}%"
                            for k in DIRECTIONS
                        )
                    )
                    print(
                        highlight_best(f"[Best]: save best weight | {changes}"),
                        flush=True,
                    )
                    best = {k: validation[k] for k in DIRECTIONS}
                    best_epoch = epoch
                score = DIRECTIONS[metric] * validation[metric]
                if (
                    early["value"] is None
                    or score > early["value"] + early_cfg["min_delta"]
                ):
                    early = {"value": score, "bad_epochs": 0}
                else:
                    early["bad_epochs"] += 1
                payload = {
                    "format": FORMAT,
                    "model": model.state_dict(),
                    "model_config": model.config,
                    "train_config": config,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "history": history,
                    "best": best,
                    "best_epoch": best_epoch,
                    "early": early,
                    "total_steps": total,
                    "rng": capture_rng(),
                    "loader_rng": generator.get_state(),
                }
                save_checkpoint(output / "last.pt", payload)
                if improved:
                    save_checkpoint(output / "best.pt", payload)
                atomic_json(output / "history.json", history)
                if config["reporting"]["enabled"]:
                    write_training_report(history, report, best_epoch)
                if writer:
                    for group, values in [
                        ("train", train_values),
                        ("validation", validation),
                    ]:
                        for key, value in values.items():
                            writer.add_scalar(f"{group}/{key}", value, epoch)
                    writer.add_scalar("schedule/lr", lr, epoch)
                    writer.add_scalar("schedule/augmentation", ratio, epoch)
                    writer.flush()
                print(
                    f"Epoch {epoch}: loss={validation['loss']:.5f}, PSNR={validation['psnr']:.3f}, SSIM={validation['ssim']:.4f}, LPIPS={validation['lpips']:.4f}",
                    flush=True,
                )
                if (
                    early_cfg["patience"]
                    and early["bad_epochs"] >= early_cfg["patience"]
                ):
                    print(
                        f"[Early stop]: no sufficient {metric.upper()} improvement for {early['bad_epochs']} epochs",
                        flush=True,
                    )
                    break
    finally:
        monitor.close()
        if writer:
            writer.close()
    if history:
        write_training_report(history, report, best_epoch, print_table=True)
    return {
        "best_checkpoint": str(output / "best.pt"),
        "report_dir": str(report),
        "history": history,
    }
