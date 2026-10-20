"""Epoch loop, checkpoints and metric persistence for train.py."""
import json
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm

from .distillation import train_batch, evaluate, module_digest, StudentAugmentation
from .training_config import TrainingSchedule
from .monitoring import GPUMonitor


def json_safe(value):
    """Represent infinite perfect-match PSNR as a JSON string, not invalid JSON."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def load_student_state(student, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != 'echorender-ssm-v2':
        raise ValueError('Expected an echorender-ssm-v2 checkpoint; legacy checkpoints are unsupported')
    student.load_state_dict(payload['student'], strict=True)


def save_checkpoint(path, student, criterion, optimizer, *, epoch, step, config, controls):
    payload = {
        'format': 'echorender-ssm-v2', 'student': student.state_dict(),
        'distillation': criterion.state_dict(), 'optimizer': optimizer.state_dict(),
        'student_config': student.config, 'train_config': config,
        'epoch': epoch, 'global_step': step, 'schedule_controls': controls,
    }
    temporary = path.with_suffix('.pt.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(teacher, student, criterion, train_loader, val_loader, optimizer, metrics, config, *, schedule=None):
    if any(value.requires_grad for value in list(teacher.parameters()) + list(teacher.buffers())):
        raise ValueError('Teacher must be frozen')
    expected = {id(p) for module in (student, criterion) for p in module.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    if expected != actual:
        raise ValueError('Optimizer must own student AND feature adapter parameters only')
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Train and validation loaders must be nonempty')
    output = Path(config['runtime']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.jsonl', 'last.pt', 'summary.json', 'resolved_train.yaml'):
        if (output / name).exists():
            raise FileExistsError(f'{output / name} exists; choose a fresh output_dir')
    import yaml
    (output / 'resolved_train.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    teacher_before = module_digest(teacher)
    student_before = module_digest(student, parameters_only=True)
    monitor = GPUMonitor(teacher.device, config['logging']['gpu_poll_seconds'])
    total_steps = len(train_loader) * config['training']['epochs']
    if config['training']['max_steps'] is not None:
        total_steps = min(total_steps, config['training']['max_steps'])
    schedule = schedule or TrainingSchedule(config, total_steps=total_steps)
    augmentation = StudentAugmentation(config)
    global_step, images_seen, history = 0, 0, []
    started = perf_counter()
    if teacher.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(teacher.device)
    try:
        with (output / 'metrics.jsonl').open('x', encoding='utf-8') as stream:
            def record(row):
                stream.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
                stream.flush()
            for epoch in range(1, config['training']['epochs'] + 1):
                epoch_loss, epoch_images, batches = 0.0, 0, 0
                augmentation_probability = augmentation.probability(epoch)
                gpu_values = []
                bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]} [Train]',
                           unit='it', dynamic_ncols=True, disable=not config['logging']['progress'])
                try:
                    for batch in bar:
                        # Apply current LR and reserved future training controls.
                        controls = schedule.values(epoch=epoch, step=global_step)
                        for group in optimizer.param_groups:
                            group['lr'] = controls['lr']
                        result = train_batch(teacher, student, criterion, batch, optimizer,
                                             crop_size=config['training']['crop_size'], controls=controls,
                                             augmentation=augmentation,
                                             augmentation_probability=augmentation_probability)
                        global_step += 1
                        batches += 1
                        epoch_images += result['batch_size']
                        images_seen += result['batch_size']
                        epoch_loss += result['loss'] * result['batch_size']
                        gpu = monitor.sample()
                        if gpu is not None:
                            gpu_values.append(gpu)
                        record({'event': 'train', 'epoch': epoch, 'step': global_step, **result,
                                'lr': controls['lr'], 'grad_clip': controls['grad_clip'],
                                'loss_weights': controls['loss_weights'],
                                'augmentation_probability': augmentation_probability,
                                'gpu_utilization_percent': gpu})
                        bar.set_postfix(loss=f'{result["loss"]:.5f}', avg_loss=f'{epoch_loss / epoch_images:.5f}',
                                        lr=f'{controls["lr"]:.2e}', gpu=monitor.display())
                        maximum = config['training']['max_steps']
                        if maximum is not None and global_step >= maximum:
                            break
                finally:
                    bar.close()
                validation = evaluate(teacher, student, criterion, val_loader, metrics, monitor,
                                      config=config, epoch=epoch, controls=controls)
                if module_digest(teacher) != teacher_before:
                    raise RuntimeError('Teacher parameters/buffers changed')
                row = {'event': 'epoch', 'epoch': epoch, 'step': global_step,
                       'train_loss': epoch_loss / epoch_images, 'train_images': epoch_images,
                       'train_partial': batches < len(train_loader), 'validation': validation,
                       'gpu_utilization_percent': sum(gpu_values) / len(gpu_values) if gpu_values else None}
                history.append(row)
                record(row)
                quality = validation['image_mean']
                tqdm.write(f'Epoch {epoch}: loss={row["train_loss"]:.6f}, '
                           f'PSNR={quality["psnr"]:.3f}, SSIM={quality["ssim"]:.4f}, LPIPS={quality["lpips"]:.4f}')
                if validation['benchmark']:
                    timing = validation['benchmark']
                    tqdm.write(f'Forward latency: Teacher={timing["teacher_latency_ms"]:.2f} ms/image, '
                               f'Student={timing["student_latency_ms"]:.2f} ms/image')
                save_checkpoint(output / 'last.pt', student, criterion, optimizer, epoch=epoch,
                                step=global_step, config=config, controls=controls)
                maximum = config['training']['max_steps']
                if maximum is not None and global_step >= maximum:
                    break
            summary = {
                'status': 'completed', 'reference': 'teacher_rgb', 'global_step': global_step,
                'epochs_run': len(history), 'train_images_seen': images_seen,
                'student_parameters_updated': module_digest(student, parameters_only=True) != student_before,
                'teacher_unchanged': True, 'student_parameter_count': sum(p.numel() for p in student.parameters()),
                'adapter_parameter_count': sum(p.numel() for p in criterion.parameters()),
                'seconds': perf_counter() - started, 'history': history,
                'peak_cuda_allocated_mib': (torch.cuda.max_memory_allocated(teacher.device) / 2**20
                                            if teacher.device.type == 'cuda' else None),
            }
            # The validation metric can be infinite only for identical RGB pairs.
            (output / 'summary.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + '\n', encoding='utf-8')
            return summary
    finally:
        monitor.close()
