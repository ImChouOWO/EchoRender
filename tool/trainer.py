"""Epoch loop, checkpoints and metric persistence for train.py."""
import json
import gc
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm
from model.card.mamba import MambaStage

from .distillation import train_batch, evaluate, module_digest, StudentAugmentation
from .training_config import TrainingSchedule, PROJECT_ROOT
from .teacher_features import TeacherFeatureCache
from .temporal import FORMAT
from .progress import progress_bar, progress_row, network_size, profile_row, highlight_best
from .reporting import (validation_values, is_better, best_message, metric_statistics,
                        print_statistics, write_training_report, EarlyStopping)

FEATURE_CACHE_DIR = PROJECT_ROOT / "output/feats"
from .monitoring import GPUMonitor
from .resume import (capture_rng, restore_rng, restore_progress, validate_resume, training_lock, checkpoint_values)


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
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError(f'Expected a {FORMAT} checkpoint; retraining is required')
    if payload['student_config']['input_channels'] != 8:
        raise ValueError('Student checkpoint requires the prepared 8-channel layout')
    student.load_state_dict(payload['student'], strict=True)


def save_checkpoint(path, student, criterion, optimizer, *, epoch, step, config, controls, blend_scale, resume_state=None):
    from copy import deepcopy
    checkpoint_config = deepcopy(config)
    checkpoint_config['rendering']['profiles'] = {name: config['rendering']['profiles'][name]
        for name in config['training']['profiles']}
    payload = {
        'format': FORMAT, 'student': student.state_dict(),
        'distillation': criterion.state_dict(), 'optimizer': optimizer.state_dict(),
        'student_config': student.config, 'train_config': checkpoint_config,
        'epoch': epoch, 'global_step': step, 'schedule_controls': controls,
        'teacher_blend_scale': blend_scale, 'resume_state': checkpoint_values(resume_state),
    }
    temporary = path.with_suffix('.pt.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def move_student(student, criterion, optimizer, device):
    optimizer.zero_grad(set_to_none=True)
    student.to(device)
    criterion.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def release_cuda(device):
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def fit(teacher, student, criterion, train_loader, val_loader, optimizer, metrics, config, *, schedule=None, device=None, cache_cleanup=None, report_dir=None, on_complete=None, resume=None):
    with training_lock(Path(config['runtime']['output_dir'])):
        return _fit(teacher, student, criterion, train_loader, val_loader, optimizer, metrics, config,
                    schedule=schedule, device=device, cache_cleanup=cache_cleanup, report_dir=report_dir,
                    on_complete=on_complete, resume=resume)


def _fit(teacher, student, criterion, train_loader, val_loader, optimizer, metrics, config, *, schedule=None, device=None, cache_cleanup=None, report_dir=None, on_complete=None, resume=None):
    if resume is not None:
        validate_resume(resume, config, student)
        student.load_state_dict(resume['student'], strict=True)
        criterion.load_state_dict(resume['distillation'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
    for module in student.modules():
        if isinstance(module, MambaStage):
            module.gradient_checkpointing = config['training']['gradient_checkpointing']
    if not isinstance(teacher, torch.nn.Module):
        teacher = teacher()
    blend_scale = float(torch.as_tensor(teacher.blend_scale).detach().cpu().item())
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
    for name in (() if resume is not None else ('metrics.jsonl', 'best.pt', 'last.pt', 'summary.json', 'resolved_train.yaml')):
        if (output / name).exists():
            raise FileExistsError(f'{output / name} exists; choose a fresh output_dir')
    import yaml
    if resume is None:
        (output / 'resolved_train.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    teacher_before = module_digest(teacher)
    if resume is not None:
        expected_teacher = (resume.get('resume_state') or {}).get('teacher_digest')
        if expected_teacher is not None and teacher_before != expected_teacher:
            raise ValueError('Teacher weights differ from the resumed checkpoint')
    student_before = module_digest(student, parameters_only=True)
    device = torch.device(device) if device is not None else next(student.parameters()).device
    monitor = GPUMonitor(device, config['logging']['gpu_poll_seconds'])
    total_steps = len(train_loader) * config['training']['epochs']
    if config['training']['max_steps'] is not None:
        total_steps = min(total_steps, config['training']['max_steps'])
    if resume is not None and resume.get('resume_state'):
        total_steps = resume['resume_state']['total_steps']
    schedule = schedule or TrainingSchedule(config, total_steps=total_steps)
    augmentation = StudentAugmentation(config)
    global_step, images_seen, history = 0, 0, []
    started = perf_counter()
    previous_seconds = ((resume.get('resume_state') or {}).get('seconds', 0.) if resume else 0.)
    completed = False
    best, best_epoch = None, None
    best_metric = config['reporting']['best_metric']
    early = EarlyStopping(best_metric, **config['training']['early_stopping'])
    first_epoch, previous_rows = 1, []
    if resume is not None:
        previous_rows, history, best, best_epoch, images_seen = restore_progress(resume, output, early, best_metric)
        global_step, first_epoch = resume['global_step'], resume['epoch'] + 1
    stopped_early = False
    cache = TeacherFeatureCache(FEATURE_CACHE_DIR) if config['training']['releas'] else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if cache is not None:
            cache.open()
        else:
            teacher.to(device)
            move_student(student, criterion, optimizer, device)
        if resume is not None:
            temporary = output / 'metrics.jsonl.tmp'
            temporary.write_text(''.join(json.dumps(json_safe(row), allow_nan=False) + '\n' for row in previous_rows), encoding='utf-8')
            temporary.replace(output / 'metrics.jsonl')
        with (output / 'metrics.jsonl').open('a' if resume is not None else 'x', encoding='utf-8') as stream:
            def record(row):
                stream.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
                stream.flush()
            if cache is not None:
                move_student(student, criterion, optimizer, 'cpu')
                metrics.cpu()
                release_cuda(device)
                fingerprint = cache.fingerprint(teacher_before, train_loader, val_loader, config, device)
                reused = cache.reuse(fingerprint)
                if reused:
                    tqdm.write(f'Reusing Teacher feature cache: {cache.directory}')
                else:
                    tqdm.write(f'[Cache]: {cache.reason}; resuming extraction.')
                    cache.prepare(fingerprint, blend_scale)
                    try:
                        teacher.to(device)
                        cache.build(teacher, train_loader, val_loader, config, monitor=monitor)
                        cache.seal(fingerprint, blend_scale=blend_scale)
                    finally:
                        teacher.cpu()
                        release_cuda(device)
                if teacher is not None and module_digest(teacher) != teacher_before:
                    raise RuntimeError('Teacher parameters/buffers changed during cache generation')
                teacher.cpu()
                teacher = None
                gc.collect()
                release_cuda(device)
                record({'event': 'teacher_cache', 'reused': reused, 'teacher_device': 'released',
                        'teacher_blend_scale': blend_scale,
                        'cache_dir': str(cache.directory),
                        'gpu_allocated_mib': torch.cuda.memory_allocated(device) / 2**20 if device.type == 'cuda' else None})
                move_student(student, criterion, optimizer, device)
            if resume is not None:
                state = resume.get('resume_state') or {}
                if state.get('rng'):
                    restore_rng(state['rng'], augmentation, (train_loader, val_loader), device)
                    augmentation.applied_count = state.get('augmentation_applied', 0)
                record({'event': 'resume', 'epoch': resume['epoch'], 'step': global_step,
                        'rng_restored': bool(state.get('rng'))})
                tqdm.write(f'[Resume]: epoch={first_epoch} | step={global_step} | RNG=' +
                           ('restored' if state.get('rng') else 'unavailable in this checkpoint'))
            for epoch in range(first_epoch, config['training']['epochs'] + 1):
                epoch_loss, epoch_images, epoch_clips, batches = 0.0, 0, 0, 0
                augmentation_ratio = augmentation.ratio(epoch)
                gpu_values = []
                bar = progress_bar(train_loader, enabled=config['logging']['progress'])
                try:
                    for batch in bar:
                        # Apply current LR and reserved future training controls.
                        controls = schedule.values(epoch=epoch, step=global_step)
                        for group in optimizer.param_groups:
                            group['lr'] = controls['lr']
                        result = train_batch(teacher, student, criterion, batch, optimizer,
                                             config=config, crop_size=config['training']['crop_size'], controls=controls,
                                             augmentation=augmentation,
                                             augmentation_ratio=augmentation_ratio, target_cache=cache,
                                             blend_scale=blend_scale, micro_batch_size=config['training']['micro_batch_size'])
                        global_step += 1
                        batches += 1
                        epoch_clips += result['batch_size']
                        epoch_images += result['frames']
                        images_seen += result['frames']
                        epoch_loss += result['loss'] * result['batch_size']
                        gpu = monitor.sample()
                        if gpu is not None:
                            gpu_values.append(gpu)
                        record({'event': 'train', 'epoch': epoch, 'step': global_step, **result,
                                'lr': controls['lr'], 'grad_clip': controls['grad_clip'],
                                'loss_weights': controls['loss_weights'],
                                'augmentation_ratio': augmentation_ratio,
                                'gpu_utilization_percent': gpu})
                        bar.set_description_str(progress_row(epoch, config['training']['epochs'], device, monitor, [
                            f'{result["loss"]:.5f}', f'{epoch_loss / epoch_clips:.5f}', f'{controls["lr"]:.2e}',
                            epoch_images, network_size(batch['sequences'][0]['images'][0], config['training']['crop_size'], config['rendering'])]), refresh=False)
                        maximum = config['training']['max_steps']
                        if maximum is not None and global_step >= maximum:
                            break
                finally:
                    bar.close()
                optimizer.zero_grad(set_to_none=True)
                metrics.to(device)
                validation = evaluate(teacher, student, criterion, val_loader, metrics, monitor,
                                      config=config, epoch=epoch, controls=controls, target_cache=cache,
                                      blend_scale=blend_scale)
                metrics.cpu()
                if teacher is not None and module_digest(teacher) != teacher_before:
                    raise RuntimeError('Teacher parameters/buffers changed')
                row = {'event': 'epoch', 'epoch': epoch, 'step': global_step,
                       'train_loss': epoch_loss / epoch_clips, 'train_images': epoch_images, 'train_clips': epoch_clips,
                       'train_partial': batches < len(train_loader), 'validation': validation,
                       'gpu_utilization_percent': sum(gpu_values) / len(gpu_values) if gpu_values else None}
                history.append(row)
                record(row)
                from .temporal import write_stage_report
                write_stage_report(validation, report_dir or output, prefix='temporal_latest')
                for profile, scores in validation['profiles'].items():
                    tqdm.write(profile_row(profile, scores))
                quality = validation['image_mean']
                tqdm.write(f'Epoch {epoch}: loss={row["train_loss"]:.6f}, '
                           f'PSNR={quality["psnr"]:.3f}, SSIM={quality["ssim"]:.4f}, LPIPS={quality["lpips"]:.4f}')
                if validation['benchmark']:
                    timing = validation['benchmark']
                    tqdm.write(f'Forward latency: Teacher={timing["teacher_latency_ms"]:.2f} ms/image, '
                               f'Student={timing["student_latency_ms"]:.2f} ms/image')
                current = validation_values(validation)
                improved = is_better(current, best, best_metric)
                if improved:
                    message = best_message(epoch, current, best)
                    best, best_epoch = dict(current), epoch
                stopped_early = early.update(current[best_metric], epoch)
                resume_state = {'rng': capture_rng(augmentation, (train_loader, val_loader), device),
                    'augmentation_applied': augmentation.applied_count, 'history': history,
                    'early': {key: getattr(early, key) for key in ('best', 'best_epoch', 'bad_epochs')},
                    'total_steps': schedule.total_steps, 'teacher_digest': teacher_before,
                    'seconds': previous_seconds + perf_counter() - started, 'report_dir': str(report_dir) if report_dir else None}
                for name in (('best.pt', 'last.pt') if improved else ('last.pt',)):
                    save_checkpoint(output / name, student, criterion, optimizer, epoch=epoch,
                        step=global_step, config=config, controls=controls, blend_scale=blend_scale,
                        resume_state=resume_state)
                    if name == 'best.pt':
                        record({'event': 'best', 'epoch': epoch, 'metric': best_metric, 'values': best})
                        tqdm.write(highlight_best(message))
                if stopped_early:
                    record({'event': 'early_stop', 'epoch': epoch, 'metric': best_metric,
                            'patience': early.patience, 'min_delta': early.min_delta})
                    tqdm.write(f'[Early stop]: {best_metric.upper()} | no sufficient improvement for '
                               f'{early.patience} epochs | best checkpoint epoch={best_epoch}')
                    break
                maximum = config['training']['max_steps']
                if maximum is not None and global_step >= maximum:
                    break
            summary = {
                'status': 'early_stopped' if stopped_early else 'completed',
                'early_stopping': {**config['training']['early_stopping'], 'metric': best_metric,
                                   'triggered': stopped_early, 'bad_epochs': early.bad_epochs},
                'profiles': config['training']['profiles'], 'reference': 'matching_profile_temporal_teacher_rgb', 'global_step': global_step,
                'epochs_run': len(history), 'train_images_seen': images_seen,
                'student_parameters_updated': module_digest(student, parameters_only=True) != student_before,
                'teacher_unchanged': True, 'teacher_released': cache is not None,
                'teacher_blend_scale': blend_scale, 'student_parameter_count': sum(p.numel() for p in student.parameters()),
                'adapter_parameter_count': sum(p.numel() for p in criterion.parameters()),
                'seconds': previous_seconds + perf_counter() - started, 'history': history,
                'best_epoch': best_epoch, 'best_metric': best_metric, 'best_values': best,
                'best_checkpoint': str(output / 'best.pt'), 'last_checkpoint': str(output / 'last.pt'),
                'statistics': metric_statistics(history, best_epoch),
                'report_dir': str(report_dir) if report_dir is not None else None,
                'peak_cuda_allocated_mib': (torch.cuda.max_memory_allocated(device) / 2**20
                                            if device.type == 'cuda' else None),
            }
            # The validation metric can be infinite only for identical RGB pairs.
            (output / 'summary.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print_statistics(summary['statistics'])
            if report_dir is not None:
                write_training_report(summary, report_dir)
            if on_complete is not None:
                # Export models sequentially after all training CUDA resources are released.
                if teacher is not None:
                    teacher.cpu()
                    teacher = None
                metrics.cpu()
                move_student(student, criterion, optimizer, 'cpu')
                gc.collect()
                release_cuda(device)
                on_complete(summary)
            completed = True
            return summary
    finally:
        try:
            if teacher is not None:
                teacher.cpu()
            metrics.cpu()
            move_student(student, criterion, optimizer, 'cpu')
            release_cuda(device)
            monitor.close()
        finally:
            if cache is not None and cache.lock is not None:
                delete = cache_cleanup is None
                try:
                    if completed and cache_cleanup is not None:
                        delete = cache_cleanup(cache.directory)
                finally:
                    cache.close(delete=delete)
