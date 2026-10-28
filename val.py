"""Validate a saved Student and export named Teacher/Student comparison reports."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader, Subset

from model.student import EchoRenderStudent
from model.teacher import EchoRenderTeacher, resolve_device
from model.card.loss import DistillationLoss
from tool.dataloader import build_dataloader
from tool.distillation import evaluate
from tool.metrics import ImageMetrics
from tool.monitoring import GPUMonitor
from tool.prediction_report import export_predictions
from tool.reporting import (DIRECTIONS, validation_values, metric_statistics,
                            print_statistics, write_training_report)
from tool.trainer import json_safe, release_cuda
from tool.training_config import DEFAULT_TRAIN_CONFIG, PROJECT_ROOT, LOSS_WEIGHTS, load_training_config, reserve_output_dir, selected_data_config


def project_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise argparse.ArgumentTypeError('project must contain only letters, digits, _ and -')
    return value


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('max-clips must be a positive integer')
    return number


def limit_loader(loader, maximum):
    if maximum is None:
        return loader
    dataset = loader.dataset
    if hasattr(dataset, 'items'):
        chosen = set(range(min(maximum, len(dataset.clips))))
        indices = [i for i, (clip, profile) in enumerate(dataset.items) if clip in chosen]
    else:
        indices = range(min(maximum, len(dataset)))
    return DataLoader(Subset(dataset, indices),
                      batch_size=loader.batch_size, num_workers=loader.num_workers,
                      collate_fn=loader.collate_fn, shuffle=False)


def export_history(weights, directory, config):
    path = weights.parent / 'summary.json'
    if not path.is_file():
        print('[Report]: No training summary found; historical curves are unavailable.', flush=True)
        return None
    summary = json.loads(path.read_text())
    history = summary.get('history')
    if not history:
        print('[Report]: Training summary has no epoch history; historical curves are unavailable.', flush=True)
        return None
    # Historical best and fresh validation of the selected file are separate records.
    for row in history:
        row['validation']['image_mean']['psnr'] = float(row['validation']['image_mean']['psnr'])
    metric = summary.get('best_metric', config['reporting']['best_metric'])
    best = max(history, key=lambda row: DIRECTIONS[metric] * validation_values(row['validation'])[metric])
    summary.update(best_epoch=best['epoch'], best_metric=metric,
                   statistics=metric_statistics(history, best['epoch']))
    print_statistics(summary['statistics'])
    write_training_report(summary, directory)
    return str(path.resolve())


def evaluate_checkpoint(payload, teacher_factory, loaders, config, device):
    """Cache isolated evaluation targets, unload Teacher, then score RGB clips."""
    student = EchoRenderStudent(payload['student_config'])
    student.load_state_dict(payload['student'], strict=True)
    widths = {name: student.channels[index] for name, index in student.FEATURE_LAYERS.items()}
    criterion = DistillationLoss(config['loss'], student_channels=widths)
    criterion.load_state_dict(payload['distillation'], strict=True)
    teacher = teacher_factory()
    from tool.teacher_features import TeacherFeatureCache, teacher_temporal_signature
    from tool.distillation import cache_key, make_frame
    from tool.temporal import teacher_target, crop_rgb, reset_history
    from tool.trainer import FEATURE_CACHE_DIR
    cache = TeacherFeatureCache(FEATURE_CACHE_DIR, index_name='evaluation.json', split_override='evaluate')
    metrics = ImageMetrics(net=config['validation']['lpips_net'], ssim_window=config['loss']['ssim_window'])
    monitor = GPUMonitor(device, config['logging']['gpu_poll_seconds'])
    results = {}
    try:
        cache.open()
        import hashlib, inspect
        from model.motion import raw_motion
        import cv2
        from tool.distillation import module_digest
        sources = []
        for loader in loaders.values():
            dataset = loader.dataset
            while isinstance(dataset, Subset):
                dataset = dataset.dataset
            for scene, paths in dataset.clips:
                for path in paths:
                    for item in (path, *(dataset.root / dataset.source / name / scene / (path.stem + suffix)
                        for name, suffix in (('flow', '.flo'), ('invalid', '.png'), ('occlusions', '.png')))):
                        if item.is_file():
                            stat = item.stat()
                            sources.append((str(item), stat.st_size, stat.st_mtime_ns))
        pipeline, previous_pipeline = teacher_temporal_signature()
        signature = {'teacher': module_digest(teacher), 'rendering': config['rendering'],
            'crop': config['training']['crop_size'], 'sources': sources, 'device': str(device), 'torch': torch.__version__,
            'pipeline': pipeline,
            'external_motion': hashlib.sha256(inspect.getsource(raw_motion).encode()).hexdigest(),
            'opencv': cv2.__version__}
        alternate = {**signature, 'pipeline': previous_pipeline} if previous_pipeline else None
        fingerprint = cache.configure_signatures(signature, alternate)
        cache.prepare(fingerprint, payload['teacher_blend_scale'])
        teacher.eval().to(device)
        for split, loader in loaders.items():
            for batch in loader:
                for sample in batch['sequences']:
                    history = previous = None
                    for position, image in enumerate(sample['images']):
                        rgb = crop_rgb(image, config['training']['crop_size'])
                        if reset_history(previous, rgb, config):
                            history = None
                        key = cache_key(sample, position)
                        if cache.finished('val', key):
                            target = cache.load_payload('val', key)
                        else:
                            target = teacher_target(teacher, make_frame(sample, position, config, history, rgb=rgb))
                            target['history'] = torch.from_numpy(target['history'])
                            cache.store('val', key, target)
                        history = target['history'].numpy() if config['rendering']['temporal'] else None
                        previous = rgb
        cache.seal(fingerprint)
        teacher.cpu()
        teacher = None
        release_cuda(device)
        for model in (student, criterion, metrics):
            model.eval().requires_grad_(False).to(device)
        evaluation_config = deepcopy(config)
        evaluation_config['benchmark']['enabled'] = False
        evaluation_config['training']['epochs'] = max(payload['epoch'], config['training']['epochs'])
        controls = {'loss_weights': {name: config['loss'][name] for name in LOSS_WEIGHTS}}
        with torch.inference_mode():
            for split, loader in loaders.items():
                print(f'[Validate]: {split} | selected profiles | paired temporal/reset predictions', flush=True)
                results[split] = evaluate(None, student, criterion, loader, metrics, monitor,
                    config=evaluation_config, epoch=payload['epoch'], controls=controls, target_cache=cache,
                    blend_scale=payload['teacher_blend_scale'])
    finally:
        if teacher is not None:
            teacher.cpu()
        for model in (student, criterion, metrics):
            model.cpu()
        if cache.lock is not None:
            cache.close(delete=False)
        monitor.close()
        release_cuda(device)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True, help='Saved best.pt or last.pt')
    parser.add_argument('--project', type=project_name, required=True, help='Report folder under output/val/')
    parser.add_argument('--config', type=Path, default=DEFAULT_TRAIN_CONFIG, help='Dataset, device and report settings')
    parser.add_argument('--device', help='Override runtime.device: cpu or cuda:N')
    parser.add_argument('--split', choices=('val', 'test', 'both'), default='both')
    parser.add_argument('--max-clips', type=positive_integer, help='Limit physical clips per split; retain all trained profiles')
    parser.add_argument('--imgsz', type=int, help='Comparison inference longest edge; 0 keeps native inference')
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config)
        if args.imgsz is not None:
            if args.imgsz < 0:
                raise ValueError('imgsz must be nonnegative')
            config['reporting']['inference_size'] = args.imgsz
        weights = args.weights.expanduser().resolve()
        payload = torch.load(weights, map_location='cpu', weights_only=True)
        required = {'student', 'student_config', 'distillation', 'train_config', 'epoch', 'teacher_blend_scale'}
        if not isinstance(payload, dict) or payload.get('format') != 'echorender-bhwc16-motion-v4' or not required <= payload.keys():
            raise ValueError('Expected a complete echorender-bhwc16-motion-v4 training checkpoint')
        # Score with the checkpoint's objective; compare inputs using the current augmentation settings.
        config['rendering'] = deepcopy(payload['train_config']['rendering'])
        config['training']['profiles'] = list(config['rendering']['profiles'])
        config['loss'] = deepcopy(payload['train_config']['loss'])
        config['training']['crop_size'] = payload['train_config']['training']['crop_size']
        torch.set_num_threads(config['runtime']['cpu_threads'])
        device = resolve_device(args.device or config['runtime']['device'])
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        config['runtime']['device'] = str(device)
        if not Path(config['model']['teacher_weights']).is_file():
            raise FileNotFoundError(config['model']['teacher_weights'])
        os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.cache' / 'torch'))
        data = selected_data_config(config)
        loaders, missing = {}, {}
        for split in (('val', 'test') if args.split == 'both' else (args.split,)):
            try:
                loaders[split] = limit_loader(build_dataloader(data, split=split), args.max_clips)
            except (FileNotFoundError, ValueError) as exc:
                if args.split != 'both':
                    raise
                missing[split] = str(exc)
                print(f'[Report]: {split} unavailable: {exc}', flush=True)
        if not loaders:
            raise ValueError('No requested image split is available')
        directory = reserve_output_dir(Path(config['runtime']['output_dir']).parent / 'val' / args.project)
        print(f'Output: {directory}', flush=True)
        print(f'Checkpoint: {weights} | epoch={payload["epoch"]} | device={device}', flush=True)
        def create_teacher():
            return EchoRenderTeacher(config['model']['teacher_weights'], device='cpu')
        history = export_history(weights, directory, config)
        validation = evaluate_checkpoint(payload, create_teacher, loaders, config, device)
        result = {'checkpoint': str(weights), 'checkpoint_epoch': payload['epoch'], 'history_source': history,
                  'device': str(device), 'max_clips_per_split': args.max_clips, 'missing_splits': missing,
                  'validation_crop_size': config['training']['crop_size'],
                  'comparison_crop_size': config['reporting']['crop_size'],
                  'comparison_inference_size': config['reporting']['inference_size'],
                  'comparison_samples_per_mode': config['reporting']['samples_per_mode'],
                  'scoring_input': 'original (no augmentation)', 'results': validation}
        from tool.temporal import write_stage_report
        for split, scores in validation.items():
            write_stage_report(scores, directory, prefix=f'temporal_{split}')
        (directory / 'validation.json').write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n')
        del payload
        gc.collect()
        release_cuda(device)
        export_predictions(weights, create_teacher, loaders, config, directory, device, missing_splits=missing)
        print(f'[Done]: Reports saved to {directory}', flush=True)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(2, f'Validation failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
