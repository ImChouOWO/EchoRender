"""Multi-epoch student distillation with a real frozen Teacher and raw-head MSE.

Uses the full config/model.yaml student by default. Both models see identical
center-cropped inputs (320 by default); use --crop-size 0 for original images.
"""
import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from model.model import DEFAULT_CONFIG
from model.student import EchoRenderStudent
from model.teacher import EchoRenderTeacher, resolve_device
from tool.dataloader import DEFAULT_DATA_CONFIG, build_dataloader, load_data_config
from tool.distillation import evaluate, module_digest, train_batch


def _positive_int(value, name, *, optional=False):
    if value is None and optional:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def load_student_state(student, path):
    """Strict student-only initialization; this does not resume optimizer/epoch state."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload.get('student', payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError('student-weights must contain a state_dict or a training checkpoint')
    student.load_state_dict(state, strict=True)


def save_checkpoint(path, student, optimizer, *, epoch, step, epoch_complete, context):
    """Save only student/optimizer state and metadata; never duplicate the Teacher."""
    payload = {
        'format': 'echorender-student-v1',
        'student': {name: tensor.detach().cpu() for name, tensor in student.state_dict().items()},
        'student_config': student.config, 'optimizer': optimizer.state_dict(),
        'epoch': epoch, 'global_step': step, 'epoch_complete': epoch_complete,
        'context': context,
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(teacher, student, train_loader, val_loader, optimizer, *, epochs=1,
        crop_size=320, grad_clip=1.0, max_steps=None, output_dir=None,
        log_every=10, context=None):
    """Keep one optimizer across all batches/epochs and validate without updates."""
    for value, name in ((epochs, 'epochs'), (log_every, 'log_every')):
        _positive_int(value, name)
    _positive_int(max_steps, 'max_steps', optional=True)
    if type(crop_size) is not int or crop_size < 0:
        raise ValueError('crop_size must be a nonnegative integer')
    if grad_clip is not None and (not np.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError('grad_clip must be finite and positive')
    teacher.eval()
    if (any(value.requires_grad for value in teacher.parameters())
            or any(value.requires_grad for value in teacher.buffers())):
        raise ValueError('Teacher must be frozen before training')
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Train and validation loaders must be nonempty')
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any((output_dir / name).exists() for name in ('metrics.jsonl', 'last.pt', 'summary.json')):
            raise FileExistsError(f'Training outputs already exist in {output_dir}; choose a new --output-dir')
    stream = None

    def record(row, *, display=True):
        line = json.dumps(row)
        if stream:
            stream.write(line + '\n')
            stream.flush()
        if display:
            print(line, flush=True)

    teacher_before = module_digest(teacher)
    student_before = module_digest(student, parameters_only=True)
    global_step, total_images, history = 0, 0, []
    context = dict(context or {})
    started = perf_counter()
    device = next(student.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    stream = (output_dir / 'metrics.jsonl').open('x', encoding='utf-8') if output_dir else None
    try:
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_images, batches = 0.0, 0, 0
            for batch in train_loader:
                result = train_batch(teacher, student, batch, optimizer, crop_size=crop_size, grad_clip=grad_clip)
                global_step += 1
                batches += 1
                epoch_images += result['batch_size']
                total_images += result['batch_size']
                epoch_loss += result['loss'] * result['batch_size']
                record({'event': 'train', 'epoch': epoch, 'step': global_step, **result},
                       display=global_step == 1 or global_step % log_every == 0)
                if max_steps is not None and global_step >= max_steps:
                    break
            validation = evaluate(teacher, student, val_loader, crop_size=crop_size)
            if module_digest(teacher) != teacher_before:
                raise RuntimeError('Teacher parameters or buffers changed during training/validation')
            epoch_result = {
                'event': 'epoch', 'epoch': epoch, 'step': global_step,
                'train_loss': epoch_loss / epoch_images, 'train_images': epoch_images,
                'train_batches': batches, 'train_partial': batches < len(train_loader),
                'validation': validation,
            }
            history.append(epoch_result)
            record(epoch_result)
            if output_dir:
                save_checkpoint(output_dir / 'last.pt', student, optimizer, epoch=epoch, step=global_step,
                                epoch_complete=batches == len(train_loader), context=context)
            if max_steps is not None and global_step >= max_steps:
                break
        student_changed = module_digest(student, parameters_only=True) != student_before
        summary = {
            'status': 'completed', 'loss_type': 'raw_head_mse_on_valid_pixels',
            'optimizer': type(optimizer).__name__, 'global_step': global_step,
            'epochs_run': len(history), 'train_images_seen': total_images,
            'student_parameters_updated': student_changed, 'teacher_unchanged': True,
            'teacher_frozen': (not teacher.training and all(not value.requires_grad for value in teacher.parameters())
                               and all(not value.requires_grad for value in teacher.buffers())
                               and all(value.grad is None for value in teacher.parameters())),
            'student_parameter_count': sum(parameter.numel() for parameter in student.parameters()),
            'crop_size': crop_size, 'gradient_clip': grad_clip, 'device': str(device),
            'peak_cuda_allocated_mib': (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None),
            'seconds': perf_counter() - started, 'history': history, 'context': context,
            'checkpoint': str(output_dir / 'last.pt') if output_dir else None,
        }
        if output_dir:
            (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        return summary
    finally:
        if stream:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-weights', type=Path, required=True)
    parser.add_argument('--student-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--student-weights', type=Path, help='Optional student-only initialization, not optimizer resume')
    parser.add_argument('--data-config', type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/train'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--crop-size', type=int, default=320, help='Shared center crop; 0 keeps original images')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient norm cap; 0 disables clipping')
    parser.add_argument('--max-steps', type=int, help='Stop training after this many optimizer updates')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    try:
        for value, name in ((args.epochs, 'epochs'), (args.cpu_threads, 'cpu-threads'), (args.log_every, 'log-every')):
            _positive_int(value, name)
        _positive_int(args.max_steps, 'max-steps', optional=True)
        if not np.isfinite(args.lr) or args.lr <= 0 or args.crop_size < 0:
            raise ValueError('lr must be finite and positive; crop-size must be nonnegative')
        if not np.isfinite(args.grad_clip) or args.grad_clip < 0:
            raise ValueError('grad-clip must be finite and nonnegative')
        # Never place generated training files inside the source dataset.
        config = load_data_config(args.data_config)
        if args.output_dir.resolve().is_relative_to(config.root):
            raise ValueError('output-dir must be outside the source dataset')
        if any((args.output_dir / name).exists() for name in ('last.pt', 'metrics.jsonl', 'summary.json')):
            raise FileExistsError(f'Training outputs already exist in {args.output_dir}; choose a new --output-dir')
        torch.set_num_threads(args.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32))
        torch.manual_seed(config.seed)
        train_loader = build_dataloader(config, split='train')
        val_loader = build_dataloader(config, split='val')
        device = resolve_device(args.device)
        print(f'Loading Teacher and student on {device}; train={len(train_loader.dataset)}, val={len(val_loader.dataset)}', flush=True)
        teacher = EchoRenderTeacher(args.teacher_weights, device=device)
        student = EchoRenderStudent(args.student_config)
        if args.student_weights:
            load_student_state(student, args.student_weights)
        student.to(device)
        optimizer = torch.optim.SGD([parameter for parameter in student.parameters() if parameter.requires_grad], lr=args.lr)
        context = {
            'student_config_path': str(args.student_config), 'student_initialized_from': str(args.student_weights) if args.student_weights else None,
            'teacher_weights': str(args.teacher_weights), 'teacher_weight_tensors': teacher.tensor_count,
            'teacher_weights_format': teacher.metadata.get('format'), 'data_config': str(args.data_config),
            'render_pass': config.render_pass, 'seed': config.seed, 'learning_rate': args.lr,
            'train_scenes': list(train_loader.dataset.scenes), 'val_scenes': list(val_loader.dataset.scenes),
            'train_samples': len(train_loader.dataset), 'val_samples': len(val_loader.dataset),
            'max_steps': args.max_steps,
        }
        report = fit(teacher, student, train_loader, val_loader, optimizer, epochs=args.epochs,
                     crop_size=args.crop_size, grad_clip=args.grad_clip or None, max_steps=args.max_steps,
                     output_dir=args.output_dir, log_every=args.log_every, context=context)
        print(json.dumps(report, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'Distillation training failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
