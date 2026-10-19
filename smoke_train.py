"""One real Teacher/student distillation update from a Sintel DataLoader batch.

This is an execution test: raw-head MSE, one SGD step, no model checkpoint.
The default student is a small instance of the existing EchoRenderStudent graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from model.student import EchoRenderStudent
from model.teacher import EchoRenderTeacher, resolve_device
from tool.dataloader import DEFAULT_DATA_CONFIG, build_dataloader, load_data_config
from tool.teacher_pipeline import prepare_image


DEFAULT_STUDENT_CONFIG = Path(__file__).resolve().parent / 'config' / 'model_smoke.yaml'


def module_digest(module):
    """Hash parameters AND buffers without retaining a second GPU model copy."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_smoke(teacher, student, batch, *, lr=1e-3):
    """Accumulate mean per-image loss, then make exactly one student update."""
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    if not batch['images']:
        raise ValueError('Smoke batch must contain at least one image')
    if student.input_channels != 16 or student.channels[-1] != 4:
        raise ValueError('Smoke student must have 16 input and 4 output channels')
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError('Student has no trainable parameters')
    device = parameters[0].device
    if device != teacher.device:
        raise ValueError('Teacher and student must use the same device')
    student.train()
    teacher.eval()
    teacher_before = module_digest(teacher)
    student_before = module_digest(student)
    optimizer = torch.optim.SGD(parameters, lr=lr)
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    losses, shapes = [], []
    for image in batch['images']:
        rgb = image.permute(1, 2, 0).contiguous().numpy()
        prepared = prepare_image(rgb)  # identical noise, controls, context and padding
        features = prepared.features.to(device)
        target = teacher(features)  # frozen no_grad forward, ordinary detached tensor
        if target.requires_grad:
            raise RuntimeError('Teacher target unexpectedly requires gradients')
        student_input = features if student.layout == 'BHWC' else features.permute(0, 3, 1, 2).contiguous()
        predicted = student(student_input)
        if student.layout == 'BCHW':
            predicted = predicted.permute(0, 2, 3, 1)
        if predicted.shape != target.shape:
            raise RuntimeError(f'Head shape mismatch: student {predicted.shape}, teacher {target.shape}')
        height, width = rgb.shape[:2]
        # Artificial padding supplies context but is excluded from supervision.
        loss = F.mse_loss(predicted[:, :height, :width].float(), target[:, :height, :width].float())
        if not torch.isfinite(loss).item():
            raise RuntimeError('Distillation loss is not finite')
        (loss / len(batch['images'])).backward()
        losses.append(loss.detach().item())
        shapes.append({'image_hw': [height, width], 'prepared_bhwc': list(features.shape),
                       'head_bhwc': list(target.shape)})
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if len(gradients) != len(parameters) or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError('Student gradients are missing or non-finite')
    gradient_norm = torch.stack([gradient.detach().float().norm() for gradient in gradients]).norm().item()
    if not np.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError('Student gradient norm must be finite and nonzero')
    optimizer.step()
    if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
        raise RuntimeError('Student parameters became non-finite after the update')
    student_changed = module_digest(student) != student_before
    teacher_unchanged = module_digest(teacher) == teacher_before
    teacher_frozen = (not teacher.training and all(not value.requires_grad for value in teacher.parameters())
                      and all(not value.requires_grad for value in teacher.buffers())
                      and all(value.grad is None for value in teacher.parameters()))
    if not student_changed or not teacher_unchanged or not teacher_frozen:
        raise RuntimeError('Expected updated student parameters and an unchanged frozen Teacher')
    return {
        'status': 'passed', 'loss_type': 'raw_head_mse_on_valid_pixels',
        'loss': sum(losses) / len(losses), 'per_image_loss': losses,
        'optimizer': 'SGD', 'learning_rate': lr, 'optimizer_steps': 1,
        'gradient_norm': gradient_norm, 'student_parameters_updated': student_changed,
        'student_gradient_tensors': len(gradients),
        'teacher_unchanged': teacher_unchanged, 'teacher_frozen': teacher_frozen,
        'batch_size': len(batch['images']), 'sample_ids': batch['sample_ids'],
        'shapes': shapes, 'device': str(device),
        'student_parameter_count': sum(parameter.numel() for parameter in student.parameters()),
        'peak_cuda_allocated_mib': (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None),
        'seconds': perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-weights', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument('--student-config', type=Path, default=DEFAULT_STUDENT_CONFIG)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--report', type=Path, help='Optional JSON result; no model weights are saved')
    args = parser.parse_args(argv)
    try:
        if args.cpu_threads < 1 or not np.isfinite(args.lr) or args.lr <= 0:
            raise ValueError('cpu-threads and lr must be positive and finite')
        if args.report and args.report.resolve() in {
                args.teacher_weights.resolve(), args.data_config.resolve(), args.student_config.resolve()}:
            raise ValueError('Report must not overwrite weights or config files')
        torch.set_num_threads(args.cpu_threads)
        config = load_data_config(args.data_config)
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32))
        torch.manual_seed(config.seed)
        loader = build_dataloader(config, split='train')
        batch = next(iter(loader))
        device = resolve_device(args.device)
        print(f'Loaded {len(batch["images"])} training image(s): {batch["sample_ids"]}', flush=True)
        print(f'Loading real Teacher and smoke student on {device}...', flush=True)
        teacher = EchoRenderTeacher(args.teacher_weights, device=device)
        student = EchoRenderStudent(args.student_config).to(device)
        print('Running forward, MSE, backward and one SGD update...', flush=True)
        report = run_smoke(teacher, student, batch, lr=args.lr)
        report.update({'split': 'train', 'render_pass': config.render_pass,
                       'dataset_samples': len(loader.dataset), 'seed': config.seed,
                       'student_config': str(args.student_config),
                       'teacher_weight_tensors': teacher.tensor_count})
        if args.report:
            if args.report.resolve() in {Path(path).resolve() for path in batch['paths']}:
                raise ValueError('Report must not overwrite an input image')
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'Distillation smoke test failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
