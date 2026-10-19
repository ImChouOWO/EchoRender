"""Training/validation orchestration; loss definitions live in model.card.loss."""
import hashlib
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from .teacher_pipeline import prepare_image, compose_rgb_tensor
from .monitoring import benchmark_forward


def module_digest(module, *, parameters_only=False):
    digest = hashlib.sha256()
    values = module.named_parameters() if parameters_only else module.state_dict().items()
    for name, tensor in values:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_sample(image, crop_size):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected CHW RGB image')
    rgb = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    if crop_size:
        h, w = rgb.shape[:2]
        top, left = max(0, (h - crop_size) // 2), max(0, (w - crop_size) // 2)
        rgb = rgb[top:top + crop_size, left:left + crop_size]
    return prepare_image(rgb)


def sample_objective(teacher, student, criterion, image, *, crop_size, loss_weights):
    prepared = prepare_sample(image, crop_size)
    inputs = prepared.features.to(teacher.device)
    with torch.no_grad():
        target_head, target_features = teacher(inputs, return_features=True)
        target_rgb = compose_rgb_tensor(target_head, prepared, blend_scale=teacher.blend_scale)
    student_inputs = inputs if student.layout == 'BHWC' else inputs.permute(0, 3, 1, 2).contiguous()
    head, features = student(student_inputs, return_features=True)
    if student.layout == 'BCHW':
        head = head.permute(0, 2, 3, 1).contiguous()
    if head.shape != target_head.shape:
        raise RuntimeError('Student and Teacher raw head shapes differ')
    prediction_rgb = compose_rgb_tensor(head, prepared, blend_scale=teacher.blend_scale)
    loss, components = criterion(
        head, target_head, features, target_features, prediction_rgb, target_rgb,
        image_hw=prepared.rgb.shape[:2], network_hw=inputs.shape[1:3], weights=loss_weights,
    )
    return loss, components, prediction_rgb, target_rgb


def train_batch(teacher, student, criterion, batch, optimizer, *, crop_size, controls):
    student.train()
    criterion.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    count = len(batch['images'])
    if count == 0:
        raise ValueError('Empty training batch')
    totals = defaultdict(float)
    for image in batch['images']:
        loss, components, _, _ = sample_objective(
            teacher, student, criterion, image, crop_size=crop_size,
            loss_weights=controls['loss_weights'])
        (loss / count).backward()
        for name, value in components.items():
            totals[name] += value / count
    parameters = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    missing = sum(p.grad is None for p in parameters)
    if missing:
        raise RuntimeError(f'{missing} trainable parameters have no gradients')
    maximum = controls['grad_clip'] or float('inf')
    norm = torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True).item()
    optimizer.step()
    if any(not torch.isfinite(p).all().item() for p in parameters):
        raise RuntimeError('Parameters became non-finite after optimizer update')
    return {'loss': totals['total'], 'components': dict(totals), 'gradient_norm': norm,
            'batch_size': count, 'sample_ids': list(batch['sample_ids'])}


@torch.no_grad()
def evaluate(teacher, student, criterion, loader, metrics, monitor, *, config, epoch, controls):
    states = student.training, criterion.training
    student.eval()
    criterion.eval()
    totals, scene_totals, scene_counts = defaultdict(float), {}, defaultdict(int)
    images, benchmark = 0, None
    gpu_values = []
    bar = tqdm(loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]} [Val]',
               unit='it', dynamic_ncols=True, disable=not config['logging']['progress'])
    try:
        for batch in bar:
            for image, scene in zip(batch['images'], batch['scenes']):
                _, parts, prediction, target = sample_objective(
                    teacher, student, criterion, image, crop_size=config['training']['crop_size'],
                    loss_weights=controls['loss_weights'])
                quality = metrics(prediction, target)
                values = {**parts, **quality}
                for key, value in values.items():
                    totals[key] += value
                scene_totals.setdefault(scene, defaultdict(float))
                for key, value in quality.items():
                    scene_totals[scene][key] += value
                scene_counts[scene] += 1
                images += 1
                # Isolated forward benchmark, once per epoch, with eval modes and sync.
                if benchmark is None and config['benchmark']['enabled']:
                    inputs = prepare_sample(image, config['training']['crop_size']).features.to(teacher.device)
                    benchmark = benchmark_forward(teacher, student, inputs,
                                                  warmup=config['benchmark']['warmup'],
                                                  repeats=config['benchmark']['repeats'])
            gpu = monitor.sample()
            if gpu is not None:
                gpu_values.append(gpu)
            bar.set_postfix(loss=f'{totals["total"] / images:.5f}',
                            psnr=f'{totals["psnr"] / images:.2f}', gpu=monitor.display())
        if not images:
            raise ValueError('Validation loader has no images')
        scenes = {name: {key: value / scene_counts[name] for key, value in values.items()}
                  for name, values in scene_totals.items()}
        return {
            'images': images, 'reference': 'teacher_rgb',
            'components': {key: value / images for key, value in totals.items() if key not in ('psnr', 'ssim', 'lpips')},
            'image_mean': {key: totals[key] / images for key in ('psnr', 'ssim', 'lpips')},
            'scene_mean': {key: sum(row[key] for row in scenes.values()) / len(scenes)
                           for key in ('psnr', 'ssim', 'lpips')},
            'scenes': scenes, 'benchmark': benchmark,
            'gpu_utilization_percent': sum(gpu_values) / len(gpu_values) if gpu_values else None,
        }
    finally:
        bar.close()
        student.train(states[0])
        criterion.train(states[1])
