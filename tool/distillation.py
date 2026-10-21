"""Training/validation orchestration; loss definitions live in model.card.loss."""
import hashlib
from collections import defaultdict

import torch
from tqdm.auto import tqdm
from torchvision.transforms import functional as TF

from .teacher_pipeline import prepare_image, compose_rgb_tensor
from .monitoring import benchmark_forward, measure_forward


class StudentAugmentation:
    """Seeded student-only RGB corruption, scheduled by configured epochs."""
    def __init__(self, config):
        self.config = dict(config['augmentation'])
        self.epochs = config['training']['epochs']
        self.generator = torch.Generator().manual_seed(config['loader']['seed'])
        self.applied_count = 0

    def probability(self, epoch):
        c = self.config
        if not c['enabled'] or epoch >= self.epochs * c['stop_fraction']:
            return 0.0
        progress = min(max((epoch - 1) / max(self.epochs * c['peak_fraction'] - 1, 1), 0), 1)
        return c['start_probability'] + (c['max_probability'] - c['start_probability']) * progress

    def apply(self, prepared, probability):
        if probability <= 0 or torch.rand((), generator=self.generator).item() >= probability:
            return prepared
        c = self.config
        rgb = torch.from_numpy(prepared.rgb.copy()).permute(2, 0, 1)
        sigma = torch.rand((), generator=self.generator).item() * c['noise_std']
        rgb = (rgb + torch.randn(rgb.shape, generator=self.generator) * sigma).clamp(0, 1)
        mode = torch.randint(3, (), generator=self.generator).item()
        name, operation = (('brightness', TF.adjust_brightness), ('contrast', TF.adjust_contrast),
                           ('sharpness', TF.adjust_sharpness))[mode]
        factor = 1 + (2 * torch.rand((), generator=self.generator).item() - 1) * c[name]
        rgb = operation(rgb, factor).clamp(0, 1)
        self.applied_count += 1
        return prepare_image(rgb.permute(1, 2, 0).contiguous().numpy())


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


def sample_objective(teacher, student, criterion, image, *, crop_size, loss_weights,
                     augmentation=None, augmentation_probability=0.0, targets=None, blend_scale=None):
    blend_scale = teacher.blend_scale if blend_scale is None else blend_scale
    prepared = prepare_sample(image, crop_size)
    device = next(student.parameters()).device
    if targets is None:
        with torch.no_grad():
            target_head, target_features = teacher(prepared.features.to(device), return_features=True)
            target_rgb = compose_rgb_tensor(target_head, prepared, blend_scale=blend_scale)
    else:
        target_head, target_features, target_rgb = targets
    student_prepared = augmentation.apply(prepared, augmentation_probability) if augmentation else prepared
    student_inputs = student_prepared.features.to(device)
    if student.layout != 'BHWC':
        student_inputs = student_inputs.permute(0, 3, 1, 2).contiguous()
    head, features = student(student_inputs, return_features=True)
    if student.layout == 'BCHW':
        head = head.permute(0, 2, 3, 1).contiguous()
    if head.shape != target_head.shape:
        raise RuntimeError('Student and Teacher raw head shapes differ')
    prediction_rgb = compose_rgb_tensor(head, student_prepared, blend_scale=blend_scale)
    loss, components = criterion(
        head, target_head, features, target_features, prediction_rgb, target_rgb,
        image_hw=prepared.rgb.shape[:2], network_hw=prepared.features.shape[1:3], weights=loss_weights,
    )
    return loss, components, prediction_rgb, target_rgb


def training_microbatches(images, *, crop_size, micro_batch_size, augmentation, probability):
    """Batch only matching network extents; preserve order and per-image geometry."""
    group = []
    for index, image in enumerate(images):
        prepared = prepare_sample(image, crop_size)
        augmented = augmentation.apply(prepared, probability) if augmentation else prepared
        if group and (len(group) == micro_batch_size or
                      augmented.features.shape != group[0][2].features.shape):
            yield group
            group = []
        group.append((index, prepared, augmented))
    if group:
        yield group


def microbatch_objective(teacher, student, criterion, group, sample_ids, *,
                         loss_weights, target_cache, blend_scale):
    device = next(student.parameters()).device
    online_targets = []
    if target_cache is None:
        # Keep Teacher inference single-image, before Student's training forward.
        with torch.no_grad():
            for _, prepared, _ in group:
                target_head, target_features = teacher(prepared.features.to(device), return_features=True)
                online_targets.append((target_head, target_features,
                                       compose_rgb_tensor(target_head, prepared, blend_scale=blend_scale)))
    inputs = torch.cat([augmented.features for _, _, augmented in group]).to(device)
    if student.layout != 'BHWC':
        inputs = inputs.permute(0, 3, 1, 2).contiguous()
    head, features = student(inputs, return_features=True)
    if student.layout == 'BCHW':
        head = head.permute(0, 2, 3, 1).contiguous()
    loss_sum, totals = None, defaultdict(float)
    for position, (index, prepared, augmented) in enumerate(group):
        if target_cache is not None:
            target_head, target_features, target_rgb = target_cache.load('train', sample_ids[index], device)
        else:
            target_head, target_features, target_rgb = online_targets[position]
        sample_head = head[position:position + 1]
        if sample_head.shape != target_head.shape:
            raise RuntimeError('Student and Teacher raw head shapes differ')
        prediction_rgb = compose_rgb_tensor(sample_head, augmented, blend_scale=blend_scale)
        loss, components = criterion(
            sample_head, target_head, {k: v[position:position + 1] for k, v in features.items()},
            target_features, prediction_rgb, target_rgb, image_hw=prepared.rgb.shape[:2],
            network_hw=prepared.features.shape[1:3], weights=loss_weights)
        loss_sum = loss if loss_sum is None else loss_sum + loss
        for name, value in components.items():
            totals[name] += value
    return loss_sum, totals


def train_batch(teacher, student, criterion, batch, optimizer, *, crop_size, controls,
                augmentation=None, augmentation_probability=0.0, target_cache=None, blend_scale=None,
                micro_batch_size=1):
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError('micro_batch_size must be a positive integer')
    student.train()
    criterion.train()
    if teacher is not None:
        teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    count = len(batch['images'])
    if count == 0:
        raise ValueError('Empty training batch')
    totals = defaultdict(float)
    augmented_before = augmentation.applied_count if augmentation else 0
    if len(batch['sample_ids']) != count:
        raise ValueError('Each training image needs one sample ID')
    blend_scale = teacher.blend_scale if blend_scale is None else blend_scale
    sizes = []
    for group in training_microbatches(batch['images'], crop_size=crop_size,
                                      micro_batch_size=micro_batch_size, augmentation=augmentation,
                                      probability=augmentation_probability):
        loss, components = microbatch_objective(
            teacher, student, criterion, group, batch['sample_ids'],
            loss_weights=controls['loss_weights'], target_cache=target_cache, blend_scale=blend_scale)
        # Normalize by the actual effective batch, including a short final group.
        (loss / count).backward()
        del loss
        sizes.append(len(group))
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
            'batch_size': count, 'micro_batch_sizes': sizes, 'sample_ids': list(batch['sample_ids']),
            'augmented_samples': augmentation.applied_count - augmented_before if augmentation else 0}


@torch.no_grad()
def evaluate(teacher, student, criterion, loader, metrics, monitor, *, config, epoch, controls, target_cache=None, blend_scale=None):
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
            for image, scene, sample_id in zip(batch['images'], batch['scenes'], batch['sample_ids']):
                targets = target_cache.load('val', sample_id, next(student.parameters()).device) if target_cache else None
                _, parts, prediction, target = sample_objective(
                    teacher, student, criterion, image, crop_size=config['training']['crop_size'],
                    loss_weights=controls['loss_weights'], targets=targets, blend_scale=blend_scale)
                quality = metrics(prediction, target)
                del targets, prediction, target
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
                    inputs = prepare_sample(image, config['training']['crop_size']).features.to(next(student.parameters()).device)
                    timing = config['benchmark']
                    if target_cache is None:
                        benchmark = benchmark_forward(teacher, student, inputs,
                                                      warmup=timing['warmup'], repeats=timing['repeats'])
                    else:
                        student_inputs = inputs if student.layout == 'BHWC' else inputs.permute(0, 3, 1, 2).contiguous()
                        benchmark = {**target_cache.benchmark, **{f'student_{key}': value for key, value in
                            measure_forward(student, student_inputs, warmup=timing['warmup'], repeats=timing['repeats']).items()}}
                        del student_inputs
                    del inputs
            gpu = monitor.sample()
            if gpu is not None:
                gpu_values.append(gpu)
            bar.set_postfix({'GPU': monitor.display(), 'loss': f'{totals["total"] / images:.5f}',
                             'psnr': f'{totals["psnr"] / images:.2f}'})
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
