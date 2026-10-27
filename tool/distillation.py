"""Controlled RGB clip training with truncated history and paired history evaluation."""
import hashlib
import math
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import functional as TF
from mlxdlss.temporal import sample_history

from .temporal import (prepare_frame, crop_rgb, frame_guides, reset_history, student_arguments,
                       teacher_target, device_target, compose_tensor)
from .progress import progress_bar, progress_row
from .monitoring import measure_forward


def augment_rgb(rgb, mode, ratio, *, noise_std, generator, add_noise=False):
    """Positive RGB adjustments; training adds noise to one adjustment, reports use one mode."""
    parameters = {'mode': mode, 'fraction': ratio}
    if mode == 'original':
        return rgb.copy(), parameters
    if mode not in ('noise', 'brightness', 'contrast', 'sharpness') or not 0 < ratio <= 1:
        raise ValueError('Invalid augmentation mode or ratio')
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    if mode != 'noise':
        factor = 1 + ratio
        operation = {'brightness': TF.adjust_brightness, 'contrast': TF.adjust_contrast,
                     'sharpness': TF.adjust_sharpness}[mode]
        image = operation(image, factor)
        parameters['factor'] = factor
    if mode == 'noise' or add_noise:
        sigma = noise_std * ratio
        image = image + torch.randn(image.shape, generator=generator) * sigma
        parameters['sigma'] = sigma
    return image.clamp(0, 1).permute(1, 2, 0).contiguous().numpy(), parameters


class StudentAugmentation:
    """A bounded subset of each effective batch receives noise plus one adjustment."""
    def __init__(self, config):
        self.config = config['augmentation']
        self.epochs = config['training']['epochs']
        self.generator = torch.Generator().manual_seed(config['loader']['seed'])
        self.applied_count = 0

    def ratio(self, epoch):
        c = self.config
        if not c['enabled'] or epoch >= self.epochs * c['stop_fraction']:
            return 0.
        progress = min(max((epoch - 1) / max(self.epochs * c['peak_fraction'] - 1, 1), 0), 1)
        return c['start_ratio'] + (c['max_ratio'] - c['start_ratio']) * progress

    def plan(self, count, ratio, lengths=None):
        maximum = math.floor(count * ratio + 1e-9)
        order = torch.randperm(count, generator=self.generator).tolist()
        selected, used = [], 0
        budget = math.floor(sum(lengths) * ratio + 1e-9) if lengths else count
        for index in order:
            weight = lengths[index] if lengths else 1
            if len(selected) < maximum and used + weight <= budget:
                selected.append(index)
                used += weight
        return {index: ('brightness', 'contrast', 'sharpness')[
            torch.randint(3, (), generator=self.generator).item()] for index in selected}

    def apply_rgb(self, rgb, mode, ratio):
        image, _ = augment_rgb(rgb, mode, ratio, noise_std=self.config['noise_std'],
                               generator=self.generator, add_noise=True)
        self.applied_count += 1
        return image


def module_digest(module, *, parameters_only=False):
    digest = hashlib.sha256()
    values = module.named_parameters() if parameters_only else module.state_dict().items()
    for name, tensor in values:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_key(sample, position):
    return f"{sample['profile']}:{sample['clip_id']}:{position + 1}:{sample['sample_ids'][position]}"


def make_frame(sample, position, config, history=None, *, rgb=None, crop_size=None):
    crop_size = config['training']['crop_size'] if crop_size is None else crop_size
    rgb = crop_rgb(sample['images'][position], crop_size) if rgb is None else rgb
    motion, confidence = frame_guides(sample, position, crop_size)
    return prepare_frame(rgb, config, sample['profile'], history=history, motion=motion, confidence=confidence)


def objective(student, criterion, frame, target, blend_scale, loss_weights, *, head=None, features=None):
    device = next(student.parameters()).device
    if head is None:
        image, controls, history = student_arguments(frame, device, student.layout)
        head, features = student(image, controls=controls, history=history, return_features=True)
        if student.layout == 'BCHW':
            head = head.permute(0, 2, 3, 1).contiguous()
    target_head, target_features, target_rgb = device_target(target, device)
    prediction, history = compose_tensor(head, frame, blend_scale)
    if head.shape != target_head.shape:
        raise RuntimeError('Student and Teacher head geometry differs')
    loss, parts = criterion(head, target_head, features, target_features, prediction, target_rgb,
        image_hw=frame.small.shape[:2], network_hw=frame.inputs.shape[1:3], weights=loss_weights)
    return loss, parts, prediction, target_rgb, history


def train_batch(teacher, student, criterion, batch, optimizer, *, config, crop_size, controls,
                augmentation=None, augmentation_ratio=0., target_cache=None, blend_scale=None, micro_batch_size=1):
    student.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    samples = batch['sequences']
    count = len(samples)
    if not count:
        raise ValueError('Empty sequence batch')
    totals, sizes = defaultdict(float), []
    lengths = [torch.randint(min(config['rendering']['train_min_frames'], len(s['images'])),
               len(s['images']) + 1, (), generator=augmentation.generator).item()
               if augmentation else len(s['images']) for s in samples]
    plan = augmentation.plan(count, augmentation_ratio, lengths) if augmentation else {}
    device = next(student.parameters()).device
    for begin in range(0, count, micro_batch_size):
        group = samples[begin:begin + micro_batch_size]
        histories, teacher_histories, previous = [None] * len(group), [None] * len(group), [None] * len(group)
        for position in range(max(lengths[begin:begin + len(group)])):
            prepared, targets = [], []
            for offset, sample in enumerate(group):
                index = begin + offset
                if position >= lengths[index]:
                    continue
                rgb = crop_rgb(sample['images'][position], crop_size)
                if reset_history(previous[offset], rgb, config):
                    histories[offset] = teacher_histories[offset] = None
                clean = make_frame(sample, position, config, teacher_histories[offset], rgb=rgb)
                target = target_cache.load_payload('train', cache_key(sample, position)) if target_cache else teacher_target(teacher, clean)
                teacher_histories[offset] = target['history'] if config['rendering']['temporal'] else None
                student_rgb = augmentation.apply_rgb(rgb, plan[index], augmentation_ratio) if index in plan else rgb
                frame = make_frame(sample, position, config, histories[offset], rgb=student_rgb)
                prepared.append((offset, index, frame))
                targets.append(target)
                previous[offset] = rgb
            # Ragged extents are grouped without resizing away source geometry.
            buckets = defaultdict(list)
            for i, (_, _, frame) in enumerate(prepared):
                buckets[tuple(frame.inputs.shape)].append(i)
            for indices in buckets.values():
                arguments = [student_arguments(prepared[i][2], device, student.layout) for i in indices]
                image, conditioning, memory = [torch.cat([args[k] for args in arguments]) for k in range(3)]
                head, features = student(image, controls=conditioning, history=memory, return_features=True)
                if student.layout == 'BCHW':
                    head = head.permute(0, 2, 3, 1).contiguous()
                combined = None
                for row, i in enumerate(indices):
                    offset, index, frame = prepared[i]
                    loss, parts, prediction, target_rgb, history = objective(student, criterion, frame, targets[i],
                        blend_scale, controls['loss_weights'], head=head[row:row + 1],
                        features={k: v[row:row + 1] for k, v in features.items()})
                    weight = 1 / (count * lengths[index])
                    combined = loss * weight if combined is None else combined + loss * weight
                    for key, value in parts.items():
                        totals[key] += value * weight
                    histories[offset] = history if config['rendering']['temporal'] else None
                    del loss, prediction, target_rgb
                # Detach RGB history at every frame: activation memory does not grow with clip length.
                combined.backward()
                sizes.append(len(indices))
                del combined, head, features, arguments, image, conditioning, memory
    parameters = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError('A trainable Student/adapter parameter has no gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, controls['grad_clip'] or float('inf'), error_if_nonfinite=True).item()
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError('Non-finite optimizer update')
    return {'loss': totals['total'], 'components': dict(totals), 'gradient_norm': norm,
        'batch_size': count, 'frames': sum(lengths), 'clip_lengths': lengths, 'micro_batch_sizes': sizes,
        'sample_ids': [s['clip_id'] for s in samples], 'profiles': [s['profile'] for s in samples],
        'augmented_samples': len(plan), 'augmentation_fraction': len(plan) / count,
        'augmented_frames': sum(lengths[i] for i in plan),
        'augmented_frame_fraction': sum(lengths[i] for i in plan) / sum(lengths)}


def validation_forward(student, criterion, jobs, blend_scale, loss_weights, maximum):
    """Batch matching extents, retaining each image's feature/RGB loss and history."""
    device = next(student.parameters()).device
    buckets = defaultdict(list)
    for index, (frame, target) in enumerate(jobs):
        buckets[tuple(frame.inputs.shape)].append(index)
    results = [None] * len(jobs)
    for indices in buckets.values():
        for begin in range(0, len(indices), maximum):
            selected = indices[begin:begin + maximum]
            arguments = [student_arguments(jobs[i][0], device, student.layout) for i in selected]
            inputs, conditioning, memory = [torch.cat([a[k] for a in arguments]) for k in range(3)]
            head, features = student(inputs, controls=conditioning, history=memory, return_features=True)
            if student.layout == 'BCHW':
                head = head.permute(0, 2, 3, 1).contiguous()
            for row, index in enumerate(selected):
                frame, target = jobs[index]
                results[index] = objective(student, criterion, frame, target, blend_scale, loss_weights,
                    head=head[row:row + 1], features={k: v[row:row + 1] for k, v in features.items()})
    return results


def validation_items(teacher, student, criterion, samples, metrics, config, controls, target_cache, blend_scale):
    maximum = config['training']['micro_batch_size']
    for begin in range(0, len(samples), maximum):
        group = samples[begin:begin + maximum]
        states = [dict(history=None, teacher_history=None, previous=None, previous_output=None,
                       previous_single=None, depth=0) for _ in group]
        for position in range(max(len(sample['images']) for sample in group)):
            entries, jobs = [], []
            for sample, state in zip(group, states):
                if position >= len(sample['images']):
                    continue
                rgb = crop_rgb(sample['images'][position], config['training']['crop_size'])
                if reset_history(state['previous'], rgb, config):
                    state.update(history=None, teacher_history=None, previous_output=None, previous_single=None, depth=0)
                if target_cache is not None:
                    target = target_cache.load_payload('val', cache_key(sample, position))
                else:
                    clean = make_frame(sample, position, config, state['teacher_history'], rgb=rgb)
                    target = teacher_target(teacher, clean)
                frame = make_frame(sample, position, config, state['history'], rgb=rgb)
                main = len(jobs)
                jobs.append((frame, target))
                single = main
                if frame.history is not None:
                    single = len(jobs)
                    jobs.append((make_frame(sample, position, config, None, rgb=rgb), target))
                entries.append((sample, state, rgb, frame, target, main, single))
            results = validation_forward(student, criterion, jobs, blend_scale, controls['loss_weights'], maximum)
            quality, buckets = {}, defaultdict(list)
            for index, result in enumerate(results):
                buckets[tuple(result[2].shape)].append(index)
            for indices in buckets.values():
                for offset in range(0, len(indices), maximum):
                    selected = indices[offset:offset + maximum]
                    predictions = torch.cat([results[i][2] for i in selected])
                    references = torch.cat([results[i][3] for i in selected])
                    values = metrics.per_image(predictions, references)
                    quality.update(zip(selected, values))
            for sample, state, rgb, frame, target, main, single in entries:
                yield sample, position, state, rgb, frame, target, results[main], results[single], quality[main], quality[single]


@torch.no_grad()
def evaluate(teacher, student, criterion, loader, metrics, monitor, *, config, epoch, controls, target_cache=None, blend_scale=None):
    states = student.training, criterion.training
    student.eval()
    criterion.eval()
    totals, scene_totals, scene_counts, stage = defaultdict(float), {}, defaultdict(int), {}
    images, benchmark = 0, None
    bar = progress_bar(loader, enabled=config['logging']['progress'], validation=True)
    try:
        for batch in bar:
            for sample, position, state, rgb, frame, target, primary, reset, quality, single_quality in validation_items(
                    teacher, student, criterion, batch['sequences'], metrics, config, controls, target_cache, blend_scale):
                history, teacher_history, previous_output, previous_single, depth = (
                    state[key] for key in ('history', 'teacher_history', 'previous_output', 'previous_single', 'depth'))
                if benchmark is None and config['benchmark']['enabled']:
                    device = next(student.parameters()).device
                    inputs, conditioning, memory = student_arguments(frame, device, student.layout)
                    class ControlledForward(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.network = student
                        def forward(self, value):
                            return self.network(value, controls=conditioning, history=memory)
                    timing = config['benchmark']
                    benchmark = {f'student_{k}': v for k, v in measure_forward(ControlledForward().eval(),
                        inputs, warmup=timing['warmup'], repeats=timing['repeats']).items()}
                    benchmark.update(target_cache.benchmark or {} if target_cache else
                        {f'teacher_{k}': v for k, v in measure_forward(teacher, frame.inputs.to(device),
                         warmup=timing['warmup'], repeats=timing['repeats']).items()})
                    benchmark.update(device=str(device), profile=sample['profile'],
                        scope='synchronized forward only, batch=1, no feature capture',
                        student_input_shape=list(inputs.shape))
                loss, parts, prediction, target_rgb, next_history = primary
                _, single_parts, single_prediction, _, _ = reset
                row = {'loss': parts['total'], **quality}
                baseline = {'loss': single_parts['total'], **single_quality}
                key = (sample['profile'], position + 1)
                record = stage.setdefault(key, defaultdict(float))
                record['images'] += 1
                record['history_used_fraction'] += float(frame.history is not None)
                record['mean_history_frames'] += depth if frame.history is not None else 0
                for name in row:
                    record[name] += row[name]
                    record['single_' + name] += baseline[name]
                    record['delta_' + name] += row[name] - baseline[name]
                if frame.history is not None and previous_output is not None:
                    motion, confidence = frame_guides(sample, position, config['training']['crop_size'])
                    yy, xx = np.indices(rgb.shape[:2], dtype=np.float32)
                    u, v = (xx + .5) / rgb.shape[1] + motion[..., 0], (yy + .5) / rgb.shape[0] + motion[..., 1]
                    now = prediction[0].permute(1, 2, 0).cpu().numpy()
                    alone = single_prediction[0].permute(1, 2, 0).cpu().numpy()
                    pixels = float(confidence.sum())
                    record['warp_mae'] += float((np.abs(now - sample_history(previous_output, u, v)) * confidence).sum()) / 3
                    record['single_warp_mae'] += float((np.abs(alone - sample_history(previous_single, u, v)) * confidence).sum()) / 3
                    record['valid_pixels'] += pixels
                previous_output = prediction[0].permute(1, 2, 0).cpu().numpy()
                previous_single = single_prediction[0].permute(1, 2, 0).cpu().numpy()
                for name, value in {**parts, **quality}.items():
                    totals[name] += value
                scene_totals.setdefault(sample['scene'], defaultdict(float))
                for name, value in quality.items():
                    scene_totals[sample['scene']][name] += value
                scene_counts[sample['scene']] += 1
                images += 1
                state.update(history=next_history if config['rendering']['temporal'] else None,
                    teacher_history=target['history'] if config['rendering']['temporal'] else None,
                    previous=rgb, previous_output=previous_output, previous_single=previous_single,
                    depth=depth + 1 if frame.history is not None else 1)
                del loss, prediction, target_rgb, single_prediction, target
            bar.set_description_str(progress_row(epoch, config['training']['epochs'], next(student.parameters()).device,
                monitor, [f'{totals["total"] / images:.5f}', f'{totals["psnr"] / images:.2f}',
                          f'{totals["ssim"] / images:.4f}', f'{totals["lpips"] / images:.4f}', images]), refresh=False)
        if not images:
            raise ValueError('Empty validation split')
        stages = []
        for (profile, index), values in sorted(stage.items()):
            count, pixels = int(values['images']), values['valid_pixels']
            row = {'profile': profile, 'frame_index': index, 'images': count}
            for name in ('loss', 'psnr', 'ssim', 'lpips'):
                for prefix in ('', 'single_', 'delta_'):
                    row[prefix + name] = values[prefix + name] / count
            row.update(history_used_fraction=values['history_used_fraction'] / count,
                       mean_history_frames=values['mean_history_frames'] / count,
                       valid_pixels=pixels,
                       warp_mae=values['warp_mae'] / pixels if pixels else None,
                       single_warp_mae=values['single_warp_mae'] / pixels if pixels else None,
                       delta_warp_mae=(values['warp_mae'] - values['single_warp_mae']) / pixels if pixels else None)
            stages.append(row)
        scenes = {scene: {k: v / scene_counts[scene] for k, v in values.items()}
                  for scene, values in scene_totals.items()}
        profiles = {}
        for profile in config['rendering']['profiles']:
            rows = [r for r in stages if r['profile'] == profile]
            n = sum(r['images'] for r in rows)
            if n:
                profiles[profile] = {k: sum(r[k] * r['images'] for r in rows) / n for k in ('loss', 'psnr', 'ssim', 'lpips')}
        return {'images': images, 'reference': 'matching_profile_temporal_teacher_rgb',
            'components': {k: v / images for k, v in totals.items() if k not in ('psnr', 'ssim', 'lpips')},
            'image_mean': {k: totals[k] / images for k in ('psnr', 'ssim', 'lpips')},
            'scene_mean': {k: sum(r[k] for r in scenes.values()) / len(scenes) for k in ('psnr', 'ssim', 'lpips')},
            'scenes': scenes, 'profiles': profiles, 'history_stages': stages, 'benchmark': benchmark,
            'paired_reference': 'Both temporal and reset predictions are scored against the same temporal Teacher target',
            'gpu_utilization_percent': monitor.sample()}
    finally:
        bar.close()
        student.train(states[0])
        criterion.train(states[1])
