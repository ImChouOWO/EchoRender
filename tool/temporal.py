"""Shared controlled RGB sequence rendering, Teacher targets and history metrics."""
from dataclasses import dataclass
from functools import lru_cache
import csv
import json
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from mlxdlss.features import make_features, scaled_color, half, NetworkGeometry
from mlxdlss.temporal import sample_history
from mlxdlss.composition import gaussian_kernel
from teacher_infer import (network_controls, resample,
                           truncate_half, compose_output, dataset_motion)

PROFILES = ('standard', 'natural', 'cinematic', 'neutral')
FORMAT = 'echorender-bhwc16-swin-input-raft-small-bf16-v9'


@dataclass
class Frame:
    rgb: np.ndarray
    small: np.ndarray
    geometry: object
    inputs: torch.Tensor
    history: np.ndarray | torch.Tensor | None
    confidence: np.ndarray | torch.Tensor | None
    controls: dict
    profile: str
    settings: dict
    rendering: dict
    motion_loss: torch.Tensor | None = None
    rgb_tensor: torch.Tensor | None = None
    spatial_features: torch.Tensor | None = None


def image_rgb(image):
    rgb = image.permute(1, 2, 0).contiguous().numpy() if isinstance(image, torch.Tensor) else image
    return rgb.copy()


def adapter_rgb(rgb, size):
    from model.card.card import InputAdapter
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
    return InputAdapter.resize(image, size)[0].permute(1, 2, 0).contiguous().numpy()


def prepare_frame(rgb, config, profile, *, history=None, motion=None, confidence=None):
    rendering, settings = config['rendering'], config['rendering']['profiles'][profile]
    controls, automatic = network_controls(profile, **{key: settings[key] for key in
        ('style_index', 'tone', 'structure', 'skin_structure', 'auto_structure')})
    width = height = config['training']['adapter_size']
    small = adapter_rgb(rgb, width)
    geometry = NetworkGeometry.identity(width, height)
    aligned = None
    if history is not None:
        history = np.asarray(history, dtype=np.float32)
    if history is not None and rendering['temporal'] and rendering['history_strength'] > 0:
        if motion is None or confidence is None:
            raise ValueError('Sintel temporal rendering requires flow and validity guides')
        yy, xx = np.indices(rgb.shape[:2], dtype=np.float32)
        aligned = sample_history(history, (xx + .5) / rgb.shape[1] + motion[..., 0],
                                 (yy + .5) / rgb.shape[0] + motion[..., 1])
    network = make_features(small, geometry=geometry, frame_index=rendering['noise_index'],
                           automatic_mask=automatic, **controls)
    if aligned is not None:
        condition = rgb + confidence * (aligned - rgb)
        condition = adapter_rgb(condition, width)
        network[..., 7:10] = scaled_color(condition[geometry.source_rows()[:, None],
                                                   geometry.source_columns()[None, :]])
    return Frame(rgb, small, geometry, torch.from_numpy(network).unsqueeze(0), aligned,
                 confidence if aligned is not None else None, controls, profile, settings, rendering)


def colour_tensor(frame, device):
    if frame.rgb_tensor is None:
        frame.rgb_tensor = torch.as_tensor(frame.rgb.copy(), device=device, dtype=torch.float32)
    return frame.rgb_tensor


def prepare_student_frames(requests, config, student):
    """Batch identical RGB-pair extents; preserve each image's warp and loss reductions."""
    from collections import defaultdict
    from model.motion import raw_motion, warp_history
    device = next(student.parameters()).device
    frames, groups = [], defaultdict(list)
    zero = sum(parameter.sum() * 0 for parameter in student.motion.parameters()) if student.motion is not None else None
    def tensor(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(value).copy(), device=device, dtype=torch.float32)
    for index, request in enumerate(requests):
        rgb, profile = request['rgb'], request['profile']
        rendering, settings = config['rendering'], config['rendering']['profiles'][profile]
        controls, _ = network_controls(profile, **{key: settings[key] for key in
            ('style_index', 'tone', 'structure', 'skin_structure', 'auto_structure')})
        width = height = student.adapter_size
        small = adapter_rgb(rgb, width)
        geometry = NetworkGeometry.identity(width, height)
        extended = rgb
        network = np.empty((*rgb.shape[:2], 8), dtype=np.float32)
        network[..., :3] = network[..., 3:6] = scaled_color(extended)
        network[..., 6] = half(controls['normalized_style'])
        network[..., 7] = half(controls['local_tone_strength'])
        frame = Frame(rgb, small, geometry, torch.from_numpy(network).unsqueeze(0), None, None,
                      controls, profile, settings, rendering)
        colour_tensor(frame, device)
        frame.motion_loss = zero
        frames.append(frame)
        previous, history = request.get('previous'), request.get('history')
        if (previous is None or history is None or not config['rendering']['temporal']
                or config['rendering']['history_strength'] == 0):
            continue
        if tuple(previous.shape) != tuple(frame.rgb.shape):
            raise ValueError('Reset history when frame dimensions change')
        groups[tuple(frame.rgb.shape)].append(index)
    for indices in groups.values():
        if student.motion_mode == 'raft_small':
            prior = torch.stack([tensor(requests[i]['previous']).permute(2, 0, 1) for i in indices])
            current = torch.stack([frames[i].rgb_tensor.permute(2, 0, 1) for i in indices])
            with student.autocast():
                fields, logits = student.motion(prior, current)
        for row, index in enumerate(indices):
            request, frame = requests[index], frames[index]
            if student.motion_mode == 'raw':
                field, valid = raw_motion(request['previous'], frame.rgb)
                field, valid = tensor(field).permute(2, 0, 1)[None], tensor(valid).permute(2, 0, 1)[None]
            else:
                field, logit = fields[row:row + 1], logits[row:row + 1]
                valid = torch.sigmoid(logit)
                if request.get('motion') is not None and request.get('confidence') is not None:
                    target = tensor(request['motion']).permute(2, 0, 1)[None]
                    mask = tensor(request['confidence']).permute(2, 0, 1)[None]
                    units = field.new_tensor([frame.rgb.shape[1], frame.rgb.shape[0]]).view(1, 2, 1, 1)
                    error = F.smooth_l1_loss(field * units, target * units, reduction='none')
                    flow_loss = (error * mask).sum() / (2 * mask.sum()).clamp_min(1)
                    frame.motion_loss = flow_loss + F.binary_cross_entropy_with_logits(logit, mask)
            history = tensor(request['history']).permute(2, 0, 1)[None]
            aligned, valid = warp_history(history, field, valid)
            frame.history = aligned[0].permute(1, 2, 0)
            frame.confidence = valid[0].permute(1, 2, 0)
    return frames


def prepare_student_frame(rgb, config, profile, student, *, previous=None, history=None,
                          motion=None, confidence=None):
    return prepare_student_frames([dict(rgb=rgb, profile=profile, previous=previous, history=history,
                                       motion=motion, confidence=confidence)], config, student)[0]


def student_previous(frame, student):
    return frame.rgb if student.motion_mode == 'raw' else frame.rgb_tensor.detach()


def student_inputs(frame, device, layout='BHWC'):
    """Base RGB/history/controls plus fixed low/high-frequency RGB and XY gradients."""
    if frame.inputs.shape[-1] != 8:
        raise ValueError('Student frames require eight base RGB/history/control channels')
    image = frame.inputs.to(device).clone()
    if frame.history is not None:
        current = colour_tensor(frame, device)
        memory = current + torch.as_tensor(frame.confidence, device=device) * (torch.as_tensor(frame.history, device=device) - current)
        sampled = memory.unsqueeze(0)
        image[..., 3:6] = half_ste(half_ste(half_ste(sampled) - .5) * .125)
    if frame.spatial_features is None:
        rgb = colour_tensor(frame, device).permute(2, 0, 1).unsqueeze(0)
        frame.spatial_features = rgb_feature_maps(rgb).permute(0, 2, 3, 1)
    image = torch.cat((image, frame.spatial_features), dim=-1)
    return image if layout == 'BHWC' else image.permute(0, 3, 1, 2).contiguous()


@lru_cache(maxsize=8)
def feature_kernels(device, dtype):
    positions = torch.arange(-2, 3, device=device, dtype=dtype)
    gaussian = torch.exp(-positions.square() / 2)
    gaussian = gaussian / gaussian.sum()
    sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype) / 8
    return (gaussian.view(1, 1, 1, 5).expand(3, 1, 1, 5).contiguous(),
            gaussian.view(1, 1, 5, 1).expand(3, 1, 5, 1).contiguous(),
            torch.stack((sobel, sobel.T)).unsqueeze(1))


def rgb_feature_maps(rgb):
    """Gaussian 5x5 sigma=1, signed residual and Sobel XY; RGB is raw [0,1]."""
    horizontal, vertical, sobel = feature_kernels(rgb.device, rgb.dtype)
    low = F.conv2d(F.pad(rgb, (2, 2, 0, 0), mode='replicate'), horizontal, groups=3)
    low = F.conv2d(F.pad(low, (0, 0, 2, 2), mode='replicate'), vertical, groups=3)
    gray = (rgb * rgb.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    gradient = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), sobel)
    # Match RGB input units; signed detail and gradients must not be centered or clipped.
    return torch.cat((half_ste(half_ste(half_ste(low) - .5) * .125),
                      (rgb - low) * .125, gradient * .125), dim=1)


def half_ste(value):
    return value + (value.half().float() - value).detach()


def truncate_tensor(image):
    value = image.detach()
    rounded = value.half().contiguous()
    lower = (rounded.view(torch.int16) - 1).view(torch.float16)
    return torch.where(rounded.float().abs() > value.abs(), lower, rounded).float()


def grade_tensor(image, controls):
    preset = {1: (-.1, -.25, -.1), 2: (0, 0, -.15)}.get(round(controls['normalized_style'] * 128))
    tone = min(1, max(0, controls['local_tone_strength']))
    if preset is None or tone == 0:
        return image
    # Retain the reference's toward-zero history grid with a rounding STE.
    rounded = truncate_tensor(image)
    image = image + (rounded - image).detach()
    exposure, contrast, saturation = (v * tone for v in preset)
    image = (image * 2**exposure).clamp(0, 1)
    image = (image + contrast * (image.square() * (3 - 2 * image) - image)).clamp(0, 1)
    high, low = image.amax(-1), image.amin(-1)
    light, delta = (high + low) * .5, high - low
    span = torch.where(light > .5, 2 - high - low, high + low)
    chroma = torch.where(span > 0, delta / span.clamp_min(1e-12), 0)
    chroma = (chroma * (1 + saturation)).clamp(0, 1)
    ratio = torch.where(delta > 0, span * chroma / delta.clamp_min(1e-12), 0)
    return (light[..., None] + (image - light[..., None]) * ratio[..., None]).clamp(0, 1)


def compose_tensor(head, frame, blend_scale):
    """Native raw-head mapping and differentiable grade/detail, with FP16 rounding STE."""
    h, w = frame.small.shape[:2]
    raw = head[:, :h, :w].float()
    H, W = frame.rgb.shape[:2]
    if (h, w) != (H, W):
        raw = F.interpolate(raw.permute(0, 3, 1, 2), (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    colour = colour_tensor(frame, head.device).unsqueeze(0)
    neural = (colour + .25 * half_ste(raw[..., :3])).clamp(0, 1)
    if frame.history is not None:
        history = torch.as_tensor(frame.history, device=head.device).unsqueeze(0)
        confidence = torch.as_tensor(frame.confidence, device=head.device).unsqueeze(0)
        alpha = torch.sigmoid(half_ste(raw[..., 3:])) * float(np.float16(blend_scale))
        alpha = alpha * frame.rendering['history_strength'] * confidence
        neural = neural + alpha * (history - neural)
    stored = truncate_tensor(neural[0])
    output = grade_tensor(neural, frame.controls)
    output = (colour + frame.settings['intensity'] * (output - colour)).clamp(0, 1)
    detail, colour_strength = frame.settings['detail_strength'], frame.settings['colour_strength']
    if detail != 1 or colour_strength != 1:
        change = (output - colour).permute(0, 3, 1, 2)
        kernel = torch.from_numpy(gaussian_kernel(frame.settings['detail_radius'])).to(head.device)
        radius = kernel.numel() // 2
        low = F.conv2d(F.pad(change, (radius, radius, 0, 0), mode='replicate'),
                       kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        low = F.conv2d(F.pad(low, (0, 0, radius, radius), mode='replicate'),
                       kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
        output = (colour.permute(0, 3, 1, 2) + colour_strength * low + detail * (change - low)).clamp(0, 1)
    else:
        output = output.permute(0, 3, 1, 2)
    return output.contiguous(), stored


@torch.inference_mode()
def teacher_target(teacher, frame):
    inputs = frame.inputs.to(teacher.device)
    if teacher.device.type == 'cuda':
        torch.cuda.synchronize(teacher.device)
    started = perf_counter()
    head, features = teacher(inputs, return_features=True)
    if teacher.device.type == 'cuda':
        torch.cuda.synchronize(teacher.device)
    seconds = perf_counter() - started
    h, w = frame.small.shape[:2]
    raw = resample(head[0, :h, :w].cpu().numpy(), frame.rgb.shape[1], frame.rgb.shape[0])
    output, history = compose_output(raw, frame.rgb, frame.controls, blend_scale=float(teacher.blend_scale),
        history=frame.history, confidence=frame.confidence,
        history_strength=frame.rendering['history_strength'], **{k: frame.settings[k] for k in
        ('intensity', 'detail_strength', 'colour_strength', 'detail_radius')})
    return {'head': head.detach().cpu().clone(),
            'features': {k: v.detach().cpu().clone() for k, v in features.items()},
            'source_rgb': torch.from_numpy(frame.rgb.copy()).permute(2, 0, 1).unsqueeze(0),
            'rgb': torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0), 'history': history,
            'inference_seconds': seconds}


def device_target(target, device):
    def move(value):
        # Only inference-mode targets need a normal clone for loss backward.
        value = value.to(device, non_blocking=True)
        return value.clone() if value.is_inference() and torch.is_grad_enabled() else value
    return move(target['head']), {k: move(v) for k, v in target['features'].items()}, move(target['rgb'])


def frame_guides(sample, index):
    if index == 0:
        return None, None
    motion, confidence = sample['motions'][index], sample['confidences'][index]
    if motion is None:
        raise ValueError('Missing Sintel flow for a consecutive frame')
    h, w = motion.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    valid = ((xx + motion[..., 0] * w >= 0) & (xx + motion[..., 0] * w <= w - 1)
             & (yy + motion[..., 1] * h >= 0) & (yy + motion[..., 1] * h <= h - 1))
    return motion, confidence * valid[..., None]


def reset_history(previous, rgb, config):
    threshold = config['rendering']['scene_cut_threshold']
    return previous is not None and (previous.shape != rgb.shape or
        bool(threshold and np.abs(previous - rgb).mean() > threshold))


def write_stage_report(result, directory, prefix='temporal'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from .trainer import json_safe
    (directory / f'{prefix}.json').write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n')
    fields = ['profile', 'frame_index', 'images', 'loss', 'psnr', 'ssim', 'lpips',
              'single_loss', 'single_psnr', 'single_ssim', 'single_lpips',
              'delta_loss', 'delta_psnr', 'delta_ssim', 'delta_lpips', 'history_used_fraction',
              'mean_history_frames', 'warp_mae', 'single_warp_mae', 'delta_warp_mae', 'valid_pixels']
    with (directory / f'{prefix}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result['history_stages'])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key in zip(axes.flat, ('loss', 'psnr', 'ssim', 'lpips')):
        for profile in PROFILES:
            rows = [row for row in result['history_stages'] if row['profile'] == profile]
            if rows:
                axis.plot([row['frame_index'] for row in rows], [row[key] for row in rows], label=profile)
                axis.plot([row['frame_index'] for row in rows], [row['single_' + key] for row in rows],
                          linestyle='--', alpha=.45)
        axis.set(title=key.upper(), xlabel='Frame index (1 = no history)')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.suptitle('Solid: temporal | Dashed: same-frame history reset', fontweight='bold')
    figure.savefig(directory / f'{prefix}.png', dpi=150)
    plt.close(figure)
