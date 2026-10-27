"""Shared controlled RGB sequence rendering, Teacher targets and history metrics."""
from dataclasses import dataclass
import csv
import json
import math
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from mlxdlss.features import make_features, scaled_color
from mlxdlss.temporal import sample_history
from mlxdlss.composition import gaussian_kernel
from teacher_infer import (network_controls, network_geometry, render_extent, resample,
                           truncate_half, compose_output, dataset_motion)

PROFILES = ('standard', 'natural', 'cinematic', 'neutral')
FORMAT = 'echorender-rgb-temporal-v3'


@dataclass
class Frame:
    rgb: np.ndarray
    small: np.ndarray
    geometry: object
    inputs: torch.Tensor
    history: np.ndarray | None
    confidence: np.ndarray | None
    controls: dict
    profile: str
    settings: dict
    rendering: dict


def crop_rgb(image, size):
    rgb = image.permute(1, 2, 0).contiguous().numpy() if isinstance(image, torch.Tensor) else image
    h, w = rgb.shape[:2]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return (rgb[top:top + size, left:left + size] if size else rgb).copy()


def prepare_frame(rgb, config, profile, *, history=None, motion=None, confidence=None):
    rendering, settings = config['rendering'], config['rendering']['profiles'][profile]
    controls, automatic = network_controls(profile, **{key: settings[key] for key in
        ('style_index', 'tone', 'structure', 'skin_structure', 'auto_structure')})
    width, height = render_extent(rgb.shape[1], rgb.shape[0], rendering['render_scale'], rendering['min_extent'])
    small = resample(rgb, width, height)
    geometry = network_geometry(width, height, rendering['min_extent'])
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
        condition = resample(condition, width, height)
        network[..., 7:10] = scaled_color(condition[geometry.source_rows()[:, None],
                                                   geometry.source_columns()[None, :]])
    return Frame(rgb, small, geometry, torch.from_numpy(network).unsqueeze(0), aligned,
                 confidence if aligned is not None else None, controls, profile, settings, rendering)


def student_arguments(frame, device, layout='BHWC'):
    g, rgb = frame.geometry, frame.small
    aligned = rgb[g.source_rows()[:, None], g.source_columns()[None, :]]
    image = torch.from_numpy(aligned.copy()).unsqueeze(0).to(device)
    history = np.zeros_like(aligned)
    if frame.history is not None:
        memory = frame.rgb + frame.confidence * (frame.history - frame.rgb)
        memory = resample(memory, rgb.shape[1], rgb.shape[0])
        history = memory[g.source_rows()[:, None], g.source_columns()[None, :]]
    history = torch.from_numpy(history.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    s, c, rendering = frame.settings, frame.controls, frame.rendering
    scalar = [c['normalized_style'] * 64, c['local_tone_strength'] / 2,
        c['local_structure_strength'] / 2,
        (s['skin_structure'] + 1) / 3 if s['skin_structure'] is not None else -1,
        (s['auto_structure'] + 1) / 3 if s['auto_structure'] is not None else -1,
        s['intensity'] / 2, s['detail_strength'] / 2, s['colour_strength'] / 2,
        s['detail_radius'] / 64, math.sin(rendering['noise_index']),
        rendering['history_strength'], float(frame.history is not None)]
    vector = [float(frame.profile == name) for name in PROFILES] + scalar
    controls = torch.tensor([vector], device=device, dtype=torch.float32)
    if layout == 'BCHW':
        image = image.permute(0, 3, 1, 2).contiguous()
    return image, controls, history


def half_ste(value):
    return value + (value.half().float() - value).detach()


def grade_tensor(image, controls):
    preset = {1: (-.1, -.25, -.1), 2: (0, 0, -.15)}.get(round(controls['normalized_style'] * 128))
    tone = min(1, max(0, controls['local_tone_strength']))
    if preset is None or tone == 0:
        return image
    # Retain the reference's toward-zero history grid with a rounding STE.
    rounded = truncate_half(image.detach().cpu().numpy())
    image = image + (torch.from_numpy(rounded).to(image) - image).detach()
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
    colour = torch.from_numpy(frame.rgb).to(head.device).unsqueeze(0)
    neural = (colour + .25 * half_ste(raw[..., :3])).clamp(0, 1)
    if frame.history is not None:
        history = torch.from_numpy(frame.history.copy()).to(head.device).unsqueeze(0)
        confidence = torch.from_numpy(frame.confidence).to(head.device).unsqueeze(0)
        alpha = torch.sigmoid(half_ste(raw[..., 3:])) * float(np.float16(blend_scale))
        alpha = alpha * frame.rendering['history_strength'] * confidence
        neural = neural + alpha * (history - neural)
    stored = truncate_half(neural.detach().cpu().numpy()[0])
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
            'rgb': torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0), 'history': history,
            'inference_seconds': seconds}


def device_target(target, device):
    # Clone outside inference_mode so loss backward can save target tensors safely.
    return (target['head'].to(device).clone(), {k: v.to(device).clone() for k, v in target['features'].items()},
            target['rgb'].to(device).clone())


def frame_guides(sample, index, crop_size):
    if index == 0:
        return None, None
    motion, confidence = sample['motions'][index], sample['confidences'][index]
    if motion is None:
        raise ValueError('Missing Sintel flow for a consecutive frame')
    H, W = motion.shape[:2]
    if crop_size:
        top, left = max(0, (H - crop_size) // 2), max(0, (W - crop_size) // 2)
        motion = motion[top:top + crop_size, left:left + crop_size].copy()
        confidence = confidence[top:top + crop_size, left:left + crop_size].copy()
        h, w = motion.shape[:2]
        motion *= np.array([W / w, H / h], np.float32)
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
