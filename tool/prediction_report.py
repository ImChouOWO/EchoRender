"""Single-mode input previews and timed Teacher/Student prediction grids."""
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from model.student import EchoRenderStudent
from .distillation import augment_rgb
from .progress import BAR_FORMAT

MODES = ('original', 'noise', 'brightness', 'contrast', 'sharpness')


@dataclass(frozen=True)
class VariantInput:
    display_rgb: np.ndarray
    network_rgb: np.ndarray
    parameters: dict


def augment_image(rgb, mode, fraction, config, *, seed, sample_id):
    digest = hashlib.sha256(f'{seed}:{sample_id}:{mode}'.encode()).digest()
    sample_seed = int.from_bytes(digest[:8], 'little') % (2**63 - 1)
    image, parameters = augment_rgb(rgb, mode, fraction, noise_std=config['noise_std'],
        generator=torch.Generator().manual_seed(sample_seed))
    parameters['seed'] = sample_seed
    return image, parameters


def resize_rgb(rgb, inference_size):
    if type(inference_size) is not int or inference_size < 0:
        raise ValueError('inference_size must be a nonnegative integer')
    h, w = rgb.shape[:2]
    scale = min(1.0, inference_size / max(h, w)) if inference_size else 1.0
    scaled_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
    if scaled_hw == (h, w):
        return rgb
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(tensor, size=scaled_hw, mode='bilinear', align_corners=False,
        antialias=True)[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()


def prepare_variant(rgb, mode, fraction, config, *, seed, sample_id, inference_size):
    """Apply corruption after resizing, then map the same input change to display size."""
    resized = resize_rgb(rgb, inference_size)
    network_rgb, parameters = augment_image(resized, mode, fraction, config, seed=seed, sample_id=sample_id)
    delta = network_rgb - resized
    parameters.update(inference_delta_mae=float(np.abs(delta).mean()),
                      inference_delta_std=float(delta.std()))
    display_rgb = network_rgb
    if resized.shape != rgb.shape:
        mapped = F.interpolate(torch.from_numpy(delta).permute(2, 0, 1).unsqueeze(0),
            size=rgb.shape[:2], mode='bilinear', align_corners=False)
        display_rgb = np.clip(rgb + mapped[0].permute(1, 2, 0).numpy(), 0, 1)
    return VariantInput(display_rgb, network_rgb, parameters)


def bold_font(size):
    for name in ('DejaVuSans-Bold.ttf', 'arialbd.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError('Comparison images require DejaVu Sans Bold or Arial Bold')


def save_grid(path, variants, title, predictions=None):
    """One source per file: input row, followed by its matching prediction row."""
    inputs = [variant.display_rgb for variant in variants]
    h, w = inputs[0].shape[:2]
    if any(image.shape != inputs[0].shape or not np.isfinite(image).all() for image in inputs):
        raise ValueError('Grid inputs must be finite RGB images of identical shape')
    rows = [inputs]
    if predictions is not None:
        if len(predictions) != len(inputs) or any(
                image.shape != inputs[0].shape or not np.isfinite(image).all() for image in predictions):
            raise ValueError('Each grid input needs a matching finite prediction')
        rows.append(predictions)
    cell_w = min(384, w)
    cell_h = max(1, round(h * cell_w / w))
    gap, margin = 12, 16
    content_w = len(inputs) * cell_w + (len(inputs) - 1) * gap
    width = max(640, content_w + 2 * margin)
    row_tops = [132 + row * (cell_h + 48) for row in range(len(rows))]
    height = row_tops[-1] + cell_h + margin
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    def label(text, left, top, maximum, size):
        while size > 1 and draw.textbbox((0, 0), text, font=bold_font(size))[2] > maximum:
            size -= 1
        draw.text((left, top), text, font=bold_font(size), fill='black')
    label(title, margin, 12, width - 2 * margin, 22)
    lefts = [(width - content_w) // 2 + i * (cell_w + gap) for i in range(len(inputs))]
    for left, variant in zip(lefts, variants):
        parameters = variant.parameters
        if parameters['mode'] == 'original':
            caption = 'Original'
        elif parameters['mode'] == 'noise':
            caption = f'{parameters["fraction"]:.0%} | sigma {parameters["sigma"]:.3f}'
        else:
            caption = f'{parameters["fraction"]:.0%} | factor {parameters["factor"]:.2f}'
        label(caption, left, 58, cell_w, 18)
    for row, images in enumerate(rows):
        label('Input' if row == 0 else 'Pred', margin, row_tops[row] - 30, width - 2 * margin, 18)
        for left, image in zip(lefts, images):
            panel = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
            if panel.size != (cell_w, cell_h):
                panel = panel.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(panel, (left, row_tops[row]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {'canvas_hw': [height, width], 'cell_hw': [cell_h, cell_w],
            'row_tops': row_tops, 'column_lefts': lefts, 'title': title}


def output_name(split, sample_id):
    path = Path(sample_id)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Invalid prediction sample ID')
    return f'{split}__{"__".join(path.parts)}.png'


def release_model(model, device):
    model.cpu()
    gc.collect()
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def sample_modes(loader, maximum, seed):
    """Select physical clips once; all profiles use the identical selected sources."""
    dataset = loader.dataset
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = sorted({base.items[i][0] for i in dataset.indices})
    else:
        base, indices = dataset, list(range(len(dataset.clips)))
    generator = random.Random(seed)
    chosen = generator.sample(indices, min(len(indices), maximum * len(MODES)))
    groups = {}
    for mode_index, mode in enumerate(MODES):
        subset = chosen[mode_index * maximum:(mode_index + 1) * maximum]
        needed = min(maximum, len(indices)) - len(subset)
        if needed:
            subset += generator.sample([i for i in indices if i not in subset], needed)
        groups[mode] = [base[base.items.index((i, base.profiles[0]))] for i in subset]
    return groups, len(indices)


@torch.inference_mode()
def predict_clip(model, sample, mode, fraction, profile, config, device, blend_scale, is_teacher):
    from .temporal import (crop_rgb, prepare_frame, prepare_student_frame, student_inputs, compose_tensor, teacher_target,
                           frame_guides, reset_history, resample, student_previous)
    history = previous = previous_student = None
    total = 0.
    last_prediction = last_variant = None
    geometry = {}
    for position, image in enumerate(sample['images']):
        rgb = crop_rgb(image, config['reporting']['crop_size'])
        variant = prepare_variant(rgb, mode, fraction, config['augmentation'], seed=config['loader']['seed'],
            sample_id=sample['sample_ids'][position], inference_size=config['reporting']['inference_size'])
        if reset_history(previous, variant.network_rgb, config):
            history = previous_student = None
        if is_teacher:
            motion, confidence = frame_guides(sample, position, config['reporting']['crop_size'])
            if motion is not None:
                h, w = variant.network_rgb.shape[:2]
                motion, confidence = resample(motion, w, h), resample(confidence, w, h)
            frame = prepare_frame(variant.network_rgb, config, profile, history=history,
                                  motion=motion, confidence=confidence)
            target = teacher_target(model, frame)
            prediction = target['rgb'][0].permute(1, 2, 0).numpy()
            next_history = target['history']
            seconds = target['inference_seconds']
            input_shape = list(frame.inputs.shape)
        else:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = perf_counter()
            frame = prepare_student_frame(variant.network_rgb, config, profile, model,
                previous=previous_student, history=history)
            inputs = student_inputs(frame, device, model.layout)
            head = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds = perf_counter() - started
            if model.layout == 'BCHW':
                head = head.permute(0, 2, 3, 1)
            output, next_history = compose_tensor(head, frame, blend_scale)
            prediction = output[0].permute(1, 2, 0).cpu().numpy()
            input_shape = list(inputs.shape)
        total += seconds
        # Show the actual augmented input; map only the model correction back to native size.
        correction = resample(prediction - variant.network_rgb, rgb.shape[1], rgb.shape[0])
        last_prediction = np.clip(variant.display_rgb + correction, 0, 1)
        last_variant = variant
        geometry = {'model_input_shape': input_shape, 'history_used': frame.history is not None,
            'frame_index': position + 1, 'sample_id': sample['sample_ids'][position],
            'inference_input_sha256': hashlib.sha256(variant.network_rgb.tobytes()).hexdigest(),
            'input_sha256': hashlib.sha256(variant.display_rgb.tobytes()).hexdigest(),
            'prediction_sha256': hashlib.sha256(last_prediction.tobytes()).hexdigest(),
            'network_hw': list(frame.inputs.shape[1:3]), 'parameters': variant.parameters,
            'motion_source': 'sintel' if is_teacher and sample['sample_ids'][0].startswith('training/') else
                             'raw' if is_teacher else model.motion_mode}
        history = next_history if config['rendering']['temporal'] else None
        previous = variant.network_rgb
        if not is_teacher:
            previous_student = student_previous(frame, model)
    return last_variant, last_prediction, {**geometry, 'inference_seconds': total,
        'inference_frames': len(sample['images'])}


@torch.inference_mode()
def export_predictions(checkpoint, teacher_factory, loaders, config, directory, device, *, missing_splits=None):
    from .temporal import FORMAT
    device, directory = torch.device(device), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Expected a current BHWC16 motion checkpoint; retraining is required')
    config = dict(config, rendering=payload['train_config']['rendering'])
    blend_scale = payload['teacher_blend_scale']
    manifest = {'status': 'running', 'checkpoint': str(checkpoint), 'checkpoint_epoch': payload['epoch'],
        'rendering': config['rendering'], 'missing_splits': missing_splits or {},
        'selection': {}, 'grids': [], 'counts': {'preview': {}, 'student': {}, 'teacher': {}}, 'timing_scope': 'Student includes motion, input preparation and backbone; Teacher is forward only; totals include context frames'}
    cases = {}
    for split, loader in loaders.items():
        groups, available = sample_modes(loader, config['reporting']['samples_per_mode'], config['loader']['seed'])
        manifest['selection'][split] = {mode: [s['sample_ids'][-1] for s in samples] for mode, samples in groups.items()}
        cases[split] = groups
    total_variants = 2 * len(config['rendering']['profiles']) * sum(
        len(samples) * (1 if mode == 'original' else 1 + len(config['reporting']['strengths']))
        for groups in cases.values() for mode, samples in groups.items())
    model = None
    try:
        with (
            tqdm(total=total_variants, desc='Predict', unit='variant', dynamic_ncols=True,
                 bar_format=BAR_FORMAT, disable=not config['logging']['progress']) as progress,
            (directory / 'predictions.jsonl').open('w') as stream,
            (directory / 'predictions.csv').open('w', newline='') as table,
        ):
            columns = ['model', 'split', 'profile', 'mode', 'sample_id', 'strength', 'sigma', 'factor',
                       'input_delta_mae', 'prediction_change_rmse', 'inference_seconds', 'inference_frames', 'path']
            writer = csv.DictWriter(table, fieldnames=columns)
            writer.writeheader()
            for name in ('student', 'teacher'):
                model = EchoRenderStudent(payload['student_config']) if name == 'student' else teacher_factory()
                if name == 'student':
                    model.load_state_dict(payload['student'], strict=True)
                model.eval().requires_grad_(False).to(device)
                if name == 'teacher' and device.type == 'cuda' and hasattr(model.model, 'enable_cuda_graph'):
                    model.model.enable_cuda_graph()
                for split, groups in cases.items():
                    for mode, samples in groups.items():
                        for sample in samples:
                            for profile in config['rendering']['profiles']:
                                levels = [0.] if mode == 'original' else [0., *config['reporting']['strengths']]
                                progress.set_description_str(
                                    f'Predict [{split}] {name:<7} | {profile:<9} | {mode:<10}', refresh=False)
                                variants, predictions, geometries = [], [], []
                                for level in levels:
                                    variant, prediction, geometry = predict_clip(model, sample,
                                        'original' if level == 0 else mode, level, profile, config, device,
                                        blend_scale, name == 'teacher')
                                    variants.append(variant)
                                    predictions.append(prediction)
                                    geometries.append(geometry)
                                    progress.update(1)
                                sample_id = sample['sample_ids'][-1]
                                filename = output_name(split, sample_id)
                                if name == 'student' and profile == next(iter(config['rendering']['profiles'])):
                                    preview = Path('preview') / mode / filename
                                    save_grid(directory / preview, variants, f'{mode.title()} | {sample_id} | Input preview')
                                total = sum(g['inference_seconds'] for g in geometries)
                                frames = sum(g['inference_frames'] for g in geometries)
                                average = total / frames
                                relative = Path('pred') / name / profile / mode / filename
                                title = f'{name.title()} | {profile} | {mode} | frame {len(sample["images"])} | total infer {total:.3f} s | avg infer {average:.3f} s |'
                                grid = save_grid(directory / relative, variants, title, predictions)
                                manifest['grids'].append({'model': name, 'split': split, 'profile': profile, 'mode': mode,
                                    'sample_id': sample_id, 'path': relative.as_posix(), 'total_infer_seconds': total,
                                    'avg_infer_seconds': average, **grid})
                                for level, geometry, prediction, variant in zip(levels, geometries, predictions, variants):
                                    row = {'model': name, 'split': split, 'profile': profile, 'mode': mode,
                                        'sample_id': sample_id, 'strength': level,
                                        'sigma': variant.parameters.get('sigma', ''), 'factor': variant.parameters.get('factor', ''),
                                        'input_delta_mae': variant.parameters['inference_delta_mae'],
                                        'prediction_change_rmse': float(np.sqrt(np.mean((prediction - predictions[0]) ** 2))),
                                        'inference_seconds': geometry['inference_seconds'],
                                        'inference_frames': geometry['inference_frames'], 'path': relative.as_posix()}
                                    writer.writerow(row)
                                    stream.write(json.dumps({**row, **geometry}) + '\n')
                                stream.flush()
                                table.flush()
                release_model(model, device)
                model = None
        for split in cases:
            manifest['counts']['preview'][split] = sum(len(samples) for samples in cases[split].values())
            for name in ('student', 'teacher'):
                manifest['counts'][name][split] = sum(g['model'] == name and g['split'] == split for g in manifest['grids'])
        manifest['status'] = 'completed_with_missing_splits' if missing_splits else 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        if model is not None:
            release_model(model, device)
        (directory / 'predictions.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
