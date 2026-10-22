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
from .teacher_pipeline import prepare_image, compose_rgb_tensor

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


def sample_modes(loader, maximum, seed):
    """Use distinct sources across modes when enough exist; never decode unselected images."""
    if hasattr(loader, 'dataset'):
        available = len(loader.dataset)
        def read(index):
            sample = loader.dataset[index]
            return sample['image'], sample['sample_id']
    else:
        samples = [(image, sample_id) for batch in loader
                   for image, sample_id in zip(batch['images'], batch['sample_ids'])]
        available = len(samples)
        def read(index):
            return samples[index]
    if not available:
        raise ValueError('Prediction loader has no images')
    generator = random.Random(seed)
    if available >= maximum * len(MODES):
        selected = generator.sample(range(available), maximum * len(MODES))
        indices = {mode: selected[i * maximum:(i + 1) * maximum] for i, mode in enumerate(MODES)}
    else:
        indices = {mode: generator.sample(range(available), min(maximum, available)) for mode in MODES}
    decoded = {i: read(i) for i in {i for group in indices.values() for i in group}}
    return {mode: [decoded[i] for i in group] for mode, group in indices.items()}, available


@torch.inference_mode()
def predict_resized(model, variant, *, device, blend_scale, layout='BHWC', teacher_graph=False, warmup=False):
    """Time only synchronized model forward; compose RGB before mapping its correction back."""
    device = torch.device(device)
    rgb, resized = variant.display_rgb, variant.network_rgb
    h, w = rgb.shape[:2]
    prepared = prepare_image(resized)
    inputs = prepared.features.to(device)
    if layout == 'BCHW':
        inputs = inputs.permute(0, 3, 1, 2).contiguous()
    def forward():
        if teacher_graph:
            head, captured = model(inputs, return_features=True)
            del captured
            return head
        return model(inputs)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    if warmup:
        unused = forward()
        del unused
    synchronize()
    started = perf_counter()
    head = forward()
    synchronize()
    seconds = perf_counter() - started
    if layout == 'BCHW':
        head = head.permute(0, 2, 3, 1).contiguous()
    if tuple(head.shape) != (*prepared.features.shape[:-1], 4):
        raise RuntimeError('Inference returned an invalid four-channel head')
    prediction = compose_rgb_tensor(head, prepared, blend_scale=blend_scale)
    if resized.shape[:2] != (h, w):
        base = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)
        correction = F.interpolate(prediction - base, size=(h, w), mode='bilinear', align_corners=False)
        full_input = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        prediction = (full_input + correction).clamp(0, 1)
    prediction = prediction[0].permute(1, 2, 0).cpu().numpy()
    geometry = {'image_hw': [h, w], 'resized_hw': list(resized.shape[:2]),
        'network_hw': list(prepared.features.shape[1:3]), 'inference_seconds': seconds,
        'input_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'inference_input_sha256': hashlib.sha256(resized.tobytes()).hexdigest(),
        'prepared_features_sha256': hashlib.sha256(prepared.features.numpy().tobytes()).hexdigest(),
        'prediction_sha256': hashlib.sha256(prediction.tobytes()).hexdigest()}
    return prediction, geometry


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


@torch.inference_mode()
def export_predictions(checkpoint, teacher_factory, loaders, config, directory, device, *, missing_splits=None):
    device, directory = torch.device(device), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != 'echorender-ssm-v2':
        raise ValueError('Prediction reports require an echorender-ssm-v2 checkpoint')
    epoch, blend_scale = payload['epoch'], payload['teacher_blend_scale']
    student_config, student_state = payload['student_config'], payload['student']
    del payload
    manifest = {'status': 'running', 'report_format': 'augmentation-grids-v1',
        'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_epoch': epoch,
        'teacher_weights': config['model']['teacher_weights'], 'device': str(device),
        'crop_size': config['reporting']['crop_size'], 'inference_size': config['reporting']['inference_size'],
        'samples_per_mode': config['reporting']['samples_per_mode'], 'seed': config['loader']['seed'],
        'strengths': config['reporting']['strengths'], 'noise_std': config['augmentation']['noise_std'],
        'augmentation': 'one mode per validation input; factor=1+strength; sigma=noise_std*strength',
        'mapping': 'compose RGB at inference size; bilinear-map RGB correction onto original-size augmented input',
        'timing_scope': 'synchronized model forward in seconds; excludes warmup, preprocessing, RGB composition and file I/O',
        'selection': {}, 'counts': {'preview': {}}, 'inference_counts': {}, 'grids': [],
        'missing_splits': missing_splits or {}}
    manifest_path = directory / 'predictions.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    cases, model = {}, None
    try:
        for split, loader in loaders.items():
            seed = int.from_bytes(hashlib.sha256(f'{config["loader"]["seed"]}:{split}'.encode()).digest()[:8], 'little')
            groups, available = sample_modes(loader, config['reporting']['samples_per_mode'], seed)
            manifest['selection'][split] = {'available_images': available, 'modes': {}}
            cases[split] = []
            for mode, samples in groups.items():
                manifest['selection'][split]['modes'][mode] = [sid for _, sid in samples]
                for image, sample_id in samples:
                    rgb = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
                    crop = config['reporting']['crop_size']
                    if crop:
                        h, w = rgb.shape[:2]
                        top, left = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
                        rgb = rgb[top:top + crop, left:left + crop]
                    levels = [0.] if mode == 'original' else [0., *config['reporting']['strengths']]
                    variants = [prepare_variant(rgb, 'original' if value == 0 else mode, value,
                        config['augmentation'], seed=config['loader']['seed'], sample_id=sample_id,
                        inference_size=config['reporting']['inference_size']) for value in levels]
                    filename = output_name(split, sample_id)
                    relative = Path('preview') / mode / filename
                    grid = save_grid(directory / relative, variants, f'{mode.title()} | {sample_id} | Input preview')
                    manifest['grids'].append({'kind': 'preview', 'split': split, 'mode': mode,
                        'sample_id': sample_id, 'path': relative.as_posix(), **grid})
                    cases[split].append((mode, sample_id, filename, variants))
            manifest['counts']['preview'][split] = len(cases[split])
            tqdm.write(f'[Preview]: {split} | {len(cases[split])} grids | {config["reporting"]["samples_per_mode"]} sources per mode')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        with (directory / 'predictions.jsonl').open('x') as stream, \
             (directory / 'predictions.csv').open('x', newline='') as table:
            columns = ('model', 'split', 'mode', 'sample_id', 'strength', 'sigma', 'factor',
                'input_delta_mae', 'prediction_change_rmse', 'inference_seconds',
                'total_infer_seconds', 'avg_infer_seconds', 'path')
            writer = csv.DictWriter(table, fieldnames=columns)
            writer.writeheader()
            for name in ('student', 'teacher'):
                if name == 'student':
                    model = EchoRenderStudent(student_config)
                    model.load_state_dict(student_state, strict=True)
                    del student_state
                else:
                    model = teacher_factory()
                model.eval().requires_grad_(False).to(device)
                graph_teacher = name == 'teacher' and device.type == 'cuda' and hasattr(
                    getattr(model, 'model', None), 'enable_cuda_graph')
                if graph_teacher:
                    model.model.enable_cuda_graph()
                warmed_shapes, counts, inference_counts = set(), {}, {}
                for split, samples in cases.items():
                    count = inference_count = 0
                    bar = tqdm(samples, desc=f'[Predict] {name.title()} {split}', unit='grid',
                        dynamic_ncols=True, disable=not config['logging']['progress'])
                    try:
                        for mode, sample_id, filename, variants in bar:
                            predictions, geometries = [], []
                            for variant in variants:
                                shape = variant.network_rgb.shape
                                prediction, geometry = predict_resized(model, variant, device=device,
                                    blend_scale=blend_scale, layout=model.layout if name == 'student' else 'BHWC',
                                    teacher_graph=graph_teacher, warmup=shape not in warmed_shapes)
                                warmed_shapes.add(shape)
                                predictions.append(prediction)
                                geometries.append(geometry)
                            total = sum(g['inference_seconds'] for g in geometries)
                            average = total / len(geometries)
                            relative = Path('pred') / name / mode / filename
                            title = (f'{name.title()} | {mode.title()} | {sample_id} | '
                                     f'total infer {total:.3f} s | avg infer {average:.3f} s |')
                            grid = save_grid(directory / relative, variants, title, predictions)
                            manifest['grids'].append({'kind': name, 'split': split, 'mode': mode,
                                'sample_id': sample_id, 'path': relative.as_posix(),
                                'total_infer_seconds': total, 'avg_infer_seconds': average, **grid})
                            for variant, prediction, geometry in zip(variants, predictions, geometries):
                                parameters = variant.parameters
                                change = float(np.sqrt(np.mean((prediction - predictions[0]) ** 2)))
                                record = {'model': name, 'split': split, 'mode': mode, 'sample_id': sample_id,
                                    'path': relative.as_posix(), 'parameters': parameters, 'grid': grid,
                                    'prediction_change_rmse': change, 'total_infer_seconds': total,
                                    'avg_infer_seconds': average, **geometry}
                                stream.write(json.dumps(record) + '\n')
                                writer.writerow({'model': name, 'split': split, 'mode': mode, 'sample_id': sample_id,
                                    'strength': parameters['fraction'], 'sigma': parameters.get('sigma', ''),
                                    'factor': parameters.get('factor', ''), 'input_delta_mae': parameters['inference_delta_mae'],
                                    'prediction_change_rmse': change, 'inference_seconds': geometry['inference_seconds'],
                                    'total_infer_seconds': total, 'avg_infer_seconds': average, 'path': relative.as_posix()})
                            stream.flush()
                            table.flush()
                            count += 1
                            inference_count += len(variants)
                    finally:
                        bar.close()
                    counts[split], inference_counts[split] = count, inference_count
                release_model(model, device)
                model = None
                gc.collect()
                manifest['counts'][name] = counts
                manifest['inference_counts'][name] = inference_counts
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        manifest['status'] = 'completed_with_missing_splits' if missing_splits else 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        if model is not None:
            release_model(model, device)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
