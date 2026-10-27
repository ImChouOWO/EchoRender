"""Edit demo() and run `python teacher_infer.py`; no command-line options are needed.

Python API: run_inference(scene='alley_1', max_frames=2, project='example').
Use infer_image(teacher, rgb, ...) to reuse a loaded Teacher across experiments.
Input/host logic follows dlss-nr-on-intel at the reference revision below.
The neural graph executes through EchoRender's pinned MLX-DLSS PyTorch backend.
"""
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from mlxdlss.composition import compose_detail
from mlxdlss.features import AutomaticMask, NetworkGeometry, PROFILES, half, make_features, scaled_color
from mlxdlss.temporal import sample_history

from model.teacher import EchoRenderTeacher, resolve_device
from tool.teacher_pipeline import load_rgb, save_rgb
from tool.training_config import DEFAULT_TRAIN_CONFIG, load_training_config, reserve_output_dir

REFERENCE_REVISION = '2f90064d6a405ac956b4f8ace61c557e61fe13a2'
REFERENCE_URL = 'https://github.com/uzbekunknown/dlss-nr-on-intel/tree/' + REFERENCE_REVISION


def demo():
    """Preview Teacher enhancement on the current Sintel dataset."""
    return run_inference(
        input_image=None,
        scene="alley_1",
        render_pass="clean",
        max_frames=10,
        project="teacher_sintel_clean",
        device="cuda:1",

        # Preserve native detail for inspecting Teacher targets.
        render_scale=1.0,
        min_extent=320,

        profile="neutral",
        style_index=2,
        tone=None,
        structure=1.8,
        skin_structure=1.5,
        auto_structure=1,
        noise_index=1,

        intensity=1.8,
        detail_strength=1.5,
        colour_strength=1,
        detail_radius=0.5,

        # Reproject previous neural predictions with Sintel flow.
        temporal=True,
        motion_mode="sintel",
        inspect_data=True,
    )


# Fixed Teacher input contract (BHWC float32, 16 channels):
# 0:3 deterministic noise; 3=1; 4:7 scaled current RGB; 7:10 scaled history/current RGB;
# 10=style/128; 11=tone; 12=structure; 13=skin structure; 14=automatic structure; 15=0.
# RGB inputs are [0,1]. Internal noise is separate from RGB augmentation.
# Only clean RGB is accepted; final/albedo are listed for inventory diagnostics.
# flow_viz is visualization, not motion. .flo is forward pixel motion, so it must be inverted.
# Occlusions/invalid mark unreliable source pixels; neither is an artist control mask.
# The current download has no depth/normal input, and the network has no independent slots for them.


def field_alignment(extent):
    reductions = 0
    for level in range(6):
        following = (((extent + 1) // 2 + 3) // 4) * 4
        reductions += int(following < extent) + int(level == 0 and following % 8 != 0)
        extent = following
    return 2 ** reductions


def network_geometry(width, height, minimum=320):
    """Intel host field rule, including extra width at four-alignment boundaries."""
    ax, ay = field_alignment(width), field_alignment(height)
    nw = max(128, minimum, ((width + ax - 1) // ax) * ax)
    nh = max(128, minimum, ((height + ay - 1) // ay) * ay)
    if nw % (4 * ax) == 0 and nh % (4 * ay) == 0:
        nw += ax
    return NetworkGeometry(width, height, ((nw + 63) // 64) * 64, ((nh + 63) // 64) * 64)


def render_extent(width, height, scale, minimum=320):
    """Fill the cheapest field at or above the requested scale, preserving aspect."""
    if scale == 1:
        return width, height
    best = (max(64, round(width * scale)), max(64, round(height * scale)))
    geometry = network_geometry(*best, minimum)
    cost = geometry.network_width * geometry.network_height
    for w in range(best[0] + 1, width + 1):
        candidate = (w, max(64, round(height * w / width)))
        geometry = network_geometry(*candidate, minimum)
        current_cost = geometry.network_width * geometry.network_height
        if current_cost <= cost:
            best, cost = candidate, current_cost
    return best


def resample(image, width, height):
    """Intel host's area mean for integer reduction, otherwise separable bilinear sampling."""
    image = np.asarray(image, np.float32)
    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image
    if h > height and w > width and h % height == 0 and w % width == 0:
        fy, fx = h // height, w // width
        blocks = image.reshape(height, fy, width, fx, -1)
        result = blocks[:, 0, :, 0].copy()
        for y in range(fy):
            for x in range(fx):
                if y or x:
                    result += blocks[:, y, :, x]
        return result / np.float32(fy * fx)
    for axis, size in ((0, height), (1, width)):
        extent = image.shape[axis]
        if extent == size:
            continue
        points = (np.arange(size, dtype=np.float32) + .5) * (extent / size) - .5
        low = np.clip(np.floor(points), 0, extent - 1).astype(np.int32)
        high = np.minimum(low + 1, extent - 1)
        weight = np.clip(points - low, 0, 1).astype(np.float32)
        weight = weight.reshape((-1, 1, 1) if axis == 0 else (1, -1, 1))
        image = np.take(image, low, axis) * (1 - weight) + np.take(image, high, axis) * weight
    return image


def truncate_half(image):
    """Store neural history toward zero on the FP16 grid, before display processing."""
    rounded = image.astype(np.float16)
    lower = np.nextafter(rounded, np.float16(0))
    return np.where(np.abs(rounded.astype(np.float32)) > np.abs(image), lower, rounded).astype(np.float32)


def style_grade(image, controls):
    """Style 1/2 exposure, smoothstep contrast and HSL saturation, scaled by tone."""
    presets = {1: (-.1, -.25, -.1), 2: (0., 0., -.15)}
    preset = presets.get(round(controls['normalized_style'] * 128))
    tone = np.float32(np.clip(controls['local_tone_strength'], 0, 1))
    if preset is None or tone == 0:
        return image
    exposure, contrast, saturation = (np.float32(value) * tone for value in preset)
    image = np.clip(truncate_half(image) * np.float32(2 ** float(exposure)), 0, 1)
    image = np.clip(image + contrast * (image * image * (np.float32(3) - (image + image)) - image), 0, 1)
    high, low = image.max(axis=-1), image.min(axis=-1)
    light = (high + low) * np.float32(.5)
    delta = high - low
    span = np.where(light > .5, (np.float32(2) - high) - low, high + low)
    chroma = np.divide(delta, span, out=np.zeros_like(delta), where=span > 0)
    chroma = np.clip(chroma * (saturation + np.float32(1)), 0, 1)
    ratio = np.divide(span * chroma, delta, out=np.zeros_like(delta), where=delta > 0)
    return np.clip(light[..., None] + (image - light[..., None]) * ratio[..., None], 0, 1).astype(np.float32)


def validate_range(value, name, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be finite and numeric')
    if not minimum <= value <= maximum or (integer and type(value) is not int):
        raise ValueError(f'{name} must be within [{minimum},{maximum}]' + (' and an integer' if integer else ''))


def network_controls(profile='standard', *, style_index=None, tone=None, structure=None,
                     skin_structure=None, auto_structure=None, control_mask=None):
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    controls = dict(PROFILES[profile])
    for name, value in (('tone', tone), ('structure', structure)):
        if value is not None:
            validate_range(value, name, 0, 2)
            controls['local_' + name + '_strength'] = value
    if style_index is not None:
        validate_range(style_index, 'style_index', 0, 2, integer=True)
        controls['normalized_style'] = style_index / 128
    automatic = None
    if skin_structure is not None or auto_structure is not None:
        if control_mask is not None:
            raise ValueError('Choose a control mask or automatic structure controls, not both')
        for value in (skin_structure, auto_structure):
            if value is not None:
                validate_range(value, 'automatic structure', -1, 2)
        automatic = AutomaticMask(-1 if skin_structure is None else skin_structure,
                                  -1 if auto_structure is None else auto_structure)
    return controls, automatic


def compose_output(head, rgb, controls, *, blend_scale, history=None, confidence=None,
                   previous_rgb=None, history_strength=1., history_hold=1., history_release=24.,
                   intensity=1., control_mask=None, detail_strength=1., colour_strength=1., detail_radius=4.):
    """Compose at native size; the first frame ignores the gate, as in the Intel host."""
    prediction = np.clip(rgb + half(head[..., :3]) * np.float32(.25), 0, 1)
    if history is not None:
        with np.errstate(over='ignore'):
            alpha = np.clip(1 / (1 + np.exp(-half(head[..., 3:4]))) * half(blend_scale), 0, 1)
        alpha *= np.float32(history_strength)
        if confidence is not None:
            alpha *= confidence
        if previous_rgb is not None:
            moved = np.abs(rgb - previous_rgb).max(axis=-1, keepdims=True)
            if history_release:
                alpha *= np.clip(np.float32(1) + moved * np.float32(-255 / history_release), 0, 1)
            floor = np.clip(moved * np.float32(-255 * history_hold / 4) + np.float32(history_hold), 0, history_hold)
            np.maximum(alpha, floor * np.float32(blend_scale), out=alpha)
        prediction += alpha * (history - prediction)
    neural_history = truncate_half(prediction)
    prediction = style_grade(prediction, controls)
    blend = np.float32(intensity)
    if control_mask is not None:
        blend = control_mask[..., :1] * blend
    output = np.clip(rgb + blend * (prediction - rgb), 0, 1).astype(np.float32)
    output = compose_detail(rgb, output, detail_strength=detail_strength,
                            colour_strength=colour_strength, radius=detail_radius)
    return output, neural_history


@torch.inference_mode()
def infer_image(teacher, rgb, *, controls=None, automatic_mask=None, render_scale=1., min_extent=320,
                noise_index=0, history=None, motion=None, confidence=None, previous_rgb=None,
                control_mask=None, **composition):
    """Reuse a loaded Teacher; return native RGB, pre-display history and timings."""
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('RGB must be finite HWC float32 in [0,1]')
    controls = dict(PROFILES['standard']) if controls is None else controls
    started = perf_counter()
    width, height = render_extent(rgb.shape[1], rgb.shape[0], render_scale, min_extent)
    small = resample(rgb, width, height)
    geometry = network_geometry(width, height, min_extent)
    if control_mask is not None and control_mask.shape != rgb.shape:
        raise ValueError('Control mask must match the original RGB dimensions')
    mask = resample(control_mask, width, height) if control_mask is not None else None
    aligned_history = history
    if history is not None:
        if history.shape != rgb.shape:
            raise ValueError('History must match native RGB dimensions')
        if motion is not None:
            yy, xx = np.indices(rgb.shape[:2], dtype=np.float32)
            aligned_history = sample_history(history, (xx + .5) / rgb.shape[1] + motion[..., 0],
                                            (yy + .5) / rgb.shape[0] + motion[..., 1])
        history_condition = aligned_history
        if confidence is not None:
            history_condition = rgb + confidence * (history_condition - rgb)
        history_condition = resample(history_condition, width, height)
    network = make_features(small, geometry=geometry, frame_index=noise_index,
                            automatic_mask=automatic_mask, control_mask=mask, **controls)
    if history is not None:
        network[..., 7:10] = scaled_color(history_condition[geometry.source_rows()[:, None],
                                                           geometry.source_columns()[None, :]])
    prepared_at = perf_counter()
    inputs = torch.from_numpy(network).unsqueeze(0).to(teacher.device)
    if teacher.device.type == 'cuda':
        torch.cuda.synchronize(teacher.device)
    forward_at = perf_counter()
    head = geometry.crop(teacher(inputs).cpu().numpy()[0])
    inferred_at = perf_counter()
    # Intel's host enlarges the raw head BEFORE composition; it does not enlarge composed RGB.
    head = resample(head, rgb.shape[1], rgb.shape[0])
    prediction, neural = compose_output(head, rgb, controls, blend_scale=float(teacher.blend_scale),
        history=aligned_history, confidence=confidence, previous_rgb=previous_rgb,
        control_mask=control_mask, **composition)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Teacher output contains non-finite values')
    return prediction, neural, {'input_bhwc': list(inputs.shape), 'processing_hw': [height, width],
        'history_used': history is not None,
        'history_valid_fraction': (float(confidence.mean()) if confidence is not None else 1.) if history is not None else None,
        'seconds': {'prepare': prepared_at - started, 'transfer': forward_at - prepared_at,
                    'inference': inferred_at - forward_at, 'compose': perf_counter() - inferred_at}}


def read_flow(path):
    """Read little-endian Middlebury .flo pixel displacements, preserving x/y order."""
    with Path(path).open('rb') as source:
        if source.read(4) != b'PIEH':
            raise ValueError(f'Invalid .flo signature: {path}')
        shape = np.frombuffer(source.read(8), dtype='<i4')
        if shape.size != 2 or min(shape) < 1:
            raise ValueError(f'Invalid .flo dimensions: {path}')
        width, height = map(int, shape)
        values = np.frombuffer(source.read(), dtype='<f4')
    if values.size != height * width * 2:
        raise ValueError(f'Invalid .flo payload size: {path}')
    return values.reshape(height, width, 2).copy()

def read_invalid(path, shape):
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L')) != 0
    if mask.shape != shape:
        raise ValueError(f'Mask dimensions differ from the source RGB: {path}')
    return mask

def invert_forward_flow(flow, valid):
    """Bilinearly splat source coordinates; inversion is approximate, not just -flow."""
    height, width = flow.shape[:2]
    if flow.shape != (height, width, 2) or valid.shape != (height, width):
        raise ValueError('Flow and source validity dimensions must match')
    yy, xx = np.indices((height, width), dtype=np.float64)
    valid = valid & np.isfinite(flow).all(axis=2) & (np.abs(flow) < 1e9).all(axis=2)
    sx, sy = xx[valid], yy[valid]
    tx, ty = sx + flow[..., 0][valid], sy + flow[..., 1][valid]
    x0, y0 = np.floor(tx).astype(np.int64), np.floor(ty).astype(np.int64)
    accum = np.zeros((4, height * width), dtype=np.float64)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = x0 + dx, y0 + dy
        weight = (1 - np.abs(tx - x)) * (1 - np.abs(ty - y))
        keep = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (weight > 0)
        index, weight = y[keep] * width + x[keep], weight[keep]
        for target, value in zip(accum, (np.ones_like(sx), sx, sy, sx * sx + sy * sy)):
            target += np.bincount(index, weights=weight * value[keep], minlength=height * width)
    mass, sum_x, sum_y, square = accum.reshape(4, height, width)
    divisor = np.maximum(mass, 1e-12)
    px, py = sum_x / divisor, sum_y / divisor
    variance = np.maximum(0, square / divisor - px * px - py * py)
    # Ordinary bilinear neighbours have <=0.5 px^2 spread. Larger collisions are ambiguous.
    accepted = (mass > 1e-6) & (variance <= 1.0)
    motion = np.stack(((px - xx) / width, (py - yy) / height), axis=-1).astype(np.float32)
    motion[~accepted] = 0
    confidence = np.where(accepted, np.minimum(mass, 1), 0).astype(np.float32)[..., None]
    return motion, confidence

def dataset_motion(root, source, scene, previous, current, shape):
    base = root / source
    paths = {name: base / name / scene / (previous.stem + suffix)
             for name, suffix in (('flow', '.flo'), ('occlusions', '.png'), ('invalid', '.png'))}
    paths['current_invalid'] = base / 'invalid' / scene / current.name
    flow = read_flow(paths['flow'])
    if flow.shape[:2] != shape:
        raise ValueError('Sintel flow and RGB dimensions must match')
    valid = ~(read_invalid(paths['occlusions'], shape) | read_invalid(paths['invalid'], shape))
    motion, confidence = invert_forward_flow(flow, valid)
    confidence[read_invalid(paths['current_invalid'], shape)] = 0
    return motion, confidence, {name: str(path) for name, path in paths.items()}

def inspect_dataset(root, source):
    roles = {'clean': 'rendered RGB input', 'final': 'rendered RGB input with final effects',
             'albedo': 'diagnostic RGB alternative; not an additional model input',
             'flow': 'forward pixel motion; invert to reproject output history',
             'occlusions': 'exclude unmatched source pixels before flow inversion',
             'invalid': 'exclude invalid source and destination pixels',
             'flow_viz': 'visualization only; cannot replace .flo motion'}
    result = {}
    for name, role in roles.items():
        path = root / source / name
        extension = '*.flo' if name == 'flow' else '*.png'
        result[name] = {'files': sum(1 for _ in path.rglob(extension)) if path.is_dir() else 0,
                        'path': str(path), 'use': role}
    return result


def run_inference(*, input_image=None, scene=None, project='teacher', config_path=DEFAULT_TRAIN_CONFIG,
                  config=None, dataset_root=None, source='training', render_pass='clean', max_frames=10,
                  device=None, render_scale=1., min_extent=320, profile='standard', style_index=None,
                  tone=None, structure=None, skin_structure=None, auto_structure=None,
                  noise_index=0, intensity=1., detail_strength=1., colour_strength=1., detail_radius=4.,
                  temporal=False, motion_mode='sintel', history_strength=1., scene_cut_threshold=.15, inspect_data=False):
    """Load Teacher once, infer one image/scene, and return the saved report dictionary."""
    parameters = {key: str(value) if isinstance(value, Path) else value
                  for key, value in locals().copy().items() if key != 'config'}
    config = load_training_config(config_path) if config is None else config
    root = Path(dataset_root or config['dataset']['root']).expanduser().resolve()
    inventory = inspect_dataset(root, source)
    if inspect_data:
        print(json.dumps(inventory, indent=2), flush=True)
    if input_image is None and scene is None:
        if inspect_data:
            return inventory
        raise ValueError('Set input_image, scene, or inspect_data=True')
    if input_image is not None and scene is not None:
        raise ValueError('Set either input_image or scene, not both')
    for name, value in (('project', project), ('scene', scene)):
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value)):
            raise ValueError(f'{name} must contain only letters, digits, _ and -')
    if source not in ('training', 'test') or render_pass != 'clean':
        raise ValueError('Only clean RGB inputs are supported')
    if type(temporal) is not bool or type(inspect_data) is not bool:
        raise ValueError('temporal and inspect_data must be bool')
    if motion_mode != 'sintel':
        raise ValueError('motion_mode is fixed to sintel')
    for name, value, low, high, integer in (
        ('render_scale', render_scale, .05, 1, False), ('min_extent', min_extent, 320, 4096, True),
        ('max_frames', max_frames, 1, 10, True), ('noise_index', noise_index, 0, 1e9, True),
        ('intensity', intensity, 0, 2, False), ('detail_strength', detail_strength, 0, 2, False),
        ('colour_strength', colour_strength, 0, 2, False), ('detail_radius', detail_radius, .1, 64, False),
        ('history_strength', history_strength, 0, 1, False), ('scene_cut_threshold', scene_cut_threshold, 0, 1, False)):
        validate_range(value, name, low, high, integer=integer)
    controls, automatic = network_controls(profile, style_index=style_index, tone=tone, structure=structure,
        skin_structure=skin_structure, auto_structure=auto_structure)
    if temporal and scene and source != 'training':
        raise ValueError('Sintel motion requires training flow and masks')
    paths = ([Path(input_image).expanduser().resolve()] if input_image else
             sorted((root / source / render_pass / scene).glob('frame_*.png')))
    if max_frames:
        paths = paths[:max_frames]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError('No input images found')
    if temporal and motion_mode == 'sintel':
        for previous, current in zip(paths, paths[1:]):
            if int(current.stem.split('_')[-1]) != int(previous.stem.split('_')[-1]) + 1:
                continue
            for name, stem, suffix in (('flow', previous.stem, '.flo'),
                ('occlusions', previous.stem, '.png'), ('invalid', previous.stem, '.png'),
                ('invalid', current.stem, '.png')):
                guide = root / source / name / scene / (stem + suffix)
                if not guide.is_file():
                    raise FileNotFoundError(f'Missing temporal guide: {guide}')
    device = resolve_device(device or config['runtime']['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(config['runtime']['cpu_threads'])
    teacher = EchoRenderTeacher(config['model']['teacher_weights'], device=device)
    output = reserve_output_dir(Path(config['runtime']['output_dir']).parent / 'infer' / project)
    print(f'Output: {output} | device={device} | temporal={temporal} | motion={motion_mode}', flush=True)
    report = {'device': str(device), 'weights': config['model']['teacher_weights'],
        'reference': REFERENCE_URL, 'dataset': inventory, 'parameters': parameters,
        'controls': controls, 'automatic_mask': vars(automatic) if automatic is not None else None,
        'blend_scale': float(teacher.blend_scale), 'frames': []}
    history = previous_rgb = previous_inner = previous_path = None
    for path in paths:
        rgb = load_rgb(path)
        reset, motion, confidence, guides = None, None, None, {}
        w, h = render_extent(rgb.shape[1], rgb.shape[0], render_scale, min_extent)
        inner = resample(rgb, w, h)
        if history is not None:
            consecutive = int(path.stem.split('_')[-1]) == int(previous_path.stem.split('_')[-1]) + 1
            if rgb.shape != previous_rgb.shape or inner.shape != previous_inner.shape or not consecutive:
                reset = 'image dimensions or frame sequence changed'
            elif scene_cut_threshold and float(np.abs(inner - previous_inner).mean()) > scene_cut_threshold:
                reset = 'scene cut'
            if reset:
                history = None
            else:
                motion, confidence, guides = dataset_motion(root, source, scene,
                    previous_path, path, rgb.shape[:2])
        prediction, next_history, metadata = infer_image(teacher, rgb, controls=controls,
            automatic_mask=automatic, render_scale=render_scale, min_extent=min_extent, noise_index=noise_index,
            history=history, motion=motion, confidence=confidence,
            intensity=intensity, detail_strength=detail_strength, colour_strength=colour_strength,
            detail_radius=detail_radius, history_strength=history_strength)
        destination = output / (path.stem + '.png')
        save_rgb(prediction, destination)
        report['frames'].append({'input': str(path), 'output': str(destination),
            'noise_index': noise_index, 'reset': reset, 'guides': guides, **metadata})
        if temporal and history_strength > 0:
            history, previous_rgb, previous_inner, previous_path = next_history, rgb, inner, path
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'[Teacher]: {path.name} | infer={metadata["seconds"]["inference"]:.3f}s | '
              f'history={metadata["history_used"]}', flush=True)
    return report


if __name__ == '__main__':
    demo()
