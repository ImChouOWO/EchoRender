"""Validated training configuration and optimizer-update scheduling."""
from copy import deepcopy
from pathlib import Path
import math

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / 'config/train.yaml'
LOSS_WEIGHTS = {'distillation', 'reconstruction', 'mid', 'output', 'e1', 'e2', 'e3',
                'vit', 'vit_feature', 'vit_affinity', 'rgb_mse', 'rgb_ssim', 'rgb_ms_ssim'}


def reserve_output_dir(path):
    """Atomically reserve an unused run directory; suffix numbering starts at 2."""
    base = Path(path)
    number = 1
    while True:
        candidate = base if number == 1 else base.with_name(f'{base.name}_{number}')
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            number += 1


def _positive(value, name, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if integer and type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    if value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be {"nonnegative" if zero else "positive"}')


def load_training_config(path=DEFAULT_TRAIN_CONFIG):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8-sig') as source:
        c = yaml.safe_load(source)
    required = {'model', 'runtime', 'training', 'loss', 'validation', 'benchmark',
                'logging', 'schedule', 'augmentation', 'dataset', 'loader'}
    if not isinstance(c, dict) or set(c) != required or any(not isinstance(c[k], dict) for k in required):
        raise ValueError(f'train.yaml must contain exactly these mappings: {sorted(required)}')
    expected = {
        'model': {'student_config', 'student_weights', 'teacher_weights'},
        'runtime': {'device', 'cpu_threads', 'output_dir'},
        'training': {'epochs', 'releas', 'max_steps', 'crop_size', 'optimizer', 'grad_clip', 'micro_batch_size', 'gradient_checkpointing'},
        'loss': LOSS_WEIGHTS | {'projection', 'norm_eps', 'affinity_chunk_size', 'ssim_window', 'ms_ssim_weights'},
        'validation': {'reference', 'lpips_net'}, 'benchmark': {'enabled', 'warmup', 'repeats'},
        'logging': {'progress', 'gpu_poll_seconds'}, 'schedule': {'kind', 'eta_min'},
        'augmentation': {'enabled', 'start_probability', 'max_probability', 'peak_fraction',
                         'stop_fraction', 'noise_std', 'brightness', 'contrast', 'sharpness'},
        'dataset': {'root', 'render_pass', 'splits'},
        'loader': {'batch_size', 'num_workers', 'seed', 'train_shuffle'},
    }
    for section, keys in expected.items():
        if set(c[section]) != keys:
            raise ValueError(f'{section} must contain exactly: {sorted(keys)}')
    if not isinstance(c['training']['optimizer'], dict) or set(c['training']['optimizer']) != {'lr', 'momentum', 'weight_decay'}:
        raise ValueError('optimizer requires lr, momentum, weight_decay')
    for section, key in (('model', 'student_config'), ('model', 'student_weights'),
                         ('model', 'teacher_weights'), ('runtime', 'output_dir'), ('dataset', 'root')):
        value = c[section][key]
        if value is None and key == 'student_weights':
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{section}.{key} must be a nonempty path')
        value = Path(value).expanduser()
        base = PROJECT_ROOT if key == 'teacher_weights' else path.parent
        resolved = (base / value).resolve()
        if key == 'teacher_weights' and not resolved.is_relative_to((PROJECT_ROOT / 'weights').resolve()):
            raise ValueError('model.teacher_weights must be inside the project weights directory')
        c[section][key] = str(resolved)
    for section, key, zero in (('runtime', 'cpu_threads', False), ('training', 'epochs', False),
                               ('training', 'micro_batch_size', False),
                               ('training', 'crop_size', True), ('benchmark', 'warmup', True),
                               ('benchmark', 'repeats', False), ('loss', 'affinity_chunk_size', False),
                               ('loss', 'ssim_window', False)):
        _positive(c[section][key], f'{section}.{key}', integer=True, zero=zero)
    if c['training']['max_steps'] is not None:
        _positive(c['training']['max_steps'], 'max_steps', integer=True)
    _positive(c['training']['grad_clip'], 'grad_clip', zero=True)
    for key, value in c['training']['optimizer'].items():
        _positive(value, f'optimizer.{key}', zero=key != 'lr')
    for key in LOSS_WEIGHTS:
        _positive(c['loss'][key], f'loss.{key}', zero=True)
    _positive(c['loss']['norm_eps'], 'norm_eps')
    _positive(c['logging']['gpu_poll_seconds'], 'gpu_poll_seconds')
    for section, key in (('training', 'gradient_checkpointing'), ('training', 'releas'), ('loss', 'projection'), ('logging', 'progress'), ('benchmark', 'enabled')):
        if type(c[section][key]) is not bool:
            raise ValueError(f'{section}.{key} must be boolean')
    weights = c['loss']['ms_ssim_weights']
    if not isinstance(weights, list) or not 1 <= len(weights) <= 5:
        raise ValueError('ms_ssim_weights must have 1 to 5 entries')
    for weight in weights:
        _positive(weight, 'ms_ssim_weights entry')
    window = c['loss']['ssim_window']
    if window < 3 or window % 2 == 0:
        raise ValueError('ssim_window must be odd and >= 3')
    minimum = (window - 1) * 2 ** (len(weights) - 1)
    if c['training']['crop_size'] and c['training']['crop_size'] <= minimum:
        raise ValueError(f'crop_size must exceed {minimum} for configured MS-SSIM')
    if c['validation']['reference'] != 'teacher':
        raise ValueError('Only the agreed Teacher RGB reference is implemented')
    if c['validation']['lpips_net'] not in ('alex', 'vgg', 'squeeze'):
        raise ValueError('lpips_net must be alex, vgg or squeeze')
    if c['schedule']['kind'] not in ('constant', 'cosine'):
        raise ValueError('schedule.kind must be constant or cosine')
    _positive(c['schedule']['eta_min'], 'schedule.eta_min', zero=True)
    if c['schedule']['eta_min'] > c['training']['optimizer']['lr']:
        raise ValueError('schedule.eta_min cannot exceed the initial optimizer.lr')
    augmentation = c['augmentation']
    if type(augmentation['enabled']) is not bool:
        raise ValueError('augmentation.enabled must be boolean')
    for key in set(augmentation) - {'enabled'}:
        _positive(augmentation[key], f'augmentation.{key}', zero=True)
    if not 0 <= augmentation['start_probability'] <= augmentation['max_probability'] <= 0.25:
        raise ValueError('augmentation probabilities must satisfy 0 <= start <= max <= 0.25')
    if not 0 < augmentation['peak_fraction'] < augmentation['stop_fraction'] <= 1:
        raise ValueError('augmentation fractions must satisfy 0 < peak < stop <= 1')
    if any(augmentation[key] > 1 for key in ('noise_std', 'brightness', 'contrast', 'sharpness')):
        raise ValueError('augmentation strengths must be within [0,1]')
    c['config_path'] = str(path)
    return c


class TrainingSchedule:
    """Stateless per-update LR schedule; other controls remain fixed.

    Called before each optimizer update. Return absolute current values, never
    mutate the base config. For T >= 2 updates, cosine starts at base LR on step
    zero and reaches eta_min on step T-1. One update uses the base LR. Values
    beyond the horizon stay at eta_min; no warmup/restarts. This interface also
    reserves future epoch/step-dependent loss-weight and training policies.
    """
    def __init__(self, config, *, total_steps=None):
        self.config = deepcopy(config)
        self.total_steps = total_steps
        if config['schedule']['kind'] == 'cosine':
            if type(total_steps) is not int or total_steps < 1:
                raise ValueError('Cosine scheduling requires positive total_steps')

    def values(self, *, epoch, step):
        if type(step) is not int or step < 0:
            raise ValueError('step must be a nonnegative integer')
        lr = self.config['training']['optimizer']['lr']
        if self.config['schedule']['kind'] == 'cosine':
            minimum = self.config['schedule']['eta_min']
            progress = min(step / max(self.total_steps - 1, 1), 1.0)
            lr = minimum + 0.5 * (lr - minimum) * (1 + math.cos(math.pi * progress))
        return {
            'lr': lr,
            'grad_clip': self.config['training']['grad_clip'],
            'loss_weights': {key: self.config['loss'][key] for key in LOSS_WEIGHTS},
        }
