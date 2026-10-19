"""Validated training configuration and the reserved schedule entry point."""
from copy import deepcopy
from pathlib import Path
import math

import yaml

DEFAULT_TRAIN_CONFIG = Path(__file__).resolve().parents[1] / 'config/train.yaml'
LOSS_WEIGHTS = {'distillation', 'reconstruction', 'mid', 'output', 'e1', 'e2', 'e3',
                'vit', 'vit_feature', 'vit_affinity', 'rgb_mse', 'rgb_ssim', 'rgb_ms_ssim'}


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
                'logging', 'schedule', 'dataset', 'loader'}
    if not isinstance(c, dict) or set(c) != required or any(not isinstance(c[k], dict) for k in required):
        raise ValueError(f'train.yaml must contain exactly these mappings: {sorted(required)}')
    expected = {
        'model': {'student_config', 'student_weights', 'teacher_weights'},
        'runtime': {'device', 'cpu_threads', 'output_dir'},
        'training': {'epochs', 'max_steps', 'crop_size', 'optimizer', 'grad_clip'},
        'loss': LOSS_WEIGHTS | {'projection', 'norm_eps', 'affinity_chunk_size', 'ssim_window', 'ms_ssim_weights'},
        'validation': {'reference', 'lpips_net'}, 'benchmark': {'enabled', 'warmup', 'repeats'},
        'logging': {'progress', 'gpu_poll_seconds'}, 'schedule': {'kind'},
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
        c[section][key] = str((path.parent / value).resolve())
    for section, key, zero in (('runtime', 'cpu_threads', False), ('training', 'epochs', False),
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
    for section, key in (('loss', 'projection'), ('logging', 'progress'), ('benchmark', 'enabled')):
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
    if c['schedule']['kind'] != 'constant':
        raise ValueError('Only constant schedule is implemented; extend TrainingSchedule for future strategies')
    c['config_path'] = str(path)
    return c


class TrainingSchedule:
    """Reserved call site for future epoch/step-dependent training policies.

    Called before each optimizer update. Return absolute current values, never
    mutate the base config. Add future strategy definitions here and validate
    their YAML in load_training_config; the default intentionally stays constant.
    """
    def __init__(self, config):
        self.config = deepcopy(config)

    def values(self, *, epoch, step):
        return {
            'lr': self.config['training']['optimizer']['lr'],
            'grad_clip': self.config['training']['grad_clip'],
            'loss_weights': {key: self.config['loss'][key] for key in LOSS_WEIGHTS},
        }
