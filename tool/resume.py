"""Restore epoch-boundary training state without restarting the LR/augmentation schedules."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .reporting import validation_values, is_better
from .temporal import FORMAT


def checkpoint_values(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: checkpoint_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(checkpoint_values(item) for item in value)
    return value


def read_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError(f'Expected a {FORMAT} checkpoint')
    for key in ('student', 'distillation', 'optimizer', 'student_config', 'train_config', 'epoch', 'global_step'):
        if key not in payload:
            raise ValueError(f'Incomplete training checkpoint: missing {key}')
    return payload


def validate_resume(payload, config, student):
    if payload['student_config'] != student.config:
        raise ValueError('Resume requires the checkpoint Student architecture and motion mode')
    saved = payload['train_config']
    for section in ('dataset', 'loader', 'loss', 'augmentation', 'schedule', 'validation'):
        if saved[section] != config[section]:
            raise ValueError(f'Resume requires the original {section} settings')
    profiles = saved['training'].get('profiles', list(saved['rendering']['profiles']))
    rendering = {**config['rendering'], 'profiles': {name: config['rendering']['profiles'][name]
                                                  for name in config['training']['profiles']}}
    if profiles != config['training']['profiles'] or saved['rendering'] != rendering:
        raise ValueError('Resume requires the original profiles and rendering settings')
    for key in ('epochs', 'max_steps', 'crop_size', 'optimizer', 'grad_clip', 'micro_batch_size'):
        if saved['training'][key] != config['training'][key]:
            raise ValueError(f'Resume requires the original training.{key}')
    if saved['model']['teacher_weights'] != config['model']['teacher_weights']:
        raise ValueError('Resume requires the original teacher_weights')
    if saved['reporting']['best_metric'] != config['reporting']['best_metric']:
        raise ValueError('Resume requires the original best_metric')
    if payload.get('resume_state') and saved['training']['early_stopping'] != config['training']['early_stopping']:
        raise ValueError('Resume requires the original early_stopping settings')
    if payload['epoch'] >= config['training']['epochs']:
        raise ValueError('The checkpoint has already reached the configured total epochs')
    maximum = config['training']['max_steps']
    if maximum is not None and payload['global_step'] >= maximum:
        raise ValueError('The checkpoint has already reached max_steps')


def capture_rng(augmentation, loaders, device):
    generator, keys, position, gaussian, cached = np.random.get_state()
    state = {'python': random.getstate(), 'numpy': (generator, torch.from_numpy(keys.astype(np.int64)),
             position, gaussian, cached), 'torch': torch.get_rng_state(),
             'augmentation': augmentation.generator.get_state(),
             'loaders': [loader.generator.get_state() if getattr(loader, 'generator', None) is not None else None
                         for loader in loaders]}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, augmentation, loaders, device):
    random.setstate(state['python'])
    generator, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((generator, keys.numpy().astype(np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'])
    augmentation.generator.set_state(state['augmentation'])
    for loader, value in zip(loaders, state['loaders']):
        if value is not None:
            loader.generator.set_state(value)
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)


def load_log(output, epoch):
    path = output / 'metrics.jsonl'
    def numbers(value):
        if isinstance(value, dict):
            return {key: numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [numbers(item) for item in value]
        return float(value) if isinstance(value, str) and value in ('Infinity', '-Infinity') else value
    lines = path.read_text().splitlines() if path.exists() else []
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(numbers(json.loads(line)))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError('Training log is corrupt before its final line') from None
    if any(row.get('event') == 'epoch' and row['epoch'] > epoch for row in rows):
        raise ValueError('Checkpoint predates completed epochs in this run; resume its latest last.pt')
    # Discard an interrupted epoch's partial batches; they are replayed after resume.
    return [row for row in rows if row.get('epoch', 0) <= epoch]


def restore_progress(payload, output, early, metric):
    rows = load_log(output, payload['epoch'])
    state = payload.get('resume_state') or {}
    history = state.get('history', [row for row in rows if row.get('event') == 'epoch'])
    if not history or history[-1]['epoch'] != payload['epoch']:
        raise ValueError('Epoch history is missing; keep metrics.jsonl alongside this checkpoint')
    best, best_epoch = None, None
    for row in history:
        values = validation_values(row['validation'])
        if is_better(values, best, metric):
            best, best_epoch = values, row['epoch']
        early.update(values[metric], row['epoch'])
    if state.get('early'):
        for key, value in state['early'].items():
            setattr(early, key, value)
    if not (output / 'best.pt').is_file():
        raise ValueError('best.pt is missing from the resumed run')
    return rows, history, best, best_epoch, sum(row['train_images'] for row in history)


@contextmanager
def training_lock(output):
    """Prevent concurrent writers, including an already-running pre-lock trainer."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Training output is already in use: {output}') from None
        metrics = str((output / 'metrics.jsonl').resolve())
        for process in Path('/proc').glob('[0-9]*'):
            if process.name == str(os.getpid()):
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                if any(os.readlink(fd) == metrics for fd in (process / 'fd').iterdir()):
                    raise RuntimeError(f'Training output is already in use by PID {process.name}: {output}')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        yield
