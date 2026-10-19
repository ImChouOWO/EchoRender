"""Single-image Sintel loading with explicit scene splits and CPU-only batches.

The loader reads RGB images. Teacher feature preparation, models, loss and GPU
work belong to the caller. Ragged batches keep images at their original sizes.
"""
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .teacher_pipeline import load_rgb


DEFAULT_DATA_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'data.yaml'


@dataclass(frozen=True)
class SplitSpec:
    source: str
    scenes: tuple[str, ...] | None


@dataclass(frozen=True)
class DataConfig:
    root: Path
    render_pass: str
    splits: dict[str, SplitSpec]
    batch_size: int
    num_workers: int
    seed: int
    train_shuffle: bool


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _scene_names(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f'{name} must be a nonempty scene list or null for all scenes')
    for scene in value:
        if (not isinstance(scene, str) or not scene or scene in ('.', '..')
                or scene.startswith('.') or any(char in scene for char in '/\\:')):
            raise ValueError(f'{name} contains an invalid scene name: {scene!r}')
    if len(set(value)) != len(value):
        raise ValueError(f'{name} contains duplicate scenes')
    return tuple(value)


def load_data_config(path=DEFAULT_DATA_CONFIG):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8-sig') as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError('Data config must be a mapping')
    dataset, loader = config.get('dataset'), config.get('loader')
    if not isinstance(dataset, dict) or not isinstance(loader, dict):
        raise ValueError('Data config needs dataset and loader mappings')
    root = dataset.get('root')
    if not isinstance(root, str) or not root.strip():
        raise ValueError('dataset.root must be a path string')
    root = Path(root).expanduser()
    root = (root if root.is_absolute() else path.parent / root).resolve()
    render_pass = dataset.get('render_pass', 'clean')
    if render_pass not in ('clean', 'final'):
        raise ValueError('dataset.render_pass must be clean or final')
    raw_splits = dataset.get('splits')
    if not isinstance(raw_splits, dict) or set(raw_splits) != {'train', 'val', 'test'}:
        raise ValueError('dataset.splits must define train, val and test')
    splits = {}
    for name, expected_source in (('train', 'training'), ('val', 'training'), ('test', 'test')):
        spec = raw_splits[name]
        if not isinstance(spec, dict) or spec.get('source') != expected_source:
            raise ValueError(f'{name}.source must be {expected_source!r}')
        scenes = _scene_names(spec.get('scenes'), f'{name}.scenes')
        splits[name] = SplitSpec(expected_source, scenes)
    # Train and validation may never silently select the same training images.
    train, val = splits['train'].scenes, splits['val'].scenes
    if train is None or val is None:
        raise ValueError('train/val require explicit disjoint scene lists; null is allowed for test')
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f'train/val scenes overlap: {sorted(overlap)}')
    shuffle = loader.get('train_shuffle', True)
    if type(shuffle) is not bool:
        raise ValueError('loader.train_shuffle must be true or false')
    return DataConfig(
        root, render_pass, splits,
        _integer(loader.get('batch_size', 1), 'loader.batch_size', 1),
        _integer(loader.get('num_workers', 0), 'loader.num_workers'),
        _integer(loader.get('seed', 42), 'loader.seed'), shuffle,
    )


class SintelImageDataset(Dataset):
    def __init__(self, root, *, source='training', render_pass='clean', scenes=None):
        if source not in ('training', 'test') or render_pass not in ('clean', 'final'):
            raise ValueError('Sintel source must be training/test and render_pass must be clean/final')
        self.root = Path(root).expanduser().resolve()
        self.source, self.render_pass = source, render_pass
        base = self.root / source / render_pass
        if not base.is_dir():
            raise FileNotFoundError(f'Sintel image directory not found: {base}')
        selected = _scene_names(scenes, 'scenes')
        self.scenes = tuple(sorted(selected if selected is not None else (
            path.name for path in base.iterdir() if path.is_dir() and not path.name.startswith('.')
        )))
        self.samples = []
        for scene in self.scenes:
            directory = base / scene
            if not directory.is_dir():
                raise FileNotFoundError(f'Sintel scene not found: {directory}')
            paths = sorted(path for path in directory.glob('frame_*.png') if path.is_file())
            if not paths:
                raise ValueError(f'No frame_*.png images found in {directory}')
            self.samples.extend((scene, path) for path in paths)
        if not self.samples:
            raise ValueError(f'No Sintel image scenes found in {base}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene, path = self.samples[index]
        rgb = load_rgb(path)
        return {
            'image': torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
            'sample_id': f'{self.source}/{self.render_pass}/{scene}/{path.stem}',
            'scene': scene, 'path': str(path), 'original_size': tuple(rgb.shape[:2]),
        }


def collate_images(samples):
    """Keep image sizes intact; the caller can accumulate gradients per image."""
    return {
        'images': [sample['image'] for sample in samples],
        'sample_ids': [sample['sample_id'] for sample in samples],
        'scenes': [sample['scene'] for sample in samples],
        'paths': [sample['path'] for sample in samples],
        'original_sizes': [sample['original_size'] for sample in samples],
    }


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(config=DEFAULT_DATA_CONFIG, *, split='train'):
    config = config if isinstance(config, DataConfig) else load_data_config(config)
    if split not in config.splits:
        raise ValueError('split must be train, val or test')
    spec = config.splits[split]
    dataset = SintelImageDataset(config.root, source=spec.source, render_pass=config.render_pass, scenes=spec.scenes)
    return DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.num_workers,
        shuffle=split == 'train' and config.train_shuffle,
        generator=torch.Generator().manual_seed(config.seed),
        worker_init_fn=_seed_worker, collate_fn=collate_images, drop_last=False,
    )
