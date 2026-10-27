"""Scene-disjoint clean RGB clips; shuffle clips, never frames within a clip."""
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .teacher_pipeline import load_rgb


DEFAULT_DATA_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'train.yaml'


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
    max_frames: int
    temporal: bool
    profiles: tuple[str, ...]


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
    if isinstance(path, dict):
        config = path
        path = Path(config['config_path'])
    else:
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
    if render_pass != 'clean':
        raise ValueError('dataset.render_pass is fixed to clean')
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
        config['rendering']['max_frames'], config['rendering']['temporal'],
        tuple(config['rendering']['profiles']),
    )


class SintelSequenceDataset(Dataset):
    def __init__(self, config, split):
        self.root = config.root
        spec = config.splits[split]
        self.source, self.render_pass = spec.source, 'clean'
        self.profiles, self.temporal = config.profiles, config.temporal
        base = self.root / self.source / 'clean'
        if not base.is_dir():
            raise FileNotFoundError(f'Sintel clean images not found: {base}')
        self.scenes = tuple(sorted(spec.scenes or (p.name for p in base.iterdir() if p.is_dir())))
        self.clips = []
        size = config.max_frames if self.temporal else 1
        for scene in self.scenes:
            paths = sorted((base / scene).glob('frame_*.png'))
            if not paths:
                raise ValueError(f'No clean frames found for {scene}')
            group = []
            for path in paths:
                number = int(path.stem.split('_')[-1])
                if group and (len(group) == size or number != int(group[-1].stem.split('_')[-1]) + 1):
                    self.clips.append((scene, tuple(group)))
                    group = []
                group.append(path)
            if group:
                self.clips.append((scene, tuple(group)))
        self.items = [(index, profile) for index in range(len(self.clips)) for profile in self.profiles]
        # Fail before model loading when multi-frame ground truth is unavailable.
        if self.temporal:
            for scene, paths in self.clips:
                for previous, current in zip(paths, paths[1:]):
                    for name, frame, suffix in (('flow', previous, '.flo'), ('occlusions', previous, '.png'),
                                               ('invalid', previous, '.png'), ('invalid', current, '.png')):
                        guide = self.root / self.source / name / scene / (frame.stem + suffix)
                        if not guide.is_file():
                            raise FileNotFoundError(f'Missing Sintel temporal guide: {guide}')

    def __len__(self):
        return len(self.items)

    def metadata(self, index):
        clip_index, profile = self.items[index]
        scene, paths = self.clips[clip_index]
        return {
            'sample_ids': [f'{self.source}/clean/{scene}/{path.stem}' for path in paths],
            'clip_id': f'{self.source}/clean/{scene}/{paths[0].stem}',
            'scene': scene, 'profile': profile, 'paths': list(map(str, paths)),
        }

    def __getitem__(self, index):
        from .temporal import dataset_motion
        meta = self.metadata(index)
        paths = [Path(p) for p in meta['paths']]
        rgbs = [load_rgb(path) for path in paths]
        if any(rgb.shape != rgbs[0].shape for rgb in rgbs):
            raise ValueError('Frame dimensions must remain constant within a clip')
        motions, confidences = [None], [None]
        for previous, current in zip(paths, paths[1:]):
            motion, confidence, _ = dataset_motion(self.root, self.source, meta['scene'], previous, current, rgbs[0].shape[:2])
            motions.append(motion)
            confidences.append(confidence)
        return {**meta, 'images': [torch.from_numpy(rgb).permute(2, 0, 1).contiguous() for rgb in rgbs],
                'motions': motions, 'confidences': confidences}


def collate_sequences(samples):
    return {'sequences': samples}


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(config=DEFAULT_DATA_CONFIG, *, split='train'):
    config = config if isinstance(config, DataConfig) else load_data_config(config)
    if split not in config.splits:
        raise ValueError('split must be train, val or test')
    dataset = SintelSequenceDataset(config, split)
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers,
        shuffle=split == 'train' and config.train_shuffle,
        generator=torch.Generator().manual_seed(config.seed), worker_init_fn=_seed_worker,
        collate_fn=collate_sequences, drop_last=False)
