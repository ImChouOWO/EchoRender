"""Ordered Sintel/REDS RGB clips; shuffle clips, never frames within a clip."""
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
    kind: str = 'sintel'


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
    kind = dataset.get('type', 'sintel')
    if kind not in ('sintel', 'reds'):
        raise ValueError('dataset.type must be sintel or reds')
    render_pass = dataset.get('render_pass', 'clean')
    expected_pass = 'clean' if kind == 'sintel' else 'sharp'
    if render_pass != expected_pass:
        raise ValueError(f'{kind} dataset.render_pass must be {expected_pass}')
    raw_splits = dataset.get('splits')
    if not isinstance(raw_splits, dict) or set(raw_splits) != {'train', 'val', 'test'}:
        raise ValueError('dataset.splits must define train, val and test')
    splits = {}
    sources = (('train', 'training'), ('val', 'training'), ('test', 'test'))
    if kind == 'reds':
        sources = tuple((name, name) for name in ('train', 'val', 'test'))
    for name, expected_source in sources:
        spec = raw_splits[name]
        if not isinstance(spec, dict) or spec.get('source') != expected_source:
            raise ValueError(f'{name}.source must be {expected_source!r}')
        scenes = _scene_names(spec.get('scenes'), f'{name}.scenes')
        splits[name] = SplitSpec(expected_source, scenes)
    # Train and validation may never silently select the same training images.
    train, val = splits['train'].scenes, splits['val'].scenes
    if kind == 'sintel' and (train is None or val is None):
        raise ValueError('train/val require explicit disjoint scene lists; null is allowed for test')
    same_source = splits['train'].source == splits['val'].source
    overlap = set(train or ()) & set(val or ()) if same_source else set()
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
        tuple(config['rendering']['profiles']), kind,
    )


class RGBSequenceDataset(Dataset):
    def __init__(self, config, split):
        self.root = config.root
        spec = config.splits[split]
        self.kind = config.kind
        self.source, self.render_pass = spec.source, config.render_pass
        self.profiles, self.temporal = config.profiles, config.temporal
        base = self.root / self.source / ('clean' if self.kind == 'sintel' else f'{self.source}_sharp')
        if not base.is_dir():
            raise FileNotFoundError(f'{self.kind} {self.render_pass} images not found: {base}')
        self.scenes = tuple(sorted(spec.scenes or (p.name for p in base.iterdir() if p.is_dir())))
        self.clips = []
        size = config.max_frames if self.temporal else 1
        for scene in self.scenes:
            pattern = 'frame_*.png' if self.kind == 'sintel' else '[0-9]*.png'
            paths = sorted((base / scene).glob(pattern))
            if not paths:
                raise ValueError(f'No {self.render_pass} frames found for {scene}')
            if self.kind == 'reds':
                paired = base.parent / f'{self.source}_sharp_bicubic/X4' / scene
                if {p.name for p in paths} != {p.name for p in paired.glob('*.png')}:
                    raise ValueError(f'REDS sharp/X4 frame names do not match: {scene}')
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
        if self.temporal and self.kind == 'sintel' and self.source == 'training':
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
        prefix = f'{self.source}/{self.render_pass}/{scene}'
        meta = {
            'sample_ids': [f'{prefix}/{path.stem}' for path in paths],
            'clip_id': f'{prefix}/{paths[0].stem}',
            'scene': scene, 'profile': profile, 'paths': list(map(str, paths)),
            'motion_reference': 'sintel_ground_truth' if self.kind == 'sintel' and self.source == 'training' else 'estimated_from_rgb',
        }
        if self.kind == 'reds':
            low = self.root / self.source / f'{self.source}_sharp_bicubic/X4' / scene
            meta['paired_lr_paths'] = [str(low / path.name) for path in paths]
        return meta

    def __getitem__(self, index):
        from .temporal import dataset_motion
        meta = self.metadata(index)
        paths = [Path(p) for p in meta['paths']]
        rgbs = [load_rgb(path) for path in paths]
        if any(rgb.shape != rgbs[0].shape for rgb in rgbs):
            raise ValueError('Frame dimensions must remain constant within a clip')
        motions, confidences = [None], [None]
        for i, (previous, current) in enumerate(zip(paths, paths[1:])) if self.temporal else ():
            if meta['motion_reference'] == 'sintel_ground_truth':
                motion, confidence, _ = dataset_motion(self.root, self.source, meta['scene'], previous, current, rgbs[0].shape[:2])
            else:
                from model.motion import raw_motion
                motion, confidence = raw_motion(rgbs[i], rgbs[i + 1])
            motions.append(motion)
            confidences.append(confidence)
        def tensors(images):
            return [torch.from_numpy(rgb).permute(2, 0, 1).contiguous() for rgb in images]
        result = {**meta, 'images': tensors(rgbs), 'motions': motions, 'confidences': confidences}
        if self.kind == 'reds':
            # Keep Teacher supervision on sharp RGB; expose paired LR without changing the loss target.
            low = [load_rgb(Path(path)) for path in meta['paired_lr_paths']]
            if any((image.shape[0] * 4, image.shape[1] * 4) != rgb.shape[:2] for image, rgb in zip(low, rgbs)):
                raise ValueError('REDS sharp/X4 dimensions must have an exact 4x ratio')
            result['paired_lr_images'] = tensors(low)
        return result


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
    dataset = RGBSequenceDataset(config, split)
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers,
        shuffle=split == 'train' and config.train_shuffle,
        generator=torch.Generator().manual_seed(config.seed), worker_init_fn=_seed_worker,
        collate_fn=collate_sequences, drop_last=False)
