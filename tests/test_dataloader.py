from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import RandomSampler, SequentialSampler
import yaml

from tool.dataloader import SintelImageDataset, build_dataloader, load_data_config


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / 'sintel'
        for source, scene in (('training', 'train_scene'), ('training', 'val_scene'), ('test', 'test_scene')):
            for render_pass, color in (('clean', (255, 0, 0)), ('final', (0, 255, 0))):
                directory = self.root / source / render_pass / scene
                directory.mkdir(parents=True)
                for frame in range(1, 4):
                    pixels = np.empty((frame + 3, frame + 5, 3), dtype=np.uint8)
                    pixels[:] = color
                    Image.fromarray(pixels).save(directory / f'frame_{frame:04}.png')
        (self.root / 'test' / 'clean' / '.AppleDouble').mkdir()
        self.config_path = self.directory / 'config' / 'data.yaml'
        self.config_path.parent.mkdir()
        self.config = {
            'dataset': {'root': '../sintel', 'render_pass': 'clean', 'splits': {
                'train': {'source': 'training', 'scenes': ['train_scene']},
                'val': {'source': 'training', 'scenes': ['val_scene']},
                'test': {'source': 'test', 'scenes': None},
            }},
            'loader': {'batch_size': 2, 'num_workers': 0, 'seed': 42, 'train_shuffle': True},
        }
        self.write_config()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')

    def test_relative_paths_and_cpu_rgb_samples(self):
        config = load_data_config(self.config_path)
        self.assertEqual(config.root, self.root.resolve())
        sample = build_dataloader(config).dataset[0]
        self.assertEqual(sample['sample_id'], 'training/clean/train_scene/frame_0001')
        self.assertEqual(sample['original_size'], (4, 6))
        self.assertEqual(sample['image'].shape, (3, 4, 6))
        self.assertEqual(sample['image'].dtype, torch.float32)
        self.assertEqual(sample['image'].device.type, 'cpu')
        self.assertFalse(sample['image'].requires_grad)
        self.assertTrue(torch.all(sample['image'][0] == 1))
        self.assertTrue(torch.all(sample['image'][1:] == 0))

    def test_pass_is_independent_of_scene_split(self):
        self.config['dataset']['render_pass'] = 'final'
        self.write_config()
        dataset = build_dataloader(self.config_path, split='val').dataset
        sample = dataset[0]
        self.assertEqual(sample['sample_id'], 'training/final/val_scene/frame_0001')
        self.assertTrue(torch.all(sample['image'][1] == 1))
        self.assertTrue(torch.all(sample['image'][0] == 0))

    def test_distinct_splits_and_evaluation_order(self):
        train = build_dataloader(self.config_path, split='train')
        val = build_dataloader(self.config_path, split='val')
        test = build_dataloader(self.config_path, split='test')
        self.assertIsInstance(train.sampler, RandomSampler)
        self.assertIsInstance(val.sampler, SequentialSampler)
        self.assertIsInstance(test.sampler, SequentialSampler)
        self.assertEqual(test.dataset.source, 'test')
        self.assertEqual(test.dataset.scenes, ('test_scene',))
        self.assertEqual(len(test.dataset), 3)
        paths = [{str(path) for _, path in loader.dataset.samples} for loader in (train, val, test)]
        self.assertFalse(paths[0] & paths[1])
        self.assertFalse(paths[0] & paths[2])
        self.assertEqual(next(iter(val))['sample_ids'][0], 'training/clean/val_scene/frame_0001')

    def test_ragged_batch_preserves_shapes_and_keeps_last_sample(self):
        loader = build_dataloader(self.config_path, split='val')
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual([tuple(image.shape) for image in batches[0]['images']], [(3, 4, 6), (3, 5, 7)])
        self.assertEqual(batches[0]['original_sizes'], [(4, 6), (5, 7)])
        self.assertEqual(len(batches[1]['images']), 1)

    def test_training_shuffle_is_reproducible(self):
        first = list(build_dataloader(self.config_path))
        second = list(build_dataloader(self.config_path))
        self.assertEqual([batch['sample_ids'] for batch in first], [batch['sample_ids'] for batch in second])

    def test_overlap_and_using_test_for_training_are_rejected(self):
        self.config['dataset']['splits']['val']['scenes'] = ['train_scene']
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            load_data_config(self.config_path)
        self.config['dataset']['splits']['val']['scenes'] = ['val_scene']
        self.config['dataset']['splits']['train']['source'] = 'test'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'source'):
            load_data_config(self.config_path)

    def test_missing_empty_and_invalid_scene_selection(self):
        with self.assertRaises(FileNotFoundError):
            SintelImageDataset(self.root, scenes=['missing'])
        empty = self.root / 'training' / 'clean' / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'No frame'):
            SintelImageDataset(self.root, scenes=['empty'])
        with self.assertRaises(ValueError):
            SintelImageDataset(self.root, scenes=['../train_scene'])
        self.config['loader']['batch_size'] = 0
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            load_data_config(self.config_path)


if __name__ == '__main__':
    unittest.main()
