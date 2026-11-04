"""REDS pairing, temporal supervision and Teacher-cache integration."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from .dataloader import build_dataloader, load_data_config
from .distillation import (train_batch, evaluate, StudentAugmentation, cache_key,
                           make_frame, supervised_motion)
from .teacher_features import TeacherFeatureCache
from .temporal import teacher_target, image_rgb
from .training_config import load_training_config, selected_data_config, TrainingSchedule
from .test_training import tiny_model, FixtureTeacher
from .test_temporal import Metrics, Monitor
from model.card.loss import DistillationLoss


class RedsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(load_training_config('config/train_reds.yaml'))
        self.config['dataset']['root'] = str(self.root)
        for split in ('train', 'val'):
            self.config['dataset']['splits'][split]['scenes'] = None
        self.config['training'].update(adapter_size=32, precision='float32', micro_batch_size=2)
        self.config['training']['distillation_layers'] = ['RGB composition', 'OutputHead', 'E2', 'VIT', 'D1', 'D3']
        self.config['rendering'].update(max_frames=2, train_min_frames=2)
        self.config['loader'].update(num_workers=0, batch_size=2, train_shuffle=False)
        self.config['loss'].update(rgb_lpips=0., ssim_window=3, ms_ssim_weights=[1.])
        self.config['logging']['progress'] = self.config['benchmark']['enabled'] = False
        generator = np.random.default_rng(8)
        for split in ('train', 'val'):
            for sequence in ('000', '001'):
                high = self.root / split / f'{split}_sharp' / sequence
                low = self.root / split / f'{split}_sharp_bicubic/X4' / sequence
                high.mkdir(parents=True)
                low.mkdir(parents=True)
                pixels = generator.integers(32, 224, (64, 96, 3), dtype=np.uint8)
                for frame in range(3):
                    image = Image.fromarray(np.roll(pixels, frame, axis=1))
                    image.save(high / f'{frame:08}.png')
                    image.resize((24, 16), Image.Resampling.BICUBIC).save(low / f'{frame:08}.png')

    def loader(self, split='train'):
        return build_dataloader(selected_data_config(self.config), split=split)

    def test_native_pairs_split_isolation_and_clip_order(self):
        train, val = self.loader(), self.loader('val')
        self.assertEqual(len(train.dataset), 4)
        self.assertEqual([len(paths) for _, paths in train.dataset.clips], [2, 1, 2, 1])
        a, b = train.dataset[0], val.dataset[0]
        self.assertEqual(a['sample_ids'][0], 'train/sharp/000/00000000')
        self.assertEqual(b['sample_ids'][0], 'val/sharp/000/00000000')
        self.assertEqual(a['images'][0].shape, (3, 64, 96))
        self.assertEqual(a['paired_lr_images'][0].shape, (3, 16, 24))
        self.assertEqual(a['motion_reference'], 'estimated_from_rgb')
        self.assertEqual(a['motions'][1].shape, (64, 96, 2))
        with self.assertRaises(FileNotFoundError):
            self.loader('test')
        self.config['dataset']['splits']['train']['scenes'] = ['001']
        self.assertEqual(self.loader().dataset.scenes, ('001',))

    def test_missing_pair_and_wrong_scale_fail_before_training(self):
        low = self.root / 'train/train_sharp_bicubic/X4/000/00000000.png'
        low.unlink()
        with self.assertRaisesRegex(ValueError, 'frame names'):
            self.loader()
        Image.new('RGB', (25, 16)).save(low)
        with self.assertRaisesRegex(ValueError, '4x ratio'):
            self.loader().dataset[0]

    def test_single_frames_do_not_estimate_motion(self):
        self.config['rendering']['temporal'] = False
        with patch('model.motion.raw_motion', side_effect=AssertionError('unexpected flow')):
            loader = self.loader()
            self.assertEqual(len(loader.dataset), 6)
            self.assertEqual(loader.dataset[0]['motions'], [None])

    def test_profiles_remain_independent_and_sintel_validation_is_unchanged(self):
        self.config['training']['profiles'] = ['neutral', 'standard']
        loader = self.loader()
        self.assertEqual(len(loader.dataset), 8)
        self.assertEqual(loader.dataset[1]['profile'], 'standard')
        self.config['dataset']['render_pass'] = 'final'
        with self.assertRaises(ValueError):
            load_data_config(self.config)
        self.assertEqual(load_data_config(load_training_config()).kind, 'sintel')

    def test_estimated_flow_is_not_a_ground_truth_label(self):
        sample = self.loader().dataset[0]
        motion, confidence = sample['motions'][1], sample['confidences'][1]
        self.assertEqual(supervised_motion(sample, motion, confidence), (None, None))
        sample['motion_reference'] = 'sintel_ground_truth'
        self.assertIs(supervised_motion(sample, motion, confidence)[0], motion)

    def test_reds_cache_is_outside_the_sintel_cache_cleanup_tree(self):
        from .trainer import feature_cache_dir
        with patch('tool.trainer.FEATURE_CACHE_DIR', self.root / 'feats'):
            self.assertEqual(feature_cache_dir(self.config), self.root / 'feats_reds')
            self.assertEqual(feature_cache_dir(load_training_config()), self.root / 'feats')

    def test_cached_teacher_targets_train_and_evaluate_on_reds(self):
        config = self.config
        loader = self.loader()
        batch = {'sequences': [loader.dataset[0], loader.dataset[2]]}
        student, teacher = tiny_model('raft_small'), FixtureTeacher()
        criterion = DistillationLoss(config['loss'], channels=dict.fromkeys(('E2', 'VIT', 'D1', 'D3'), 4))
        cache = TeacherFeatureCache(self.root / 'cache')
        cache.open()
        self.addCleanup(cache.close)
        cache.configure_signatures({'rendering': config['rendering']})
        cache.prepare('fixture', .5)
        for sample in batch['sequences']:
            history = None
            for position in range(len(sample['images'])):
                target = teacher_target(teacher, make_frame(sample, position, config, history))
                history = target['history']
                target['history'] = torch.from_numpy(history.copy())
                for split in ('train', 'val'):
                    cache.store(split, cache_key(sample, position), target)
        cache.seal('fixture')
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
        controls = TrainingSchedule(config, total_steps=2).values(epoch=1, step=0)
        value = train_batch(None, student, criterion, batch, optimizer, config=config, controls=controls,
                            target_cache=cache, blend_scale=.5, micro_batch_size=2)
        self.assertEqual(value['frames'], 4)
        self.assertEqual(value['components']['motion'], 0.)
        result = evaluate(None, student, criterion, [batch], Metrics(), Monitor(), config=config,
                          epoch=1, controls=controls, target_cache=cache, blend_scale=.5)
        self.assertEqual(result['images'], 4)
        self.assertEqual(result['reference_motion'], ['estimated_from_rgb'])
        self.assertTrue(all(p.grad is not None for p in student.motion.parameters()))
        augmentation = StudentAugmentation(config)
        changed = augmentation.apply_rgb(image_rgb(batch['sequences'][0]['images'][0]), 'brightness', .25)
        self.assertTrue(np.isfinite(changed).all())


if __name__ == '__main__':
    unittest.main()
