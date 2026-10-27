"""Run with python -m unittest tool.test_training tool.test_reporting -v."""
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image
import torch
from torch import nn

from .reporting import (validation_values, is_better, best_message, metric_statistics,
                        write_training_report)
from .prediction_report import (augment_image, export_predictions, prepare_variant,
                                resize_rgb, sample_modes, VariantInput, MODES)
from .test_training import tiny_model, FixtureTeacher, identity_prepare
from model.card.loss import DistillationLoss
from .training_config import load_training_config

from .trainer import fit


def validation(loss, psnr, ssim, lpips):
    return {'components': {'total': loss}, 'image_mean': {'psnr': psnr, 'ssim': ssim, 'lpips': lpips},
            'benchmark': None}


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_best_metric_directions_and_improvements(self):
        old = {'loss': 4., 'psnr': 20., 'ssim': .5, 'lpips': .4}
        new = {'loss': 3., 'psnr': 22., 'ssim': .6, 'lpips': .3}
        for metric in old:
            self.assertTrue(is_better(new, old, metric))
            self.assertFalse(is_better(old, new, metric))
            self.assertFalse(is_better(old, old, metric))
        message = best_message(2, new, old)
        self.assertTrue(message.startswith('[Best]: save best weight |'))
        self.assertIn('loss=+25.00%', message)
        self.assertIn('PSNR=+10.00%', message)
        self.assertIn('LPIPS=+25.00%', message)
        self.assertIn('initial', best_message(1, old, None))
        self.assertIn('N/A', best_message(3, {**new, 'psnr': float('inf')}, old))



    def test_reporting_config_rejects_invalid_values(self):
        import yaml
        original = load_training_config()
        for key, value in (('enabled', 'true'), ('project_name', '../escape'), ('project_name', ''),
                           ('best_metric', 'unknown'), ('crop_size', -1), ('inference_size', -1),
                           ('inference_size', True), ('strengths', []),
                           ('samples_per_mode', 0), ('samples_per_mode', True), ('samples_per_mode', 1.5),
                           ('strengths', [0]), ('strengths', [1.1]), ('strengths', [.5, .25]),
                           ('strengths', [.5, .5]), ('strengths', [True])):
            config = deepcopy(original)
            config.pop('config_path')
            config['reporting'][key] = value
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_training_config(path)

    def test_retired_augmentation_and_report_keys_are_rejected(self):
        import yaml
        original = load_training_config()
        original.pop('config_path')
        for section, old, current in (('augmentation', 'start_probability', 'start_ratio'),
                                      ('reporting', 'sample_images', 'samples_per_mode')):
            config = deepcopy(original)
            config[section][old] = config[section].pop(current)
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_training_config(path)

    def test_infinite_psnr_is_serialized_and_plotted(self):
        history = [{'epoch': 1, 'train_loss': 1., 'validation': validation(1, float('inf'), 1, 0)}]
        summary = {'history': history, 'best_epoch': 1, 'best_metric': 'psnr',
                   'statistics': metric_statistics(history, 1)}
        with tempfile.TemporaryDirectory() as directory:
            write_training_report(summary, directory)
            data = json.loads((Path(directory) / 'statistics.json').read_text())
            self.assertEqual(data['metrics']['psnr']['best'], 'Infinity')
            self.assertGreater((Path(directory) / 'training_metrics.png').stat().st_size, 0)

    def test_augmentation_strengths_are_isolated_and_reproducible(self):
        config = load_training_config()['augmentation']
        rgb = np.linspace(.15, .85, 32 * 40 * 3, dtype=np.float32).reshape(32, 40, 3)
        for mode, fraction in [('original', 0.), *[(m, f) for m in MODES[1:] for f in (.25, .5, .75, 1.)]]:
            image, params = augment_image(rgb, mode, fraction, config, seed=42, sample_id='fixture/1')
            other, _ = augment_image(rgb, mode, fraction, config, seed=42, sample_id='fixture/1')
            np.testing.assert_array_equal(image, other)
            self.assertEqual(image.shape, rgb.shape)
            self.assertTrue(0 <= image.min() <= image.max() <= 1)
            if mode == 'noise':
                self.assertAlmostEqual(params['sigma'], .20 * fraction)
            elif mode != 'original':
                self.assertAlmostEqual(params['factor'], 1 + fraction)
            else:
                np.testing.assert_array_equal(image, rgb)







if __name__ == '__main__':
    unittest.main()
