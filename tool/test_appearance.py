"""Appearance formulas, Student-only gradients, integration and independent logging."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from model.card.appearance import AppearanceLoss, APPEARANCE_TERMS, rgb_to_lab, roi_means
from model.card.loss import DistillationLoss
from .training_config import load_training_config, TrainingSchedule
from .reporting import write_training_report, metric_statistics


class AppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(12)
        self.config = deepcopy(load_training_config())
        self.config['loss'].update(rgb_lpips=0., ssim_window=3, ms_ssim_weights=[1.])
        self.appearance = AppearanceLoss(self.config['loss']['appearance_config'])

    def test_lab_matches_d65_reference_primaries(self):
        rgb = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        expected = torch.tensor([[0., 0., 0.], [100., 0., 0.], [53.241, 80.092, 67.203],
                                 [87.735, -86.183, 83.179], [32.296, 79.186, -107.857]])
        actual = rgb_to_lab(rgb[:, :, None, None])[:, :, 0, 0]
        torch.testing.assert_close(actual, expected, atol=.02, rtol=0)

    def test_roi_pool_matches_explicit_boundary_windows_and_covers_every_pixel(self):
        value = torch.rand(2, 5, 101, 155, requires_grad=True)
        starts = lambda n: sorted(set([*range(0, n - 64 + 1, 32), n - 64]))
        expected = torch.stack([torch.stack([value[..., y:y + 64, x:x + 64].mean((-2, -1))
                                           for x in starts(155)], -1) for y in starts(101)], -2)
        actual = roi_means(value, 64, 32)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
        actual.sum().backward()
        self.assertTrue((value.grad > 0).all())

    def test_roi_lcs_and_local_contrast_match_population_statistics(self):
        a = torch.rand(1, 1, 64, 64, dtype=torch.float64) * .4 + .2
        b = torch.rand_like(a) * .4 + .3
        va, vb = a.var(correction=0), b.var(correction=0)
        covariance = ((a - a.mean()) * (b - b.mean())).mean()
        expected = {'roi_l_score': (2 * a.mean() * b.mean() + .01**2) / (a.mean().square() + b.mean().square() + .01**2),
                    'roi_c_score': (2 * (va * vb).sqrt() + .03**2) / (va + vb + .03**2),
                    'roi_s_score': (covariance + .03**2 / 2) / ((va * vb).sqrt() + .03**2 / 2),
                    'roi_contrast': (va.sqrt() - vb.sqrt()).abs()}
        actual = self.appearance.roi_terms(a.float(), b.float())
        for name, reference in expected.items():
            torch.testing.assert_close(actual[name].double(), reference, atol=2e-6, rtol=0)
        for component in 'lcs':
            self.assertAlmostEqual(float(actual[f'roi_{component}'] + actual[f'roi_{component}_score']), 1., places=6)

    def test_structure_is_invariant_to_positive_affine_change_but_contrast_is_not(self):
        a = torch.rand(1, 1, 96, 99) * .3 + .2
        scores = self.appearance.roi_terms(a * 1.5 + .1, a)
        self.assertGreater(float(scores['roi_s_score']), .99999)
        self.assertLess(float(scores['roi_l_score']), .99)
        self.assertLess(float(scores['roi_c_score']), .99)
        self.assertGreater(float(scores['roi_contrast']), .01)

    def test_frequency_bands_keep_signed_detail_and_reconstruct_rgb(self):
        rgb = torch.zeros(1, 3, 64, 64)
        rgb[..., ::2, ::2] = 1
        low, high = self.appearance.frequency_bands(rgb)
        torch.testing.assert_close(low + high, rgb, atol=1e-7, rtol=0)
        self.assertLess(float(high.min()), 0)
        self.assertGreater(float(high.max()), 0)
        constant = self.appearance.frequency_bands(torch.full_like(rgb, .4))[1]
        torch.testing.assert_close(constant, torch.zeros_like(constant), atol=2e-7, rtol=0)

    def test_identical_and_flat_images_have_zero_losses_and_finite_gradients(self):
        for rgb in (torch.rand(1, 3, 80, 99), torch.zeros(1, 3, 64, 64), torch.ones(1, 3, 64, 64)):
            prediction, target = rgb.clone().requires_grad_(), rgb.clone().requires_grad_()
            terms = self.appearance(prediction, target)
            for name in APPEARANCE_TERMS:
                self.assertAlmostEqual(float(terms[name].detach()), 0., delta=2e-6)
            sum(terms[name] for name in APPEARANCE_TERMS).backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertIsNone(target.grad)
            self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_every_term_backpropagates_only_to_student(self):
        for name in APPEARANCE_TERMS:
            prediction = torch.rand(2, 3, 64, 64, requires_grad=True)
            target = torch.rand_like(prediction, requires_grad=True)
            terms = self.appearance(prediction, target)
            terms[name].backward()
            self.assertTrue(torch.isfinite(prediction.grad).all(), name)
            self.assertGreater(float(prediction.grad.abs().sum()), 0, name)
            self.assertIsNone(target.grad)

    def test_brightness_masks_are_teacher_defined_and_empty_groups_are_safe(self):
        target = torch.cat((torch.full((1, 3, 64, 32), .1), torch.full((1, 3, 64, 32), .9)), -1)
        prediction = target + .04
        terms = self.appearance(prediction, target)
        error = (rgb_to_lab(prediction)[:, :1] - rgb_to_lab(target)[:, :1]).abs()
        self.assertAlmostEqual(float(terms['lab_l_dark_mae']), float(error[..., :32].mean()), places=4)
        self.assertAlmostEqual(float(terms['lab_l_bright_mae']), float(error[..., 32:].mean()), places=4)
        self.assertEqual(float(terms['lab_l_dark_fraction']), .5)
        empty = self.appearance(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))
        self.assertEqual(float(empty['lab_l_bright_mae']), 0.)

    def test_total_adds_independent_weighted_terms_and_zero_disables_computation(self):
        config = self.config['loss']
        criterion = DistillationLoss(config, stages=['RGB composition'], channels={})
        head = torch.zeros(1, 64, 64, 4)
        prediction = torch.rand(1, 3, 64, 64, requires_grad=True)
        teacher = torch.rand_like(prediction, requires_grad=True)
        source = torch.rand_like(prediction, requires_grad=True)
        arguments = (head, head, {}, {}, prediction, teacher)
        options = dict(image_hw=(64, 64), network_hw=(64, 64), clean_source=source)
        loss, parts = criterion(*arguments, **options)
        base, old = criterion(*arguments, **options, weights={'appearance': 0.})
        self.assertAlmostEqual(float(loss.detach() - base.detach()), config['appearance'] * parts['appearance'], places=6)
        self.assertAlmostEqual(float(loss.detach()), sum(v for k, v in parts.items() if k.startswith('weighted_')), places=6)
        for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim', 'rgb_lpips', 'output', 'distillation'):
            self.assertEqual(parts[name], old[name])
        with patch.object(criterion.appearance, 'forward', side_effect=AssertionError('disabled appearance ran')):
            criterion(*arguments, **options, weights={'appearance': 0.})
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(source.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.)

    def test_cpu_autocast_keeps_appearance_fp32_and_finite(self):
        prediction = torch.rand(1, 3, 64, 64, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            terms = self.appearance(prediction, torch.rand_like(prediction))
        self.assertTrue(all(value.dtype == torch.float32 for value in terms.values()))
        sum(terms[name] for name in APPEARANCE_TERMS).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_config_and_schedule_validate_new_settings(self):
        original = self.config
        schedule = TrainingSchedule(original, total_steps=2).values(epoch=1, step=0)
        self.assertEqual(schedule['loss_weights']['appearance'], original['loss']['appearance'])
        invalid = [('roi_size', 1), ('roi_stride', 65), ('roi_stride', True), ('gaussian_size', 10),
                   ('gaussian_sigma', 0), ('brightness_split', 100), ('weights', {'lab_l': 1.})]
        for name, value in invalid:
            config = deepcopy(original)
            config.pop('config_path')
            config['loss']['appearance_config'][name] = value
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_training_config(path)

    def test_reports_keep_each_component_and_score_separate(self):
        terms = self.appearance(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
        parts = {f'appearance_{name}': float(value) for name, value in terms.items()}
        parts.update(total=1., appearance=.2)
        validation = {'components': parts, 'image_mean': {'psnr': 20., 'ssim': .8, 'lpips': .2}}
        history = [{'epoch': 1, 'train_loss': 2., 'train_components': parts, 'validation': validation}]
        summary = {'history': history, 'best_epoch': 1, 'best_metric': 'psnr', 'statistics': metric_statistics(history, 1)}
        with tempfile.TemporaryDirectory() as directory:
            write_training_report(summary, directory)
            with (Path(directory) / 'appearance.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['split'] for row in rows], ['train', 'val'])
            self.assertIn('appearance_roi_s_score', rows[0])
            self.assertIn('appearance_high_frequency', rows[0])
            self.assertGreater((Path(directory) / 'appearance_metrics.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
