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
from .prediction_report import (augment_image, export_predictions, predict_resized, prepare_variant,
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

    def test_best_and_last_checkpoints_report_and_callback(self):
        config = load_training_config()
        config['training'].update(epochs=3, max_steps=None, crop_size=32, releas=False)
        config['logging']['progress'] = False
        config['benchmark']['enabled'] = False
        student, teacher = tiny_model(), FixtureTeacher()
        loss_config = {**config['loss'], 'ssim_window': 3, 'ms_ssim_weights': [.3, .3, .4]}
        criterion = DistillationLoss(loss_config, channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        results = [validation(3, 20, .5, .4), validation(2, 22, .6, .3), validation(1, 21, .65, .25)]
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            config['runtime']['output_dir'] = str(Path(temporary) / 'run')
            directory = Path(temporary) / 'reports'
            def completed(summary):
                calls.append(summary['best_epoch'])
                self.assertEqual(next(student.parameters()).device.type, 'cpu')
                self.assertTrue((directory / 'training_metrics.png').is_file())
            text = io.StringIO()
            with patch('tool.distillation.prepare_image', identity_prepare), \
                 patch('tool.trainer.evaluate', side_effect=results), redirect_stdout(text):
                summary = fit(teacher, student, criterion, [batch], [batch], optimizer, nn.Identity(), config,
                              report_dir=directory, on_complete=completed)
            output = Path(config['runtime']['output_dir'])
            best = torch.load(output / 'best.pt', weights_only=True)
            last = torch.load(output / 'last.pt', weights_only=True)
            self.assertEqual((best['epoch'], last['epoch']), (2, 3))
            self.assertEqual(calls, [2])
            self.assertEqual(sorted(p.name for p in output.glob('*.pt')), ['best.pt', 'last.pt'])
            self.assertEqual(text.getvalue().count('[Best]: save best weight'), 2)
            for label in ('Max', 'Min', 'Mean', 'Best', 'psnr', 'ssim', 'lpips', 'train_loss'):
                self.assertIn(label, text.getvalue())
            self.assertEqual(summary['statistics']['loss']['best_epoch'], 3)
            self.assertEqual(summary['statistics']['loss']['at_best_checkpoint'], 2)
            self.assertEqual(summary['statistics']['psnr']['mean'], 21)
            for name in ('statistics.json', 'statistics.csv', 'epochs.csv', 'training_metrics.png'):
                self.assertTrue((directory / name).is_file())

    def test_export_failure_keeps_cache_and_releases_lock(self):
        from .teacher_features import TeacherFeatureCache
        config = load_training_config()
        config['training'].update(epochs=1, max_steps=1, crop_size=32, releas=True)
        config['logging']['progress'] = False
        config['benchmark']['enabled'] = False
        student, teacher = tiny_model(), FixtureTeacher()
        criterion = DistillationLoss({**config['loss'], 'ssim_window': 3,
                                     'ms_ssim_weights': [.3, .3, .4]},
                                    channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        def fail(summary):
            raise RuntimeError('Injected export failure')
        class Metrics(nn.Module):
            def forward(self, prediction, target):
                return {'psnr': 20., 'ssim': .5, 'lpips': .4}
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / 'feats'
            config['runtime']['output_dir'] = str(Path(temporary) / 'run')
            with patch('tool.distillation.prepare_image', identity_prepare),                  patch('tool.trainer.FEATURE_CACHE_DIR', cache_dir), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Injected export failure'):
                    fit(teacher, student, criterion, [batch], [batch], optimizer, Metrics(), config,
                        on_complete=fail, cache_cleanup=lambda _: False)
            self.assertTrue((cache_dir / 'manifest.json').is_file())
            self.assertTrue((Path(config['runtime']['output_dir']) / 'best.pt').is_file())
            cache = TeacherFeatureCache(cache_dir)
            cache.open()
            cache.close(delete=False)

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

    def test_resized_inference_preserves_detail_with_zero_residual(self):
        class ZeroHead(nn.Module):
            def forward(self, x):
                return torch.zeros((*x.shape[:-1], 4), device=x.device)
        pattern = (np.indices((101, 257)).sum(0) % 2).astype(np.float32)
        rgb = np.repeat(pattern[..., None], 3, axis=2)
        with patch('tool.prediction_report.prepare_image', identity_prepare):
            prediction, geometry = predict_resized(ZeroHead(), VariantInput(rgb, resize_rgb(rgb, 64), {}),
                                                   device='cpu', blend_scale=.5)
        np.testing.assert_array_equal(prediction, rgb)
        self.assertEqual(geometry['image_hw'], [101, 257])
        self.assertEqual(geometry['resized_hw'], [25, 64])

    def test_sampling_reads_only_ten_random_sources_and_repeats_with_seed(self):
        from torch.utils.data import DataLoader, Dataset
        class Sources(Dataset):
            def __init__(self):
                self.read = []
            def __len__(self):
                return 200
            def __getitem__(self, index):
                self.read.append(index)
                return {'image': torch.zeros(3, 8, 8), 'sample_id': f'fixture/{index}'}
        dataset = Sources()
        loader = DataLoader(dataset)
        first, available = sample_modes(loader, 2, 42)
        self.assertEqual(available, 200)
        self.assertEqual(len(dataset.read), 10)
        self.assertEqual(len(set(dataset.read)), 10)
        self.assertNotEqual(dataset.read, list(range(10)))
        repeated, _ = sample_modes(loader, 2, 42)
        changed, _ = sample_modes(loader, 2, 43)
        self.assertEqual([s[1] for group in first.values() for s in group], [s[1] for group in repeated.values() for s in group])
        self.assertNotEqual([s[1] for group in first.values() for s in group], [s[1] for group in changed.values() for s in group])
        small, total = sample_modes([{'images': [torch.zeros(3, 8, 8)], 'sample_ids': ['only']}], 2, 42)
        self.assertEqual((len(small), total), (5, 1))
        self.assertTrue(all(len(group) == 1 for group in small.values()))

    def test_noise_is_applied_after_resize_and_reaches_the_model(self):
        config = load_training_config()['augmentation']
        rgb = np.full((200, 600, 3), .5, dtype=np.float32)
        small = resize_rgb(rgb, 96)
        variant = prepare_variant(
            rgb, 'noise', 1., config, seed=42, sample_id='fixture/noise', inference_size=96)
        display, network_rgb, parameters = variant.display_rgb, variant.network_rgb, variant.parameters
        self.assertAlmostEqual(float((network_rgb - small).std()), .20, delta=.004)
        old_augmented, _ = augment_image(rgb, 'noise', 1., config, seed=42, sample_id='fixture/noise')
        old_noise = resize_rgb(old_augmented, 96) - small
        self.assertGreater(parameters['inference_delta_std'], float(old_noise.std()) * 4)
        class ZeroHead(nn.Module):
            def forward(self, inputs):
                self.seen = inputs.detach().clone()
                return torch.zeros((*inputs.shape[:-1], 4))
        model = ZeroHead()
        with patch('tool.prediction_report.prepare_image', identity_prepare):
            prediction, geometry = predict_resized(model, variant, device='cpu', blend_scale=.5)
        torch.testing.assert_close(model.seen, identity_prepare(network_rgb).features)
        np.testing.assert_allclose(prediction, display, atol=1e-7)
        self.assertEqual(geometry['resized_hw'], [32, 96])
        self.assertGreater(float(np.abs(display - rgb).mean()), 0)

    def test_every_variant_is_preprocessed_then_independently_predicted(self):
        config = load_training_config()['augmentation']
        rgb = np.random.default_rng(42).uniform(.15, .75, (32, 40, 3)).astype(np.float32)
        class BiasedHead(nn.Module):
            def __init__(self, layout):
                super().__init__()
                self.layout, self.calls = layout, []
            def forward(self, inputs):
                x = inputs if self.layout == 'BHWC' else inputs.permute(0, 2, 3, 1)
                self.calls.append(x.clone())
                head = torch.full((*x.shape[:-1], 4), .125)
                head[..., 3] = 0.
                return head if self.layout == 'BHWC' else head.permute(0, 3, 1, 2)
        for layout in ('BHWC', 'BCHW'):
            model, predictions = BiasedHead(layout), {}
            with patch('tool.prediction_report.prepare_image', identity_prepare):
                for mode, fraction in [('original', 0.), *[(m, f) for m in MODES[1:] for f in (.25, .5, .75, 1.)]]:
                    variant = prepare_variant(rgb, mode, fraction, config, seed=42,
                        sample_id='fixture/input', inference_size=0)
                    prediction, geometry = predict_resized(model, variant, device='cpu',
                        blend_scale=.5, layout=layout)
                    shown, parameters = variant.display_rgb, variant.parameters
                    expected_input, _ = augment_image(rgb, mode, fraction, config,
                        seed=42, sample_id='fixture/input')
                    np.testing.assert_array_equal(shown, expected_input)
                    torch.testing.assert_close(model.calls[-1], identity_prepare(expected_input).features)
                    # Composition clamps the raw RGB update before the gate blends it with input history.
                    expected_prediction = np.clip(expected_input + .03125, 0, 1) * .75 + expected_input * .25
                    np.testing.assert_allclose(prediction, expected_prediction, atol=1e-7)
                    self.assertEqual(parameters['mode'], mode)
                    predictions[(mode, fraction)] = geometry['prediction_sha256']
            self.assertEqual(len(model.calls), 17)
            self.assertEqual(len(set(predictions.values())), 17)

    def test_resize_roundoff_stays_valid_for_real_preparation(self):
        class ZeroHead(nn.Module):
            def forward(self, x):
                return torch.zeros((*x.shape[:-1], 4), device=x.device)
        rgb = np.ones((137, 773, 3), dtype=np.float32)
        prediction, geometry = predict_resized(ZeroHead(), VariantInput(rgb, resize_rgb(rgb, 320), {}),
                                               device='cpu', blend_scale=.5)
        np.testing.assert_array_equal(prediction, rgb)
        self.assertEqual(geometry['network_hw'], [320, 320])

    def test_mapping_crops_padding_and_supports_both_layouts(self):
        from torch.nn import functional as F
        from .teacher_pipeline import PreparedImage, compose_rgb_tensor
        class PatternHead(nn.Module):
            def __init__(self, layout):
                super().__init__()
                self.layout = layout
                self.seen = []
            def forward(self, x):
                if self.layout == 'BCHW':
                    x = x.permute(0, 2, 3, 1)
                self.seen.append(tuple(x.shape[1:3]))
                head = torch.zeros((*x.shape[:-1], 4))
                head[..., 0] = torch.arange(x.shape[2]) * .1
                head[..., 3] = (torch.arange(x.shape[2]) % 2) * 16. - 8.
                return head.permute(0, 3, 1, 2) if self.layout == 'BCHW' else head
        def padded(rgb):
            prepared = identity_prepare(rgb)
            h, w = rgb.shape[:2]
            features = F.pad(prepared.features.permute(0, 3, 1, 2),
                             (0, max(0, 80 - w), 0, max(0, 80 - h))).permute(0, 2, 3, 1)
            return PreparedImage(prepared.rgb, features, prepared.geometry)
        for h, w in ((101, 257), (257, 101)):
            rgb = np.full((h, w, 3), .5, dtype=np.float32)
            for layout in ('BHWC', 'BCHW'):
                model = PatternHead(layout)
                with patch('tool.prediction_report.prepare_image', padded):
                    actual, geometry = predict_resized(model, VariantInput(rgb, resize_rgb(rgb, 64), {}),
                                                        device='cpu', blend_scale=.5, layout=layout)
                small_h, small_w = geometry['resized_hw']
                head = torch.zeros(1, 4, small_h, small_w)
                head[:, 0] = torch.arange(small_w) * .1
                head[:, 3] = (torch.arange(small_w) % 2) * 16. - 8.
                small = resize_rgb(rgb, 64)
                prediction = compose_rgb_tensor(head.permute(0, 2, 3, 1), padded(small), blend_scale=.5)
                correction = prediction - torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0)
                expected = np.clip(rgb + F.interpolate(correction, size=(h, w), mode='bilinear',
                    align_corners=False)[0].permute(1, 2, 0).numpy(), 0, 1)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(model.seen, [(80, 80)])
                self.assertEqual(actual.shape, (h, w, 3))

    def test_export_uses_best_weights_same_inputs_and_sequential_models(self):
        config = load_training_config()
        config['logging']['progress'] = False
        config['reporting']['crop_size'] = 0
        student = tiny_model().eval()
        state = deepcopy(student.state_dict())
        model_config = deepcopy(student.config)
        image = .2 + .6 * torch.rand(3, 32, 40, generator=torch.Generator().manual_seed(42))
        batch = {'images': [image], 'sample_ids': ['training/clean/scene/frame_0001']}
        refs = []
        def create_student(cfg):
            model = tiny_model()
            refs.append(weakref.ref(model))
            return model
        def create_teacher():
            gc.collect()
            self.assertIsNone(refs[0]())
            return FixtureTeacher()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'best.pt'
            torch.save({'format': 'echorender-ssm-v2', 'student': state, 'student_config': model_config,
                        'epoch': 2, 'teacher_blend_scale': .5}, checkpoint)
            # The ambient model is deliberately different; the export must load the file.
            with torch.no_grad():
                for parameter in student.parameters():
                    parameter.fill_(100)
            with patch('tool.prediction_report.EchoRenderStudent', side_effect=create_student), \
                 patch('tool.prediction_report.prepare_image', identity_prepare), \
                 patch('tool.prediction_report.perf_counter', side_effect=itertools.count(0., .01)):
                manifest = export_predictions(checkpoint, create_teacher, {'val': [batch], 'test': [batch]},
                                              config, root / 'reports', 'cpu', missing_splits={'external': 'Fixture missing split'})
            self.assertEqual(manifest['status'], 'completed_with_missing_splits')
            self.assertEqual(manifest['missing_splits'], {'external': 'Fixture missing split'})
            self.assertEqual(manifest['checkpoint_epoch'], 2)
            self.assertEqual(manifest['counts'], {name: {'val': 5, 'test': 5}
                                                 for name in ('preview', 'student', 'teacher')})
            self.assertEqual(manifest['inference_counts'], {name: {'val': 21, 'test': 21}
                                                           for name in ('student', 'teacher')})
            self.assertEqual(len(manifest['grids']), 30)
            records = [json.loads(row) for row in (root / 'reports/predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 84)
            import csv
            with (root / 'reports/predictions.csv').open() as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 84)
            for row in records:
                self.assertIn('prediction_sha256', row)
                self.assertIn('prepared_features_sha256', row)
                if row['parameters']['mode'] == 'original':
                    self.assertEqual(row['prediction_change_rmse'], 0)
                else:
                    self.assertGreater(row['prediction_change_rmse'], 0)
            hashes = {}
            for row in records:
                key = (row['split'], row['mode'], row['parameters']['fraction'])
                if key in hashes:
                    self.assertEqual(hashes[key], row['input_sha256'])
                hashes[key] = row['input_sha256']
                with Image.open(root / 'reports' / row['path']) as panel:
                    self.assertEqual(panel.size, tuple(reversed(row['grid']['canvas_hw'])))
                    levels = [0.] if row['mode'] == 'original' else [0., .25, .5, .75, 1.]
                    column = levels.index(row['parameters']['fraction'])
                    top = row['grid']['row_tops'][0]
                    left = row['grid']['column_lefts'][column]
                    actual_input = np.asarray(panel)[top:top + 32, left:left + 40]
                    expected_input, _ = augment_image(image.permute(1, 2, 0).numpy(),
                        row['parameters']['mode'], row['parameters']['fraction'], config['augmentation'],
                        seed=config['loader']['seed'], sample_id=row['sample_id'])
                    np.testing.assert_array_equal(actual_input, np.rint(expected_input * 255).astype(np.uint8))
                self.assertEqual(len(row['grid']['row_tops']), 2)
                self.assertIn('| total infer ', row['grid']['title'])
                self.assertIn(' s | avg infer ', row['grid']['title'])
                self.assertAlmostEqual(row['inference_seconds'], .01)
                self.assertAlmostEqual(row['total_infer_seconds'] / len(row['grid']['column_lefts']),
                                       row['avg_infer_seconds'])
            record = next(row for row in records if row['model'] == 'student'
                          and row['parameters']['mode'] == 'original')
            from .teacher_pipeline import compose_rgb_tensor
            expected_model = tiny_model().eval()
            expected_model.load_state_dict(state)
            prepared = identity_prepare(image.permute(1, 2, 0).numpy())
            with torch.inference_mode():
                expected = compose_rgb_tensor(expected_model(prepared.features), prepared, blend_scale=.5)
            expected = np.rint(expected[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            with Image.open(root / 'reports' / record['path']) as panel:
                top, left = record['grid']['row_tops'][1], record['grid']['column_lefts'][0]
                actual = np.asarray(panel)[top:top + 32, left:left + 40].copy()
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
