"""Run with: uv run --locked python -m unittest tool.test_training -v."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mlxdlss.features import NetworkGeometry
from mlxdlss.topology import HT

from model.card.loss import (DistillationLoss, normalized_feature_mse,
                             token_affinity_loss, align_features)
from model.student import EchoRenderStudent
from tool.training_config import load_training_config, TrainingSchedule
from tool.teacher_pipeline import PreparedImage, prepare_image, compose_image, compose_rgb_tensor
from tool.teacher_features import FeatureRecordingBackend
from tool.distillation import train_batch, sample_objective, StudentAugmentation
from tool.trainer import fit
from tool.dataloader import SintelImageDataset, build_dataloader, load_data_config
from tool.monitoring import GPUMonitor
from torchvision.transforms import functional as TF



def tiny_model():
    rows = [
        [-1, 1, 'InputAdapter', [4]], [-1, 1, 'SwinStage', [4, 1, 1, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SSMStage', [4, 1, 2, 2, 1, 'cross']],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SSMStage', [4, 1, 2, 2, 1, 'cross']],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SSMStage', [4, 1, 2, 2, 1, 'cross']],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SwinStage', [4, 1, 1, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'ViTStage', [4, 1, 1, 1]],
    ]
    for skip in (9, 7, 5, 3, 1):
        rows.extend([[[ -1, skip], 1, 'Upsample', [4]],
                     [[-1, skip], 1, 'Add', []],
                     [-1, 1, 'SwinStage', [4, 1, 1, 2, 1]]])
    rows.append([-1, 1, 'OutputHead', [4]])
    return EchoRenderStudent({'input_channels': 16, 'output_channels': 4,
                              'input_layout': 'BHWC', 'backbone': rows, 'head': []})


class FixtureTeacher(nn.Module):
    """Synthetic targets for orchestration tests; not a substitute for real Teacher validation."""
    layout = 'BHWC'
    blend_scale = 0.5

    def __init__(self):
        super().__init__()
        self.register_buffer('anchor', torch.tensor(0.1))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, x, return_features=False):
        head = x[..., :4] * self.anchor
        if not return_features:
            return head
        features = {}
        for name, stride in {'e1': 4, 'e2': 8, 'e3': 16, 'vit': 64}.items():
            value = F.adaptive_avg_pool2d(x[..., :4].permute(0, 3, 1, 2),
                                         tuple(math.ceil(size / stride) for size in x.shape[1:3]))
            features[name] = value
        return head, features


def identity_prepare(rgb):
    h, w = rgb.shape[:2]
    features = torch.zeros(1, h, w, 16)
    features[..., 4:7] = torch.from_numpy(rgb) * 0.125 - 0.0625
    features[..., 7:10] = features[..., 4:7]
    return PreparedImage(rgb.copy(), features, NetworkGeometry.identity(w, h))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def setUp(self):
        self.config = load_training_config()

    def criterion(self):
        config = deepcopy(self.config['loss'])
        config.update(ssim_window=3, ms_ssim_weights=[0.3, 0.3, 0.4])
        return DistillationLoss(config, channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))

    def test_config_and_default_architecture(self):
        import yaml
        c = yaml.safe_load(Path(self.config['model']['student_config']).read_text(encoding='utf-8-sig'))
        self.assertEqual([c['backbone'][i][2] for i in (3, 5, 7)], ['SSMStage'] * 3)
        self.assertEqual([c['backbone'][i][3][1] for i in (1, 3, 5, 7, 9, 11)], [5, 4, 6, 8, 8, 8])
        controls = TrainingSchedule(self.config, total_steps=1001)
        self.assertEqual(controls.values(epoch=1, step=0)['lr'], self.config['training']['optimizer']['lr'])
        self.assertEqual(controls.values(epoch=100, step=1000)['lr'], self.config['schedule']['eta_min'])
        data = load_data_config(self.config)
        self.assertEqual(len(data.splits['train'].scenes), 19)
        self.assertTrue(Path(self.config['model']['student_config']).is_absolute())

    def test_config_rejects_invalid_values(self):
        import yaml
        for section, key, value in [('training', 'epochs', 0), ('loss', 'output', -1),
                                     ('schedule', 'kind', 'unknown'), ('schedule', 'eta_min', -1),
                                     ('schedule', 'eta_min', 1), ('logging', 'progress', 'yes'),
                                     ('augmentation', 'enabled', 'yes'),
                                     ('augmentation', 'max_probability', 0.251),
                                     ('augmentation', 'start_probability', 0.26),
                                     ('augmentation', 'peak_fraction', 0.75),
                                     ('augmentation', 'stop_fraction', 1.1),
                                     ('augmentation', 'noise_std', -0.01),
                                     ('augmentation', 'brightness', 1.1),
                                     ('augmentation', 'contrast', float('nan'))]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                c = deepcopy(self.config)
                c.pop('config_path')
                c[section][key] = value
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(c), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_training_config(path)

    def test_cosine_schedule_endpoints_monotonicity_and_constant_controls(self):
        snapshot = deepcopy(self.config)
        schedule = TrainingSchedule(self.config, total_steps=5)
        controls = [schedule.values(epoch=1 + step // 2, step=step) for step in range(5)]
        rates = [row['lr'] for row in controls]
        initial, minimum = self.config['training']['optimizer']['lr'], self.config['schedule']['eta_min']
        self.assertEqual(rates[0], initial)
        self.assertAlmostEqual(rates[2], (initial + minimum) / 2)
        self.assertEqual(rates[-1], minimum)
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertEqual(schedule.values(epoch=100, step=1000)['lr'], minimum)
        self.assertTrue(all(row['loss_weights'] == controls[0]['loss_weights'] for row in controls))
        self.assertTrue(all(row['grad_clip'] == controls[0]['grad_clip'] for row in controls))
        self.assertEqual(self.config, snapshot)
        one_step = TrainingSchedule(self.config, total_steps=1)
        self.assertEqual(one_step.values(epoch=1, step=0)['lr'], initial)
        self.assertEqual(one_step.values(epoch=1, step=1)['lr'], minimum)
        for steps in (None, 0, -1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                TrainingSchedule(self.config, total_steps=steps)
        with self.assertRaises(ValueError):
            schedule.values(epoch=1, step=-1)
        constant_config = deepcopy(self.config)
        constant_config['schedule']['kind'] = 'constant'
        constant = TrainingSchedule(constant_config)
        self.assertEqual(constant.values(epoch=1, step=0), constant.values(epoch=100, step=1000))

    def test_normalization_scale_invariance_and_teacher_stopgrad(self):
        s = torch.rand(1, 4, 3, 3, requires_grad=True)
        t = (s.detach() * 5).requires_grad_()
        loss = normalized_feature_mse(s, t)
        self.assertLess(loss.item(), 1e-12)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(s.grad)

    def test_affinity_matches_dense_offdiagonal_and_gradients(self):
        s = torch.randn(2, 4, 3, 3, requires_grad=True)
        t = torch.randn(2, 4, 3, 3, requires_grad=True)
        sn = F.normalize(s.flatten(2).transpose(1, 2), dim=-1)
        tn = F.normalize(t.detach().flatten(2).transpose(1, 2), dim=-1)
        difference = sn @ sn.transpose(1, 2) - tn @ tn.transpose(1, 2)
        mask = ~torch.eye(9, dtype=torch.bool)
        expected = difference[:, mask].square().mean()
        actual = token_affinity_loss(s, t, chunk_size=2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(s.grad.abs().sum().item(), 0)
        self.assertIsNone(t.grad)

    def test_alignment_omits_teacher_internal_padding(self):
        s = torch.ones(1, 4, 10, 10, requires_grad=True)
        t = torch.ones(1, 4, 8, 8)
        t[..., 5:, :] = 100
        a, b = align_features(s, t, image_hw=(192, 256), network_hw=(320, 320), stride=64)
        self.assertEqual(a.shape, (1, 4, 3, 4))
        torch.testing.assert_close(a, b)
        a.sum().backward()
        self.assertGreater(s.grad.sum().item(), 0)

    def test_rgb_composition_matches_inference_and_has_gradient(self):
        prepared = prepare_image(np.random.default_rng(7).random((192, 192, 3), dtype=np.float32))
        head = (torch.randn(1, 320, 320, 4) * 0.1).requires_grad_()
        actual = compose_rgb_tensor(head, prepared, blend_scale=0.5)
        expected = compose_image(head.detach(), prepared, blend_scale=0.5)
        np.testing.assert_allclose(actual.detach().permute(0, 2, 3, 1).numpy()[0], expected, atol=2e-7)
        actual.mean().backward()
        self.assertGreater(head.grad[..., :3].abs().sum().item(), 0)
        self.assertGreater(head.grad[..., 3:].abs().sum().item(), 0)

    def test_total_loss_formula_and_projection_gradients(self):
        criterion = self.criterion()
        s = {name: torch.rand(1, 4, 4, 4, requires_grad=True) for name in ('e1', 'e2', 'e3', 'vit')}
        t = {name: torch.rand(1, 4, max(1, 128 // stride), max(1, 128 // stride), requires_grad=True)
             for name, stride in {'e1': 4, 'e2': 8, 'e3': 16, 'vit': 64}.items()}
        sh, th = torch.rand(1, 128, 128, 4, requires_grad=True), torch.rand(1, 128, 128, 4, requires_grad=True)
        sr, tr = torch.rand(1, 3, 128, 128, requires_grad=True), torch.rand(1, 3, 128, 128, requires_grad=True)
        loss, parts = criterion(sh, th, s, t, sr, tr, image_hw=(128, 128), network_hw=(128, 128))
        mid = (parts['e1'] + parts['e2'] + parts['e3'] + 0.5 * parts['vit_feature'] + parts['vit_affinity']) / 4
        rgb = 0.5 * parts['rgb_mse'] + 0.2 * parts['rgb_ssim'] + 0.3 * parts['rgb_ms_ssim']
        self.assertAlmostEqual(loss.item(), 3 * (0.5 * mid + 0.5 * parts['output']) + rgb, places=5)
        weighted = sum(value for key, value in parts.items() if key.startswith('weighted_'))
        self.assertAlmostEqual(loss.item(), weighted, places=5)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in criterion.parameters()))
        self.assertTrue(all(value.grad is None for value in t.values()))
        self.assertIsNone(th.grad)
        self.assertIsNone(tr.grad)
        self.assertGreater(sr.grad.abs().sum().item(), 0)

    def test_ms_ssim_identical_and_small_image_error(self):
        c = DistillationLoss(self.config['loss'])
        rgb = torch.rand(1, 3, 192, 192)
        terms = c.image_terms(rgb, rgb)
        self.assertTrue(all(abs(value.item()) < 1e-6 for value in terms.values()))
        with self.assertRaisesRegex(ValueError, 'MS-SSIM'):
            c.image_terms(rgb[..., :64, :64], rgb[..., :64, :64])

    def test_gpu_uuid_mapping_and_poll_interval(self):
        import pynvml
        with patch.object(pynvml, 'nvmlInit'), patch.object(pynvml, 'nvmlShutdown'), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(uuid='GPU-fixture')), \
             patch.object(pynvml, 'nvmlDeviceGetHandleByUUID', return_value='handle') as handle, \
             patch.object(pynvml, 'nvmlDeviceGetUtilizationRates', return_value=SimpleNamespace(gpu=87)) as utilization:
            monitor = GPUMonitor('cuda:1', interval=100)
            self.assertEqual(monitor.display(), '87%')
            self.assertEqual(monitor.sample(), 87)
            self.assertEqual(utilization.call_count, 1)
            handle.assert_called_once_with('GPU-fixture')
            monitor.close()
        self.assertEqual(GPUMonitor('cpu').display(), 'N/A')

    def test_sintel_loading_from_central_config(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            c = deepcopy(self.config)
            c['dataset']['root'] = directory
            c['dataset']['splits']['train']['scenes'] = ['train_scene']
            c['dataset']['splits']['val']['scenes'] = ['val_scene']
            for scene in ('train_scene', 'val_scene'):
                path = Path(directory) / 'training/clean' / scene / 'frame_0001.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(np.zeros((192, 200, 3), dtype=np.uint8)).save(path)
            loader = build_dataloader(load_data_config(c), split='train')
            batch = next(iter(loader))
            self.assertEqual(batch['images'][0].shape, (3, 192, 200))
            self.assertEqual(batch['sample_ids'], ['training/clean/train_scene/frame_0001'])

    def test_teacher_original_forward_capture_grids(self):
        # Preserve the official graph/sampling, stub only learned block operations.
        backend = FeatureRecordingBackend.__new__(FeatureRecordingBackend)
        nn.Module.__init__(backend)
        backend.topology = HT
        backend._captures = None
        backend._window_names = {8: 'e1', 14: 'e2', 22: 'e3'}
        def weight(name):
            if 'input_adapter_weight' in name:
                return torch.zeros(16, 32)
            if name.endswith('weight0'):
                block = int(name.split('.')[0][5:])
                widths = {4: (32, 64), 8: (64, 128), 14: (128, 256), 22: (256, 512),
                          48: (512, 256), 56: (256, 128), 62: (128, 64), 66: (64, 32)}
                return torch.zeros(*widths[block])
            if name == 'block30.layer4.weight':
                return torch.zeros(512, 1024)
            if name == 'block39.layer0.conv_weight':
                return torch.zeros(1024, 512)
            if name == 'block39.layer0.inp_upsample_sin':
                return torch.ones(512)
            if name.endswith(('out_gain', 'out_conv_weight')):
                return torch.zeros(16, 4)
            return torch.tensor(1.0)
        backend.weight = weight
        with patch('mlxdlss.model.NeuralRenderingModel._window', lambda self, value, index, **kw: value), \
             patch('mlxdlss.model.NeuralRenderingModel._global', lambda self, value, index: value), \
             patch.object(backend, '_split_window', side_effect=lambda value, index: value):
            head, captures = backend.forward_with_features(torch.zeros(1, 320, 320, 16))
        self.assertEqual(head.shape, (1, 320, 320, 4))
        self.assertEqual({k: tuple(v.shape) for k, v in captures.items()},
                         {'e1': (1, 64, 80, 80), 'e2': (1, 128, 40, 40),
                          'e3': (1, 256, 20, 20), 'vit': (1, 1024, 8, 8)})

    def test_teacher_capture_selection_and_cleanup(self):
        # Test the actual subclass interception against the pinned base methods.
        # Synthetic method outputs isolate capture behavior from external weights.
        backend = FeatureRecordingBackend.__new__(FeatureRecordingBackend)
        nn.Module.__init__(backend)
        backend.topology = HT
        backend._captures = None
        backend._window_names = {8: 'e1', 14: 'e2', 22: 'e3'}
        def base_window(model, value, index, **kwargs):
            return value + index
        def base_global(model, value, index):
            return value + index
        def forward(model, x):
            for index in (8, 14, 22):
                model._window(x, index, head_count=1, publish=False)
            model._global(x, 38)
            return x
        with patch('mlxdlss.model.NeuralRenderingModel._window', base_window), \
             patch('mlxdlss.model.NeuralRenderingModel._global', base_global), \
             patch('mlxdlss.model.NeuralRenderingModel.forward', forward):
            x = torch.zeros(1, 2, 3, 4, requires_grad=True)
            head, features = backend.forward_with_features(x)
            self.assertEqual(set(features), {'e1', 'e2', 'e3', 'vit'})
            self.assertEqual(features['e1'].shape, (1, 4, 2, 3))
            self.assertTrue(all(not value.requires_grad for value in features.values()))
            self.assertIsNone(backend._captures)
        with patch('mlxdlss.model.NeuralRenderingModel.forward', side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError):
                backend.forward_with_features(x)
            self.assertIsNone(backend._captures)

    def test_training_validation_checkpoint_and_schedule(self):
        student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
        optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=0.001)
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        c = deepcopy(self.config)
        c['training'].update(epochs=2, max_steps=1, crop_size=32)
        c['benchmark'].update(warmup=0, repeats=1)
        c['logging']['progress'] = False
        class MetricFixture(nn.Module):
            def forward(self, pred, target):
                return {'psnr': 20.0, 'ssim': 0.8, 'lpips': 0.1}
        class ScheduleFixture(TrainingSchedule):
            calls = []
            def values(self, *, epoch, step):
                self.calls.append((epoch, step))
                result = super().values(epoch=epoch, step=step)
                result['lr'] = 0.002
                return result
        schedule = ScheduleFixture(c, total_steps=1)
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            c['runtime']['output_dir'] = directory
            summary = fit(teacher, student, criterion, [batch], [batch], optimizer, MetricFixture(), c, schedule=schedule)
            self.assertEqual(summary['global_step'], 1)
            self.assertEqual(summary['epochs_run'], 1)
            self.assertTrue(summary['teacher_unchanged'])
            self.assertTrue(summary['student_parameters_updated'])
            self.assertEqual(schedule.calls, [(1, 0)])
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.002)
            checkpoint = torch.load(Path(directory) / 'last.pt', weights_only=True)
            self.assertEqual(checkpoint['format'], 'echorender-ssm-v2')
            self.assertIn('projections.e1.weight', checkpoint['distillation'])
            saved = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(saved['history'][0]['validation']['reference'], 'teacher_rgb')
            self.assertGreater(saved['history'][0]['validation']['benchmark']['student_latency_ms'], 0)
            with self.assertRaises(FileExistsError):
                fit(teacher, student, criterion, [batch], [batch], optimizer, MetricFixture(), c)

    def test_cosine_applied_across_epochs_with_max_steps(self):
        student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
        optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=0.001)
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        c = deepcopy(self.config)
        c['training'].update(epochs=3, max_steps=3, crop_size=32)
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        class MetricFixture(nn.Module):
            def forward(self, pred, target):
                return {'psnr': 20.0, 'ssim': 0.8, 'lpips': 0.1}
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            c['runtime']['output_dir'] = directory
            summary = fit(teacher, student, criterion, [batch, batch], [batch], optimizer, MetricFixture(), c)
            records = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
            updates = [row for row in records if row['event'] == 'train']
            self.assertEqual([row['epoch'] for row in updates], [1, 1, 2])
            initial, minimum = c['training']['optimizer']['lr'], c['schedule']['eta_min']
            for row, expected in zip(updates, (initial, (initial + minimum) / 2, minimum)):
                self.assertAlmostEqual(row['lr'], expected)
            self.assertEqual(optimizer.param_groups[0]['lr'], minimum)
            self.assertEqual(summary['epochs_run'], 2)
            self.assertEqual(summary['global_step'], 3)
            checkpoint = torch.load(Path(directory) / 'last.pt', weights_only=True)
            self.assertEqual(checkpoint['schedule_controls']['lr'], minimum)

    def test_augmentation_schedule_and_disable(self):
        c = deepcopy(self.config)
        c['training']['epochs'] = 100
        augmentation = StudentAugmentation(c)
        self.assertEqual(augmentation.probability(1), 0.10)
        self.assertAlmostEqual(augmentation.probability(25), 0.10 + 0.15 * 24 / 49)
        self.assertEqual(augmentation.probability(50), 0.25)
        self.assertEqual(augmentation.probability(74), 0.25)
        self.assertEqual(augmentation.probability(75), 0)
        self.assertEqual(augmentation.probability(100), 0)
        self.assertTrue(all(0 <= augmentation.probability(epoch) <= 0.25 for epoch in range(1, 102)))
        c['training']['epochs'] = 10
        augmentation = StudentAugmentation(c)
        self.assertEqual([augmentation.probability(e) for e in (1, 5, 7, 8, 10)], [.1, .25, .25, 0, 0])
        c['augmentation']['enabled'] = False
        self.assertEqual(StudentAugmentation(c).probability(1), 0)
        for epochs in (1, 2, 3):
            c['training']['epochs'] = epochs
            c['augmentation']['enabled'] = True
            short = StudentAugmentation(c)
            self.assertTrue(all(0 <= short.probability(e) <= .25 for e in range(1, epochs + 1)))
            self.assertEqual(short.probability(epochs), 0)

    def test_augmentation_noise_plus_exactly_one_adjustment(self):
        rgb = np.linspace(.1, .8, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        prepared = identity_prepare(rgb)
        original = prepared.features.clone()
        for mode, name in enumerate(('brightness', 'contrast', 'sharpness')):
            augmentation = StudentAugmentation(self.config)
            with patch('tool.distillation.prepare_image', identity_prepare), \
                 patch('tool.distillation.torch.rand', side_effect=[torch.tensor(0.), torch.tensor(.5), torch.tensor(.75)]), \
                 patch('tool.distillation.torch.randint', return_value=torch.tensor(mode)), \
                 patch('tool.distillation.torch.randn', return_value=torch.ones(3, 8, 8)), \
                 patch('tool.distillation.TF.adjust_brightness', wraps=TF.adjust_brightness) as brightness, \
                 patch('tool.distillation.TF.adjust_contrast', wraps=TF.adjust_contrast) as contrast, \
                 patch('tool.distillation.TF.adjust_sharpness', wraps=TF.adjust_sharpness) as sharpness:
                result = augmentation.apply(prepared, .25)
                calls = [brightness.call_count, contrast.call_count, sharpness.call_count]
                self.assertEqual(calls, [int(i == mode) for i in range(3)])
                called = (brightness, contrast, sharpness)[mode]
                torch.testing.assert_close(called.call_args.args[0], torch.from_numpy(rgb).permute(2, 0, 1) + .01)
                self.assertAlmostEqual(called.call_args.args[1], 1 + .5 * self.config['augmentation'][name])
                self.assertEqual(augmentation.applied_count, 1)
                self.assertTrue(np.isfinite(result.rgb).all())
                self.assertGreaterEqual(result.rgb.min(), 0)
                self.assertLessEqual(result.rgb.max(), 1)
                self.assertEqual(result.rgb.shape, rgb.shape)
        torch.testing.assert_close(prepared.features, original)
        np.testing.assert_array_equal(prepared.rgb, rgb)

    def test_augmentation_seed_and_noop(self):
        first, second = StudentAugmentation(self.config), StudentAugmentation(self.config)
        prepared = identity_prepare(np.full((8, 8, 3), .5, dtype=np.float32))
        state = first.generator.get_state().clone()
        self.assertIs(first.apply(prepared, 0), prepared)
        torch.testing.assert_close(first.generator.get_state(), state)
        with patch('tool.distillation.prepare_image', identity_prepare):
            for _ in range(20):
                a, b = first.apply(prepared, .25), second.apply(prepared, .25)
                np.testing.assert_array_equal(a.rgb, b.rgb)
            self.assertEqual(first.applied_count, second.applied_count)
        with patch('tool.distillation.torch.rand', return_value=torch.tensor(.25)):
            self.assertIs(first.apply(prepared, .25), prepared)

    def test_augmentation_teacher_targets_clean_and_student_gradients(self):
        student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
        image = torch.rand(3, 32, 32)
        original = image.clone()
        teacher_inputs, student_inputs = [], []
        teacher_hook = teacher.register_forward_pre_hook(lambda module, args: teacher_inputs.append(args[0].clone()))
        student_hook = student.register_forward_pre_hook(lambda module, args: student_inputs.append(args[0].clone()))
        augmentation = StudentAugmentation(self.config)
        with patch('tool.distillation.prepare_image', identity_prepare):
            _, _, _, clean_target = sample_objective(teacher, student, criterion, image, crop_size=32,
                                                     loss_weights=self.config['loss'])
            with patch('tool.distillation.compose_rgb_tensor', wraps=compose_rgb_tensor) as compose:
                loss, _, _, augmented_target = sample_objective(
                    teacher, student, criterion, image, crop_size=32, loss_weights=self.config['loss'],
                    augmentation=augmentation, augmentation_probability=1)
                self.assertFalse(np.array_equal(compose.call_args_list[0].args[1].rgb,
                                                compose.call_args_list[1].args[1].rgb))
            loss.backward()
        teacher_hook.remove()
        student_hook.remove()
        torch.testing.assert_close(teacher_inputs[0], teacher_inputs[1])
        torch.testing.assert_close(clean_target, augmented_target)
        torch.testing.assert_close(image, original)
        self.assertFalse(torch.equal(student_inputs[0], student_inputs[1]))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_augmentation_fit_cutoff_logs_and_clean_validation(self):
        student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
        optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=.001)
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        c = deepcopy(self.config)
        c['training'].update(epochs=4, max_steps=None, crop_size=32)
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        class Metrics(nn.Module):
            def forward(self, pred, target):
                return {'psnr': 20., 'ssim': .8, 'lpips': .1}
        original_apply = StudentAugmentation.apply
        calls = []
        def force_selected(augmentation, prepared, probability):
            calls.append(probability)
            return original_apply(augmentation, prepared, 1 if probability else 0)
        with tempfile.TemporaryDirectory() as directory, \
             patch('tool.distillation.prepare_image', identity_prepare), \
             patch.object(StudentAugmentation, 'apply', force_selected):
            c['runtime']['output_dir'] = directory
            summary = fit(teacher, student, criterion, [batch], [batch], optimizer, Metrics(), c)
            rows = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
            updates = [row for row in rows if row['event'] == 'train']
            self.assertEqual(calls, [.1, .25, 0, 0])
            self.assertEqual([row['augmentation_probability'] for row in updates], calls)
            self.assertEqual([row['augmented_samples'] for row in updates], [1, 1, 0, 0])
            self.assertEqual([row['loss_weights'] for row in updates], [updates[0]['loss_weights']] * 4)
            self.assertEqual(updates[0]['lr'], .001)
            self.assertEqual(updates[-1]['lr'], .000001)
            self.assertTrue(summary['teacher_unchanged'])


if __name__ == '__main__':
    unittest.main()
