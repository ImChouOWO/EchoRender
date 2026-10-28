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
from tool.training_config import load_training_config, TrainingSchedule, reserve_output_dir
from tool.teacher_pipeline import PreparedImage, prepare_image, compose_image, compose_rgb_tensor
from tool.teacher_features import FeatureRecordingBackend
from tool.distillation import train_batch, StudentAugmentation
from tool.trainer import fit
from tool.dataloader import build_dataloader, load_data_config
from tool.monitoring import GPUMonitor
from tool.teacher_features import TeacherFeatureCache, ask_delete_cache
from torchvision.transforms import functional as TF



def tiny_model(motion='raw'):
    rows = [
        [-1, 1, 'InputAdapter', [4]], [-1, 1, 'SwinStage', [4, 1, 1, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SSMStage', [4, 1, 2, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SSMStage', [4, 1, 2, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SSMStage', [4, 1, 2, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'SwinStage', [4, 1, 1, 2, 1]],
        [-1, 1, 'Downsample', [4]], [-1, 1, 'ViTStage', [4, 1, 1, 1]],
    ]
    for skip in (9, 7, 5, 3, 1):
        rows.extend([[[ -1, skip], 1, 'Upsample', [4]],
                     [[-1, skip], 1, 'Add', []],
                     [-1, 1, 'SwinStage', [4, 1, 1, 2, 1]]])
    rows.append([-1, 1, 'OutputHead', [4]])
    return EchoRenderStudent({'motion': motion, 'input_channels': 8, 'output_channels': 4,
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
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache_patch = patch('tool.trainer.FEATURE_CACHE_DIR', Path(directory.name) / 'feats')
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.config = load_training_config()

    def criterion(self):
        config = deepcopy(self.config['loss'])
        config.update(ssim_window=3, ms_ssim_weights=[0.3, 0.3, 0.4])
        return DistillationLoss(config, channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))

    def test_ssm_checkpoint_matches_outputs_and_gradients(self):
        from model.card.mamba import MambaStage
        stage = MambaStage(4, 4, depth=2, state_dim=2, rotation='clockwise', directions=2).double()
        reference = deepcopy(stage)
        for requires_grad in (False, True):
            stage.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            x = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=requires_grad)
            other = x.detach().clone().requires_grad_(requires_grad)
            actual = stage(x)
            expected = reference.blocks(reference.adapter(other))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().mean().backward()
            expected.square().mean().backward()
            for a, b in zip(stage.parameters(), reference.parameters()):
                self.assertIsNotNone(a.grad)
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)
            if requires_grad:
                torch.testing.assert_close(x.grad, other.grad, rtol=1e-10, atol=1e-12)
        with patch('model.card.mamba.checkpoint', side_effect=AssertionError('unexpected checkpoint')):
            with torch.no_grad():
                stage(x)
            stage.eval()(x)
            stage.train().gradient_checkpointing = False
            output = stage(x)
            torch.testing.assert_close(output, reference.blocks(reference.adapter(x)), rtol=0, atol=0)
            output.sum().backward()

    def test_clockwise_scans_rotate_and_reset_each_stage(self):
        from model.card.mamba import MambaStage, MambaBlock
        class PrefixSum(nn.Module):
            def forward(self, x):
                return x.cumsum(1)
        x = torch.arange(30, dtype=torch.float64).reshape(1, 2, 3, 5)
        for rotation in range(4):
            block = MambaBlock(2, rotation='clockwise', directions=2, index=rotation).double()
            block.norm = nn.Identity()
            block.mixer = PrefixSum()
            horizontal = x.flip(3).cumsum(3).flip(3) if rotation in (1, 2) else x.cumsum(3)
            vertical = x.flip(2).cumsum(2).flip(2) if rotation in (2, 3) else x.cumsum(2)
            torch.testing.assert_close(block(x), x + (horizontal + vertical) * 0.5, rtol=0, atol=0)
        for depth in (4, 6, 8):
            stage = MambaStage(4, 4, depth=depth, rotation='clockwise', directions=2)
            self.assertEqual([b.scan_directions for b in stage.blocks],
                             [((i % 4), (i + 1) % 4) for i in range(depth)])

    def test_ssm_scan_config_counts_rotation_and_validation(self):
        from model.card.mamba import MambaBlock
        class PrefixSum(nn.Module):
            def forward(self, x):
                return x.cumsum(1)
        x = torch.arange(30, dtype=torch.float64).reshape(1, 2, 3, 5)
        directional = (x.cumsum(3), x.cumsum(2),
                       x.flip(3).cumsum(3).flip(3), x.flip(2).cumsum(2).flip(2))
        config = tiny_model().config
        for rotation, cycle in (('clockwise', (0, 1, 2, 3)), ('counterclockwise', (0, 3, 2, 1))):
            for count in range(1, 5):
                for index in range(4):
                    block = MambaBlock(2, rotation=rotation, directions=count, index=index).double()
                    block.norm, block.mixer = nn.Identity(), PrefixSum()
                    expected_directions = tuple(cycle[(index + n) % 4] for n in range(count))
                    self.assertEqual(block.scan_directions, expected_directions)
                    expected = x + torch.stack([directional[d] for d in expected_directions]).mean(0)
                    torch.testing.assert_close(block(x), expected, rtol=0, atol=0)
                config['ssm'] = {'rotation': rotation, 'directions': count}
                model = EchoRenderStudent(config)
                for layer in (3, 5, 7):
                    self.assertEqual(model.layers[layer].blocks[0].scan_directions, cycle[:count])
        for invalid in ({'rotation': 'invalid', 'directions': 2},
                        {'rotation': 'clockwise', 'directions': 0},
                        {'rotation': 'clockwise', 'directions': 5},
                        {'rotation': 'clockwise', 'directions': True},
                        {'rotation': 'clockwise', 'directions': 1.5},
                        {'directions': 2}):
            config['ssm'] = invalid
            with self.assertRaises(ValueError):
                EchoRenderStudent(config)


    def test_config_and_default_architecture(self):
        import yaml
        c = yaml.safe_load(Path(self.config['model']['student_config']).read_text(encoding='utf-8-sig'))
        self.assertEqual([c['backbone'][i][2] for i in (3, 5, 7)], ['SSMStage'] * 3)
        self.assertEqual(c['input_channels'], 8)
        self.assertEqual(c['ssm'], {'rotation': 'clockwise', 'directions': 2})
        self.assertEqual([c['backbone'][i][3][1] for i in (1, 3, 5, 7, 9, 11)], [2, 2, 4, 6, 8, 8])
        controls = TrainingSchedule(self.config, total_steps=1001)
        self.assertEqual(controls.values(epoch=1, step=0)['lr'], self.config['training']['optimizer']['lr'])
        self.assertEqual(controls.values(epoch=100, step=1000)['lr'], self.config['schedule']['eta_min'])
        data = load_data_config(self.config)
        self.assertEqual(len(data.splits['train'].scenes), 19)
        self.assertTrue(Path(self.config['model']['student_config']).is_absolute())

    def test_config_rejects_invalid_values(self):
        import yaml
        for section, key, value in [('training', 'epochs', 0), ('training', 'gradient_checkpointing', 'false'), ('training', 'micro_batch_size', 0),
                                     ('training', 'micro_batch_size', True), ('training', 'releas', 'yes'), ('loss', 'output', -1),
                                     ('schedule', 'kind', 'unknown'), ('schedule', 'eta_min', -1),
                                     ('schedule', 'eta_min', 1), ('logging', 'progress', 'yes'),
                                     ('augmentation', 'enabled', 'yes'),
                                     ('augmentation', 'max_ratio', 0.251),
                                     ('augmentation', 'start_ratio', 0.26),
                                     ('augmentation', 'peak_fraction', 0.75),
                                     ('augmentation', 'stop_fraction', 1.1),
                                     ('augmentation', 'noise_std', -0.01),
                                     ('augmentation', 'noise_std', 1.1),
                                     ('augmentation', 'max_ratio', float('nan'))]:
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

    def test_narrow_student_projects_features_to_teacher_widths(self):
        source = dict(e1=2, e2=3, e3=4, vit=4)
        target = dict(e1=4, e2=5, e3=6, vit=4)
        config = dict(self.criterion().config)
        criterion = DistillationLoss(config, channels=target, student_channels=source)
        for name in source:
            torch.testing.assert_close(criterion.projections[name].weight[:, :, 0, 0],
                                       torch.eye(target[name], source[name]))
        features = {name: torch.randn(1, source[name], 4, 4, requires_grad=True) for name in source}
        teacher = {name: torch.randn(1, target[name], max(1, 32 // stride), max(1, 32 // stride),
                                    requires_grad=True)
                   for name, stride in {'e1': 4, 'e2': 8, 'e3': 16, 'vit': 64}.items()}
        head = torch.rand(1, 32, 32, 4, requires_grad=True)
        rgb = torch.rand(1, 3, 32, 32, requires_grad=True)
        loss, _ = criterion(head, torch.rand_like(head), features, teacher, rgb, torch.rand_like(rgb),
                            image_hw=(32, 32), network_hw=(32, 32))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in criterion.parameters()))
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in features.values()))
        self.assertTrue(all(v.grad is None for v in teacher.values()))
        with self.assertRaisesRegex(ValueError, 'projection must be true'):
            DistillationLoss({**config, 'projection': False}, channels=target, student_channels=source)

    def test_current_student_merge_widths_and_feature_projections(self):
        student = EchoRenderStudent(self.config['model']['student_config'])
        source = {name: student.channels[index] for name, index in student.FEATURE_LAYERS.items()}
        criterion = DistillationLoss(self.config['loss'], student_channels=source)
        for name, width in source.items():
            self.assertEqual(criterion.projections[name].in_channels, width)
        for add_index in (13, 16, 19, 22, 25):
            skip_index = student.sources[add_index][1]
            self.assertEqual(student.channels[add_index - 1], student.channels[skip_index])

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












    def test_cache_prompt_answers_and_noninteractive_keep(self):
        with patch('builtins.input', side_effect=['invalid', 'n']):
            self.assertFalse(ask_delete_cache(Path('/tmp/feats')))
        with patch('builtins.input', return_value=' y ') as ask:
            self.assertTrue(ask_delete_cache(Path('/tmp/feats')))
            ask.assert_called_once_with('[Done]: Delete cached features? [Y/N]: ')
        for error in (EOFError(), KeyboardInterrupt()):
            with patch('builtins.input', side_effect=error):
                self.assertFalse(ask_delete_cache(Path('/tmp/feats')))




    def test_output_directory_numbering_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'output' / 'test'
            self.assertEqual(reserve_output_dir(base), base)
            old_result = base / 'metrics.jsonl'
            old_result.write_text('preserve')
            self.assertEqual(reserve_output_dir(base), base.with_name('test_2'))
            self.assertEqual(reserve_output_dir(base), base.with_name('test_3'))
            self.assertEqual(old_result.read_text(), 'preserve')
            base.with_name('test_4').write_text('existing file')
            self.assertEqual(reserve_output_dir(base), base.with_name('test_5'))

    def test_output_directory_reservations_are_unique(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'test'
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda _: reserve_output_dir(base), range(8)))
            self.assertEqual(len(set(paths)), 8)
            self.assertTrue(all(path.is_dir() for path in paths))






    def test_teacher_graph_cpu_path_and_explicit_cleanup(self):
        from unittest.mock import Mock
        model = FeatureRecordingBackend.__new__(FeatureRecordingBackend)
        nn.Module.__init__(model)
        model._graph_enabled = True
        model._eager_with_features = Mock(return_value=('head', 'features'))
        value = torch.zeros(1, 2, 2, 16)
        self.assertEqual(model.forward_with_features(value), ('head', 'features'))
        model._eager_with_features.assert_called_once_with(value)
        graph = Mock()
        model._graph = graph
        model._graph_input = value
        model._graph_output = (value, {})
        model._graph_biases = {'bias': value}
        model._clear_graph()
        graph.reset.assert_called_once()
        self.assertIsNone(model._graph)
        self.assertIsNone(model._graph_input)
        self.assertIsNone(model._graph_output)
        self.assertIsNone(model._graph_biases)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_teacher_cuda_graph_replays_new_inputs_and_releases_on_cpu_move(self):
        class GraphFixture(FeatureRecordingBackend):
            def __init__(self, device):
                nn.Module.__init__(self)
                self.register_buffer('anchor', torch.tensor(.125, device=device))
                self._graph_enabled = True
                self._graph = None
                self._graph_biases = None
                self._weight_attributes = {}
            def _eager_with_features(self, value):
                head = value[..., :4] * self.anchor
                features = head.permute(0, 3, 1, 2).contiguous()
                return head, {'e1': features, 'e2': features + 1, 'e3': features + 2, 'vit': features + 3}
        device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
        model = GraphFixture(device)
        with torch.inference_mode():
            for value in (torch.zeros(1, 8, 8, 16, device=device), torch.ones(1, 8, 8, 16, device=device)):
                expected_head, expected_features = model._eager_with_features(value)
                head, features = model.forward_with_features(value)
                torch.testing.assert_close(head, expected_head, rtol=0, atol=0)
                for name in expected_features:
                    torch.testing.assert_close(features[name], expected_features[name], rtol=0, atol=0)
            self.assertIsNotNone(model._graph)
            model.cpu()
            self.assertIsNone(model._graph)
            self.assertIsNone(model._graph_biases)
            self.assertEqual(model.anchor.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
