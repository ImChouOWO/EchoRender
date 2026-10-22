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
from tool.distillation import train_batch, sample_objective, StudentAugmentation
from tool.trainer import fit
from tool.dataloader import SintelImageDataset, build_dataloader, load_data_config
from tool.monitoring import GPUMonitor
from tool.teacher_features import TeacherFeatureCache, ask_delete_cache
from torchvision.transforms import functional as TF



def tiny_model():
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

    def test_microbatch_matches_individual_gradients_and_sample_targets(self):
        for ragged in (False, True):
            with self.subTest(ragged=ragged), patch('tool.distillation.prepare_image', identity_prepare):
                student = tiny_model()
                reference = deepcopy(student)
                criterion = self.criterion()
                reference_criterion = deepcopy(criterion)
                teacher = FixtureTeacher()
                sizes = (32, 32, 40, 32, 32) if ragged else (32,) * 5
                images = [torch.rand(3, size, 32) for size in sizes]
                ids = [f'fixture/{i}' for i in (4, 1, 3, 0, 2)]
                batch = {'images': images, 'sample_ids': ids}
                target_by_id = {}
                for image, sample_id in zip(images, ids):
                    prepared = identity_prepare(image.permute(1, 2, 0).numpy())
                    head, features = teacher(prepared.features, return_features=True)
                    target_by_id[sample_id] = (head, features, compose_rgb_tensor(head, prepared, blend_scale=.5))
                class Cache:
                    def __init__(self):
                        self.loaded = []
                    def load(self, split, sample_id, device):
                        self.loaded.append((split, sample_id))
                        return target_by_id[sample_id]
                cache = Cache()
                controls = TrainingSchedule(self.config, total_steps=2).values(epoch=1, step=0)
                params = [*student.parameters(), *criterion.parameters()]
                ref_params = [*reference.parameters(), *reference_criterion.parameters()]
                optimizer = torch.optim.SGD(params, lr=.001)
                ref_optimizer = torch.optim.SGD(ref_params, lr=.001)
                ref_augmentation = StudentAugmentation(self.config)
                losses = []
                for image in images:
                    loss, _, _, _ = sample_objective(teacher, reference, reference_criterion, image,
                        crop_size=0, loss_weights=controls['loss_weights'], augmentation=ref_augmentation,
                        augmentation_ratio=1.0)
                    losses.append(loss.item())
                    (loss / len(images)).backward()
                torch.nn.utils.clip_grad_norm_(ref_params, controls['grad_clip'])
                ref_optimizer.step()
                calls = []
                handle = student.register_forward_pre_hook(lambda mod, args: calls.append(args[0].shape[0]))
                try:
                    result = train_batch(None, student, criterion, batch, optimizer, crop_size=0,
                        controls=controls, target_cache=cache, blend_scale=.5, micro_batch_size=2,
                        augmentation=StudentAugmentation(self.config), augmentation_ratio=1.0)
                finally:
                    handle.remove()
                self.assertEqual(calls, [2, 1, 2] if ragged else [2, 2, 1])
                self.assertEqual(result['micro_batch_sizes'], calls)
                self.assertEqual(result['batch_size'], 5)
                self.assertEqual(result['augmented_samples'], 5)
                self.assertEqual(cache.loaded, [('train', sample_id) for sample_id in ids])
                self.assertAlmostEqual(result['loss'], sum(losses) / 5, places=5)
                for p, expected in zip(params, ref_params):
                    torch.testing.assert_close(p.grad, expected.grad, rtol=2e-4, atol=2e-6)
                    torch.testing.assert_close(p, expected, rtol=2e-5, atol=2e-6)

    def test_config_and_default_architecture(self):
        import yaml
        c = yaml.safe_load(Path(self.config['model']['student_config']).read_text(encoding='utf-8-sig'))
        self.assertEqual([c['backbone'][i][2] for i in (3, 5, 7)], ['SSMStage'] * 3)
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
            for index in (3, 5, 7):
                self.assertEqual(student.layers[index].gradient_checkpointing, c['training']['gradient_checkpointing'])
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
        c['training'].update(epochs=3, max_steps=3, crop_size=32, gradient_checkpointing=True)
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        class MetricFixture(nn.Module):
            def forward(self, pred, target):
                return {'psnr': 20.0, 'ssim': 0.8, 'lpips': 0.1}
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            c['runtime']['output_dir'] = directory
            summary = fit(teacher, student, criterion, [batch, batch], [batch], optimizer, MetricFixture(), c)
            self.assertTrue(all(student.layers[i].gradient_checkpointing for i in (3, 5, 7)))
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
        self.assertEqual(augmentation.ratio(1), 0.10)
        self.assertAlmostEqual(augmentation.ratio(25), 0.10 + 0.15 * 24 / 49)
        self.assertEqual(augmentation.ratio(50), 0.25)
        self.assertEqual(augmentation.ratio(74), 0.25)
        self.assertEqual(augmentation.ratio(75), 0)
        self.assertEqual(augmentation.ratio(100), 0)
        self.assertTrue(all(0 <= augmentation.ratio(epoch) <= 0.25 for epoch in range(1, 102)))
        c['training']['epochs'] = 10
        augmentation = StudentAugmentation(c)
        self.assertEqual([augmentation.ratio(e) for e in (1, 5, 7, 8, 10)], [.1, .25, .25, 0, 0])
        c['augmentation']['enabled'] = False
        self.assertEqual(StudentAugmentation(c).ratio(1), 0)
        for epochs in (1, 2, 3):
            c['training']['epochs'] = epochs
            c['augmentation']['enabled'] = True
            short = StudentAugmentation(c)
            self.assertTrue(all(0 <= short.ratio(e) <= .25 for e in range(1, epochs + 1)))
            self.assertEqual(short.ratio(epochs), 0)

    def test_augmentation_noise_plus_exactly_one_adjustment(self):
        rgb = np.linspace(.1, .6, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        prepared = identity_prepare(rgb)
        original = prepared.features.clone()
        for mode, name in enumerate(('brightness', 'contrast', 'sharpness')):
            augmentation = StudentAugmentation(self.config)
            with patch('tool.distillation.prepare_image', identity_prepare), \
                 patch('tool.distillation.torch.rand', return_value=torch.tensor(0.)), \
                 patch('tool.distillation.torch.randint', return_value=torch.tensor(mode)), \
                 patch('tool.distillation.torch.randn', return_value=torch.ones(3, 8, 8)), \
                 patch('tool.distillation.TF.adjust_brightness', wraps=TF.adjust_brightness) as brightness, \
                 patch('tool.distillation.TF.adjust_contrast', wraps=TF.adjust_contrast) as contrast, \
                 patch('tool.distillation.TF.adjust_sharpness', wraps=TF.adjust_sharpness) as sharpness:
                result = augmentation.apply(prepared, .25)
                self.assertEqual([brightness.call_count, contrast.call_count, sharpness.call_count],
                                 [int(i == mode) for i in range(3)])
                called = (brightness, contrast, sharpness)[mode]
                torch.testing.assert_close(called.call_args.args[0], torch.from_numpy(rgb).permute(2, 0, 1))
                self.assertEqual(called.call_args.args[1], 1.25)
                expected = getattr(TF, 'adjust_' + name)(torch.from_numpy(rgb).permute(2, 0, 1), 1.25)
                expected = (expected + .05).clamp(0, 1).permute(1, 2, 0).numpy()
                np.testing.assert_allclose(result.rgb, expected)
                self.assertEqual(augmentation.applied_count, 1)
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
                    augmentation=augmentation, augmentation_ratio=1)
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
            with patch('tool.distillation.torch.rand', return_value=torch.tensor(0.)):
                return original_apply(augmentation, prepared, probability)
        with tempfile.TemporaryDirectory() as directory, \
             patch('tool.distillation.prepare_image', identity_prepare), \
             patch.object(StudentAugmentation, 'apply', force_selected):
            c['runtime']['output_dir'] = directory
            summary = fit(teacher, student, criterion, [batch], [batch], optimizer, Metrics(), c)
            rows = [json.loads(line) for line in (Path(directory) / 'metrics.jsonl').read_text().splitlines()]
            updates = [row for row in rows if row['event'] == 'train']
            self.assertEqual(calls, [.1, .25, 0, 0])
            self.assertEqual([row['augmentation_ratio'] for row in updates], calls)
            self.assertEqual([row['augmented_samples'] for row in updates], [1, 1, 0, 0])
            self.assertEqual([row['loss_weights'] for row in updates], [updates[0]['loss_weights']] * 4)
            self.assertEqual(updates[0]['lr'], .001)
            self.assertEqual(updates[-1]['lr'], .000001)
            self.assertTrue(summary['teacher_unchanged'])

    def test_feature_cache_matches_online_loss_and_gradients(self):
        c = deepcopy(self.config)
        c['training']['crop_size'] = 32
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
        image = torch.rand(3, 32, 32)
        batch = {'images': [image], 'sample_ids': ['../../fixture/1'], 'scenes': ['fixture']}
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            try:
                cache.build(teacher, [batch], [batch], c)
                self.assertEqual(len(list(cache.directory.glob('*.pt'))), 2)
                targets = cache.load('train', batch['sample_ids'][0], 'cpu')
                self.assertEqual(set(targets[1]), {'e1', 'e2', 'e3', 'vit'})
                self.assertTrue(all(not value.requires_grad for value in targets[1].values()))
                online, _, _, rgb = sample_objective(teacher, student, criterion, image,
                                                     crop_size=32, loss_weights=c['loss'])
                online.backward()
                gradients = [p.grad.clone() for p in student.parameters()]
                student.zero_grad(set_to_none=True)
                criterion.zero_grad(set_to_none=True)
                with patch.object(teacher, 'forward', side_effect=AssertionError('Teacher executed during student phase')):
                    cached, _, _, cached_rgb = sample_objective(teacher, student, criterion, image,
                        crop_size=32, loss_weights=c['loss'], targets=targets)
                    cached.backward()
                torch.testing.assert_close(online, cached, rtol=0, atol=0)
                torch.testing.assert_close(rgb, cached_rgb, rtol=0, atol=0)
                for expected, parameter in zip(gradients, student.parameters()):
                    torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)
                changed = {**batch, 'sample_ids': ['fixture/2']}
                cache.build(teacher, [changed], [changed], c)
                self.assertFalse(cache.path('train', batch['sample_ids'][0]).exists())
                self.assertEqual(len(list(cache.directory.glob('*.pt'))), 2)
            finally:
                cache.close()
            self.assertEqual(list(cache.directory.iterdir()), [])

    def test_feature_cache_lock_and_unrelated_file_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            first = TeacherFeatureCache(Path(directory) / 'feats')
            second = TeacherFeatureCache(first.directory)
            first.open()
            with self.assertRaisesRegex(RuntimeError, 'in use'):
                second.open()
            first.close()
            foreign = first.directory / 'keep.txt'
            foreign.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'Unexpected files'):
                first.open()
            self.assertEqual(foreign.read_text(), 'preserve')
            foreign.unlink()
            first.open()
            first.close()

    def test_feature_cache_fit_cleanup_and_online_switch(self):
        for cached in (True, False):
            with self.subTest(releas=cached), tempfile.TemporaryDirectory() as directory:
                student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
                optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=.001)
                c = deepcopy(self.config)
                c['training'].update(epochs=2, max_steps=None, crop_size=32, releas=cached)
                c['benchmark']['enabled'] = False
                c['logging']['progress'] = False
                c['runtime']['output_dir'] = str(Path(directory) / 'run')
                batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
                class Metrics(nn.Module):
                    def forward(self, pred, target):
                        return {'psnr': 20., 'ssim': .8, 'lpips': .1}
                events = []
                original_forward = teacher.forward
                original_student = student.forward
                def teacher_forward(*args, **kwargs):
                    events.append('teacher')
                    return original_forward(*args, **kwargs)
                def student_forward(*args, **kwargs):
                    events.append('student')
                    return original_student(*args, **kwargs)
                cache_dir = Path(directory) / 'feats'
                with patch('tool.trainer.FEATURE_CACHE_DIR', cache_dir), \
                     patch('tool.distillation.prepare_image', identity_prepare), \
                     patch.object(teacher, 'forward', teacher_forward), \
                     patch.object(student, 'forward', student_forward), \
                     patch.object(TeacherFeatureCache, 'build', autospec=True, wraps=None,
                                  side_effect=TeacherFeatureCache.build) as build:
                    summary = fit(teacher, student, criterion, [batch], [batch], optimizer, Metrics(), c)
                expected = ['teacher', 'teacher'] + ['student'] * 4 if cached else ['teacher', 'student', 'teacher', 'student'] * 2
                self.assertEqual(events, expected)
                self.assertEqual(build.call_count, 1 if cached else 0)
                self.assertEqual(summary['global_step'], 2)
                self.assertTrue(summary['teacher_unchanged'])
                self.assertEqual(list(cache_dir.iterdir()) if cache_dir.exists() else [], [])
                self.assertEqual(next(student.parameters()).device.type, 'cpu')

    def test_feature_cache_cleanup_on_failure_and_interrupt(self):
        for error in (RuntimeError('fixture failure'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                student, teacher, criterion = tiny_model(), FixtureTeacher(), self.criterion()
                optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=.001)
                c = deepcopy(self.config)
                c['training'].update(epochs=1, max_steps=1, crop_size=32, releas=True)
                c['benchmark']['enabled'] = False
                c['logging']['progress'] = False
                c['runtime']['output_dir'] = str(Path(directory) / 'run')
                cache_dir = Path(directory) / 'feats'
                batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
                def fail(*args, **kwargs):
                    self.assertEqual(len(list(cache_dir.glob('*.pt'))), 2)
                    raise error
                with patch('tool.trainer.FEATURE_CACHE_DIR', cache_dir), \
                     patch('tool.distillation.prepare_image', identity_prepare), \
                     patch('tool.trainer.train_batch', fail), self.assertRaises(type(error)):
                    fit(teacher, student, criterion, [batch], [batch], optimizer, nn.Identity(), c)
                self.assertEqual(list(cache_dir.iterdir()), [])
                self.assertEqual(teacher.device.type, 'cpu')

    def test_cache_prompt_answers_and_noninteractive_keep(self):
        with patch('builtins.input', side_effect=['invalid', 'n']):
            self.assertFalse(ask_delete_cache(Path('/tmp/feats')))
        with patch('builtins.input', return_value=' y ') as ask:
            self.assertTrue(ask_delete_cache(Path('/tmp/feats')))
            ask.assert_called_once_with('[Done]: Delete cached features? [Y/N]: ')
        for error in (EOFError(), KeyboardInterrupt()):
            with patch('builtins.input', side_effect=error):
                self.assertFalse(ask_delete_cache(Path('/tmp/feats')))

    def test_cache_reused_across_runs_and_deleted_on_request(self):
        c = deepcopy(self.config)
        c['training'].update(epochs=1, max_steps=1, crop_size=32, releas=True)
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        teacher = FixtureTeacher()
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        class Metrics(nn.Module):
            def forward(self, pred, target):
                return {'psnr': 20., 'ssim': .8, 'lpips': .1}
        with tempfile.TemporaryDirectory() as directory, \
             patch('tool.trainer.FEATURE_CACHE_DIR', Path(directory) / 'feats'), \
             patch('tool.distillation.prepare_image', identity_prepare):
            cache_dir = Path(directory) / 'feats'
            original_forward = teacher.forward
            for run, delete in enumerate((False, False, True)):
                c['runtime']['output_dir'] = str(Path(directory) / f'run-{run}')
                student, criterion = tiny_model(), self.criterion()
                optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=.001)
                side_effect = original_forward if run == 0 else AssertionError('Teacher must not execute for valid cache')
                with patch.object(teacher, 'forward', side_effect=side_effect) as forward:
                    fit(teacher, student, criterion, [batch], [batch], optimizer, Metrics(), c,
                        cache_cleanup=lambda path: delete)
                    self.assertEqual(forward.call_count, 2 if run == 0 else 0)
                rows = [json.loads(line) for line in (Path(c['runtime']['output_dir']) / 'metrics.jsonl').read_text().splitlines()]
                self.assertEqual(next(row for row in rows if row['event'] == 'teacher_cache')['reused'], run > 0)
                self.assertEqual((cache_dir / 'manifest.json').exists(), not delete)
            self.assertEqual(list(cache_dir.iterdir()), [])

    def test_cache_manifest_invalidation(self):
        c = deepcopy(self.config)
        c['training']['crop_size'] = 32
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        teacher = FixtureTeacher()
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            fingerprint = cache.fingerprint('teacher-a', [batch], [batch], c, torch.device('cpu'))
            cache.build(teacher, [batch], [batch], c)
            cache.seal(fingerprint)
            cache.close(delete=False)
            cache.open()
            self.assertTrue(cache.reuse(fingerprint))
            self.assertFalse(cache.reuse(cache.fingerprint('teacher-b', [batch], [batch], c, torch.device('cpu'))))
            changed = deepcopy(c)
            changed['training']['crop_size'] = 64
            self.assertFalse(cache.reuse(cache.fingerprint('teacher-a', [batch], [batch], changed, torch.device('cpu'))))
            changed = deepcopy(c)
            changed['augmentation']['enabled'] = not changed['augmentation']['enabled']
            changed['training']['epochs'] += 1
            self.assertEqual(fingerprint, cache.fingerprint('teacher-a', [batch], [batch], changed, torch.device('cpu')))
            cache.path('train', 'fixture/1').unlink()
            self.assertFalse(cache.reuse(fingerprint))
            (cache.directory / 'manifest.json').write_text(json.dumps({'fingerprint': fingerprint, 'files': []}))
            self.assertFalse(cache.reuse(fingerprint))
            cache.close()

    def test_teacher_factory_releases_object_and_weights_only_in_cache_mode(self):
        import weakref
        for cached in (True, False):
            with self.subTest(releas=cached), tempfile.TemporaryDirectory() as directory:
                refs = []
                def create_teacher():
                    teacher = FixtureTeacher()
                    refs.extend([weakref.ref(teacher), weakref.ref(teacher.anchor)])
                    return teacher
                c = deepcopy(self.config)
                c['training'].update(epochs=1, max_steps=1, crop_size=32, releas=cached)
                c['benchmark']['enabled'] = False
                c['logging']['progress'] = False
                c['runtime']['output_dir'] = str(Path(directory) / 'run')
                student, criterion = tiny_model(), self.criterion()
                optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()), lr=.001)
                def check_lifetime(module, args):
                    if cached:
                        self.assertIsNone(refs[0]())
                        self.assertIsNone(refs[1]())
                    else:
                        teacher = refs[0]()
                        self.assertIsNotNone(teacher)
                        self.assertFalse(teacher.training)
                        self.assertTrue(all(not value.requires_grad for value in teacher.buffers()))
                student.register_forward_pre_hook(check_lifetime)
                class Metrics(nn.Module):
                    def forward(self, pred, target):
                        return {'psnr': 20., 'ssim': .8, 'lpips': .1}
                batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
                cache_dir = Path(directory) / 'feats'
                with patch('tool.trainer.FEATURE_CACHE_DIR', cache_dir), \
                     patch('tool.distillation.prepare_image', identity_prepare):
                    result = fit(create_teacher, student, criterion, [batch], [batch], optimizer, Metrics(), c,
                                 cache_cleanup=lambda path: False)
                self.assertEqual(result['teacher_released'], cached)
                self.assertEqual(result['teacher_blend_scale'], .5)
                checkpoint = torch.load(Path(c['runtime']['output_dir']) / 'last.pt', weights_only=True)
                self.assertEqual(checkpoint['teacher_blend_scale'], .5)
                if cached:
                    manifest = json.loads((cache_dir / 'manifest.json').read_text())
                    self.assertEqual(manifest['blend_scale'], .5)
                else:
                    self.assertFalse(cache_dir.exists())

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

    def test_parallel_cache_dataset_loading_writes_and_exact_targets(self):
        import threading
        from tool.teacher_features import cache_worker_plan
        c = deepcopy(self.config)
        c['training']['crop_size'] = 32
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        teacher = FixtureTeacher()
        images = [torch.full((3, 32, 32), value) for value in (.2, .4, .6, .8)]
        read_threads, write_threads = [], []
        class Dataset:
            def __len__(self):
                return len(images)
            def __getitem__(self, index):
                read_threads.append(threading.current_thread().name)
                return {'sample_id': f'fixture/{index}', 'image': images[index]}
        def assert_teacher_thread(module, args):
            self.assertEqual(threading.current_thread().name, 'MainThread')
        teacher.register_forward_pre_hook(assert_teacher_thread)
        original_save = torch.save
        def record_save(*args, **kwargs):
            write_threads.append(threading.current_thread().name)
            return original_save(*args, **kwargs)
        plan = {'readers': 2, 'writers': 2, 'prefetch': 2, 'pending_writes': 2}
        with tempfile.TemporaryDirectory() as directory, \
             patch('tool.distillation.prepare_image', identity_prepare), \
             patch('tool.teacher_features.cache_worker_plan', return_value=plan), \
             patch.object(torch, 'save', record_save):
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            try:
                cache.build(teacher, SimpleNamespace(dataset=Dataset()), SimpleNamespace(dataset=Dataset()), c)
                self.assertEqual(len(list(cache.directory.glob('*.pt'))), 8)
                for index, image in enumerate(images):
                    prepared = identity_prepare(image.permute(1, 2, 0).numpy())
                    head, features = teacher(prepared.features, return_features=True)
                    rgb = compose_rgb_tensor(head, prepared, blend_scale=teacher.blend_scale)
                    for split in ('train', 'val'):
                        actual_head, actual_features, actual_rgb = cache.load(split, f'fixture/{index}', 'cpu')
                        torch.testing.assert_close(actual_head, head, rtol=0, atol=0)
                        torch.testing.assert_close(actual_rgb, rgb, rtol=0, atol=0)
                        for key in features:
                            torch.testing.assert_close(actual_features[key], features[key], rtol=0, atol=0)
                self.assertTrue(any(name.startswith('cache-read') for name in read_threads))
                self.assertTrue(any(name.startswith('cache-write') for name in write_threads))
            finally:
                cache.close()

    def test_parallel_cache_writer_errors_propagate(self):
        import threading
        c = deepcopy(self.config)
        c['training']['crop_size'] = 32
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        batch = {'images': [torch.rand(3, 32, 32), torch.rand(3, 32, 32)],
                 'sample_ids': ['fixture/1', 'fixture/2'], 'scenes': ['fixture', 'fixture']}
        original_save = torch.save
        def fail_in_background(*args, **kwargs):
            if threading.current_thread().name.startswith('cache-write'):
                raise OSError('fixture disk failure')
            return original_save(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
             patch('tool.distillation.prepare_image', identity_prepare), \
             patch.object(torch, 'save', fail_in_background):
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            try:
                with self.assertRaisesRegex(OSError, 'fixture disk failure'):
                    cache.build(FixtureTeacher(), [batch], [batch], c)
                self.assertFalse((cache.directory / 'manifest.json').exists())
            finally:
                cache.close()
            self.assertEqual(list(cache.directory.iterdir()), [])

    def test_cache_worker_limits_and_bounded_prefetch(self):
        from concurrent.futures import ThreadPoolExecutor
        from tool.teacher_features import cache_worker_plan, bounded_cache_prepare
        with patch('os.sched_getaffinity', return_value={0, 1}):
            plan = cache_worker_plan(1, .001, 1, 1024, 1024, 100)
            self.assertEqual(plan['readers'], 1)
            self.assertEqual(plan['writers'], 1)
        plan = cache_worker_plan(1, .001, 1, 10**12, 10**12, 100)
        self.assertEqual(plan['prefetch'], 1)
        self.assertEqual(plan['pending_writes'], 1)
        submitted = []
        def source():
            for i in range(20):
                submitted.append(i)
                yield i
        with ThreadPoolExecutor(max_workers=2) as pool, \
             patch('tool.teacher_features.prepare_cache_item', side_effect=lambda item, crop: item):
            items = bounded_cache_prepare(pool, source(), 32, 2)
            self.assertEqual(next(items), 0)
            self.assertEqual(submitted, [0, 1])
            items.close()
            self.assertEqual(submitted, [0, 1])

    def test_partial_cache_resumes_validation_without_reextracting_train(self):
        from tool.teacher_features import prepare_cache_item
        c = deepcopy(self.config)
        c['training']['crop_size'] = 32
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        teacher = FixtureTeacher()
        train = [{'images': [torch.rand(3, 32, 32), torch.rand(3, 32, 32)],
                  'sample_ids': ['train/1', 'train/2'], 'scenes': ['fixture', 'fixture']}]
        val = [{'images': [torch.rand(3, 32, 32)], 'sample_ids': ['val/1'], 'scenes': ['fixture']}]
        def stop_before_validation(item, crop):
            if item[0] == 'val/1':
                raise KeyboardInterrupt()
            return prepare_cache_item(item, crop)
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            cache_dir = Path(directory) / 'feats'
            cache = TeacherFeatureCache(cache_dir)
            cache.open()
            fingerprint = cache.fingerprint('teacher', train, val, c, torch.device('cpu'))
            cache.prepare(fingerprint, .5)
            with patch('tool.teacher_features.prepare_cache_item', side_effect=stop_before_validation):
                with self.assertRaises(KeyboardInterrupt):
                    cache.build(teacher, train, val, c)
            saved_train = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.glob('train-*.pt')}
            self.assertEqual(len(saved_train), 2)
            cache.close(delete=False)
            cache = TeacherFeatureCache(cache_dir)
            cache.open()
            try:
                self.assertFalse(cache.reuse(fingerprint))
                cache.prepare(fingerprint, .5)
                with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
                    cache.build(teacher, train, val, c)
                    self.assertEqual(forward.call_count, 1)
                for name, expected in saved_train.items():
                    path = cache_dir / name
                    self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), expected)
                cache.seal(fingerprint, blend_scale=.5)
                self.assertTrue(cache.reuse(fingerprint))
                cache.path('train', 'train/1').write_bytes(b'truncated')
                self.assertFalse(cache.reuse(fingerprint))
                cache.prepare(fingerprint, .5)
                with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
                    cache.build(teacher, train, val, c)
                    self.assertEqual(forward.call_count, 1)
                cache.seal(fingerprint, blend_scale=.5)
                self.assertTrue(cache.reuse(fingerprint))
            finally:
                cache.close()

    def test_completed_legacy_cache_is_reused_after_pipeline_update(self):
        c = deepcopy(self.config)
        c['training']['crop_size'] = 32
        c['benchmark']['enabled'] = False
        c['logging']['progress'] = False
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['fixture/1'], 'scenes': ['fixture']}
        with tempfile.TemporaryDirectory() as directory, patch('tool.distillation.prepare_image', identity_prepare):
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            try:
                fingerprint = cache.fingerprint('teacher', [batch], [batch], c, torch.device('cpu'))
                cache.build(FixtureTeacher(), [batch], [batch], c)
                cache.seal(cache._legacy_fingerprint, blend_scale=.5)
                manifest = cache.read_manifest()
                manifest.pop('complete')
                (cache.directory / 'manifest.json').write_text(json.dumps(manifest))
                self.assertTrue(cache.reuse(fingerprint))
            finally:
                cache.close()

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
