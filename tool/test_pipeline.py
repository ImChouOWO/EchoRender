"""Regression tests for batching, target prefetch and epoch-boundary resume."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mlxdlss.features import NetworkGeometry

from model.card.loss import DistillationLoss
from .dataloader import collate_sequences
from .distillation import (StudentAugmentation, batch_augmentation, clip_groups, train_batch,
                           make_frame, cache_key, module_digest)
from .temporal import (prepare_student_frames, prepare_student_frame, student_inputs, compose_tensor,
                       truncate_tensor, truncate_half, teacher_target, image_rgb)
from .teacher_features import TeacherFeatureCache, TargetPrefetch, teacher_temporal_signature
from .training_config import load_training_config, TrainingSchedule
from .trainer import fit, save_checkpoint
from .resume import read_checkpoint, training_lock, validate_resume, load_log
from .test_training import tiny_model, FixtureTeacher
from model.student import EchoRenderStudent
from .test_temporal import sample, Metrics


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)
        self.config = deepcopy(load_training_config())
        self.config['loss'].update(rgb_lpips=0., appearance=0.)
        self.config['training']['distillation_layers'] = ['RGB composition', 'OutputHead', 'E2', 'VIT', 'D1', 'D3']
        self.config['training']['precision'] = 'float32'
        self.config['training'].update(adapter_size=32, epochs=3, releas=False, micro_batch_size=2,
            profiles=['neutral'], early_stopping={'patience': 0, 'min_delta': 0.})
        self.config['rendering']['profiles'] = {'neutral': self.config['rendering']['profiles']['neutral']}
        self.config['logging']['progress'] = self.config['benchmark']['enabled'] = False
        self.config['loss'].update(ssim_window=3, ms_ssim_weights=[1.])

    def criterion(self):
        return DistillationLoss(self.config['loss'], channels=dict.fromkeys(('E2', 'VIT', 'D1', 'D3'), 4))

    def test_half_truncation_matches_reference_all_half_values_and_boundaries(self):
        half = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        with np.errstate(invalid="ignore"):
            near = np.nextafter(half, np.float32(0))
        values = np.concatenate([half, near,
                                 np.random.default_rng(7).uniform(-100000, 100000, 5000).astype(np.float32)])
        with np.errstate(over='ignore', invalid='ignore'):
            expected = truncate_half(values)
        actual = truncate_tensor(torch.from_numpy(values)).numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.array_equal(np.signbit(actual), np.signbit(expected)))
        self.assertFalse(truncate_tensor(torch.ones(2, requires_grad=True)).requires_grad)

    def test_feat_batch_matches_serial_inputs_composition_losses_and_gradients(self):
        student = tiny_model('raft_small')
        torch.nn.init.normal_(student.motion.decoder[-1].weight, std=.005)
        other = deepcopy(student)
        requests = []
        rng = np.random.default_rng(24)
        for i, size in enumerate((32, 32, 40, 32)):
            rgb = rng.random((size, size, 3), dtype=np.float32)
            requests.append(dict(rgb=rgb, profile='neutral', previous=rgb * .9 if i else None,
                history=rgb * .8 if i else None, motion=np.zeros((size, size, 2), np.float32),
                confidence=np.full((size, size, 1), .5, np.float32)))
        with patch.object(student.motion, 'forward', wraps=student.motion.forward) as forward:
            batched = prepare_student_frames(requests, self.config, student)
            self.assertEqual(forward.call_count, 2)
        serial = [prepare_student_frame(**request, config=self.config, student=other) for request in requests]
        losses = []
        for frames in (batched, serial):
            total = 0
            for frame in frames:
                inputs = student_inputs(frame, 'cpu')
                head = inputs[..., :4] * .02
                prediction, history = compose_tensor(head, frame, .5)
                self.assertIsInstance(history, torch.Tensor)
                self.assertFalse(history.requires_grad)
                total = total + prediction.square().mean() + frame.motion_loss
            losses.append(total)
        losses[0].backward()
        losses[1].backward()
        torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=2e-6)
        for a, b in zip(batched, serial):
            torch.testing.assert_close(student_inputs(a, 'cpu'), student_inputs(b, 'cpu'), atol=1e-6, rtol=1e-6)
        for a, b in zip(student.motion.parameters(), other.motion.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-4)

    def test_clip_grouping_preserves_lengths_augmentation_noise_and_rng(self):
        samples = [sample(frames=n, name=str(i)) for i, n in enumerate((1, 5, 2, 4, 3, 6, 2, 5))]
        lengths = [len(item['images']) for item in samples]
        augmentation = StudentAugmentation(self.config)
        reference = StudentAugmentation(self.config)
        plan = {1: 'brightness', 4: 'contrast'}
        actual = batch_augmentation(samples, lengths, plan, augmentation, .25, 2)
        expected = {}
        for begin in range(0, len(samples), 2):
            for position in range(max(lengths[begin:begin + 2])):
                for index in range(begin, min(begin + 2, len(samples))):
                    if index in plan and position < lengths[index]:
                        expected[index, position] = reference.apply_rgb(image_rgb(samples[index]['images'][position]), plan[index], .25)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        torch.testing.assert_close(augmentation.generator.get_state(), reference.generator.get_state())
        groups = clip_groups(lengths, 2)
        self.assertEqual(sorted(i for group in groups for i in group), list(range(len(samples))))
        old_waves = sum(max(lengths[i:i + 2]) for i in range(0, len(samples), 2))
        self.assertLessEqual(sum(max(lengths[i] for i in group) for group in groups), old_waves)

    def test_training_uses_actual_augmented_prior_rgb_and_own_detached_history(self):
        from . import distillation
        config = deepcopy(self.config)
        config['rendering']['train_min_frames'] = 2
        student, criterion = tiny_model('raft_small'), self.criterion()
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
        samples = [sample(frames=2, name=str(i)) for i in range(4)]
        augmentation = StudentAugmentation(config)
        requests, stored = [], []
        def prepare(items, *args):
            requests.append(items)
            return prepare_student_frames(items, *args)
        original = distillation.objective
        def objective(*args, **kwargs):
            result = original(*args, **kwargs)
            stored.append(result[-1])
            return result
        controls = TrainingSchedule(config, total_steps=2).values(epoch=1, step=0)
        with patch.object(augmentation, 'plan', return_value={0: 'brightness'}), \
             patch.object(distillation, 'prepare_student_frames', side_effect=prepare), \
             patch.object(distillation, 'objective', side_effect=objective):
            result = train_batch(FixtureTeacher(), student, criterion, {'sequences': samples}, optimizer,
                config=config, controls=controls, augmentation=augmentation,
                augmentation_ratio=.25, blend_scale=.5, micro_batch_size=4)
        self.assertEqual(len(requests), 2)
        self.assertEqual(result['clip_lengths'], [2] * 4)
        clean = image_rgb(samples[0]['images'][0])
        self.assertGreater(float(np.abs(requests[0][0]['rgb'] - clean).mean()), .02)
        for index in range(4):
            before = requests[1][index]['previous']
            torch.testing.assert_close(before, torch.from_numpy(requests[0][index]['rgb']), atol=0, rtol=0)
            self.assertIs(requests[1][index]['history'], stored[index])
            self.assertFalse(stored[index].requires_grad)
        for index in range(1, 4):
            np.testing.assert_array_equal(requests[0][index]['rgb'], image_rgb(samples[index]['images'][0]))

    def test_prefetch_preserves_identity_bounds_and_load_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.directory.mkdir()
            cache.prepare('fixture', .5)
            item = sample(frames=5)
            keys = [cache_key(item, i) for i in range(5)]
            for i, key in enumerate(keys):
                cache.store('train', key, {'head': torch.ones(1, 2, 2, 4) * i,
                    'features': {'e1': torch.ones(1, 2, 2, 2)}, 'rgb': torch.ones(1, 3, 2, 2), 'history': np.zeros(1).tolist()})
            size = max(cache.path('train', key).stat().st_size for key in keys)
            with TargetPrefetch(cache, 'train', keys, 'cpu', count=3, byte_limit=size * 2) as ready:
                for key in keys:
                    actual = ready.get(key)
                    expected = cache.load_payload('train', key)
                    torch.testing.assert_close(actual['head'], expected['head'], atol=0, rtol=0)
                    self.assertEqual(actual['sample_id'], expected['sample_id'])
                    self.assertLessEqual(ready.peak_bytes, size * 2)
                self.assertEqual(ready.loads, len(keys))
            self.assertTrue(ready.executor._shutdown)
            with patch.object(cache, 'load_payload', side_effect=ValueError('broken target')):
                with self.assertRaisesRegex(ValueError, 'broken target'):
                    with TargetPrefetch(cache, 'train', keys, 'cpu') as ready:
                        ready.get(keys[0])
            self.assertTrue(ready.executor._shutdown)

    def test_current_cache_provenance_reuses_without_rewriting_targets(self):
        signature = teacher_temporal_signature()
        self.assertEqual(len(signature), 64)
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.directory.mkdir()
            payload = {'rendering': {'profiles': {'neutral': {'tone': 1}}},
                       'adapter_size': 320, 'stages': ['E2', 'VIT', 'D1', 'D3'], 'pipeline': signature}
            fingerprint = cache.configure_signatures(payload)
            cache.prepare(fingerprint, .5)
            key = cache_key(sample(frames=1), 0)
            cache.store('train', key, {'head': torch.zeros(1)})
            manifest = cache.read_manifest()
            self.assertTrue(cache.compatible_files(manifest, fingerprint))
            for key, value in (('adapter_size', 384), ('stages', ['E2']), ('pipeline', 'obsolete')):
                changed = {**payload, key: value}
                different = cache.configure_signatures(changed)
                self.assertEqual(cache.compatible_files(manifest, different), {})

    def test_cached_and_online_steps_match_and_regrouping_preserves_update(self):
        config = deepcopy(self.config)
        config['rendering']['train_min_frames'] = 10
        config['augmentation']['enabled'] = True
        samples = [sample(frames=n, name=str(i)) for i, n in enumerate((1, 3, 2, 4))]
        controls = TrainingSchedule(config, total_steps=3).values(epoch=1, step=0)
        student, criterion = tiny_model('raft_small'), self.criterion()
        models, criteria = [deepcopy(student) for _ in range(3)], [deepcopy(criterion) for _ in range(3)]
        teacher = FixtureTeacher()
        before = module_digest(teacher)
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.directory.mkdir()
            cache.prepare('fixture', .5)
            for item in samples:
                history = None
                for position in range(len(item['images'])):
                    frame = make_frame(item, position, config, history)
                    target = teacher_target(teacher, frame)
                    history = target['history']
                    cache.store('train', cache_key(item, position), {**target, 'history': torch.from_numpy(history.copy())})
            results = []
            for index, (model, loss) in enumerate(zip(models, criteria)):
                optimizer = torch.optim.SGD([*model.parameters(), *loss.parameters()], lr=.001)
                context = patch('tool.distillation.clip_groups', lambda lengths, maximum:
                    [list(range(i, min(i + maximum, len(lengths)))) for i in range(0, len(lengths), maximum)])
                if index != 2:
                    from contextlib import nullcontext
                    context = nullcontext()
                with context:
                    results.append(train_batch(teacher if index != 1 else None, model, loss, {'sequences': samples},
                        optimizer, config=config, controls=controls, augmentation=StudentAugmentation(config),
                        augmentation_ratio=.25, target_cache=cache if index == 1 else None,
                        blend_scale=.5, micro_batch_size=2))
            self.assertEqual(module_digest(teacher), before)
            for index in (1, 2):
                self.assertAlmostEqual(results[0]['loss'], results[index]['loss'], places=5)
                self.assertEqual(results[0]['clip_lengths'], results[index]['clip_lengths'])
                for a, b in zip(models[0].parameters(), models[index].parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=3e-7)

    def run_fit(self, directory, *, interrupt=False, resume=None):
        config = deepcopy(self.config)
        config['runtime']['output_dir'] = str(directory)
        config['training']['optimizer']['momentum'] = .9
        samples = [sample(frames=3, name=str(i)) for i in range(4)]
        loaders = [DataLoader(samples, batch_size=4, shuffle=True, collate_fn=collate_sequences,
                             generator=torch.Generator().manual_seed(4)) for _ in range(2)]
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        student, criterion = tiny_model('raft_small'), self.criterion()
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], **config['training']['optimizer'])
        def save(*args, **kwargs):
            save_checkpoint(*args, **kwargs)
            if args[0].name == 'last.pt' and kwargs['epoch'] == (interrupt if type(interrupt) is int else 1):
                raise KeyboardInterrupt('test interruption')
        with redirect_stdout(io.StringIO()), patch('tool.trainer.save_checkpoint', save if interrupt else save_checkpoint):
            return fit(FixtureTeacher, student, criterion, *loaders, optimizer, Metrics(), config,
                       device='cpu', resume=resume)

    def test_resume_matches_uninterrupted_weights_optimizer_rng_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.run_fit(root / 'full')
            with self.assertRaises(KeyboardInterrupt):
                self.run_fit(root / 'resume', interrupt=True)
            first = read_checkpoint(root / 'resume/last.pt')
            actual = self.run_fit(root / 'resume', resume=first)
            a, b = [read_checkpoint(root / name / 'last.pt') for name in ('full', 'resume')]
            for group in ('student', 'distillation'):
                for key in a[group]:
                    torch.testing.assert_close(a[group][key], b[group][key], atol=0, rtol=0)
            for key, state in a['optimizer']['state'].items():
                torch.testing.assert_close(state['momentum_buffer'], b['optimizer']['state'][key]['momentum_buffer'], atol=0, rtol=0)
            self.assertEqual(a['global_step'], b['global_step'])
            self.assertEqual(a['schedule_controls'], b['schedule_controls'])
            self.assertEqual(expected['history'], actual['history'])
            self.assertEqual(expected['best_epoch'], actual['best_epoch'])
            self.assertEqual(a['resume_state']['early'], b['resume_state']['early'])
            torch.testing.assert_close(a['resume_state']['rng']['augmentation'], b['resume_state']['rng']['augmentation'])
            rows = [json.loads(line) for line in (root / 'resume/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in rows if row.get('event') == 'epoch'], [1, 2, 3])
            self.assertTrue(next(row for row in rows if row.get('event') == 'resume')['rng_restored'])
            self.assertEqual(set(path.name for path in (root / 'resume').glob('*.pt')), {'best.pt', 'last.pt'})

    def test_incomplete_resume_without_rng_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                self.run_fit(output, interrupt=True)
            payload = read_checkpoint(output / 'last.pt')
            payload.pop('resume_state')
            with self.assertRaisesRegex(ValueError, 'complete checkpoint'):
                self.run_fit(output, resume=payload)

    def test_run_lock_prevents_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with training_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, 'already in use'):
                    with training_lock(Path(directory)):
                        self.fail('second writer admitted')

    def test_resume_log_discards_partial_epoch_and_accepts_infinite_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metrics.jsonl').write_text('{"event":"epoch","epoch":1,"psnr":"Infinity"}\n'
                '{"event":"train","epoch":2,"step":2}\n{"event":')
            rows = load_log(root, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['psnr'], float('inf'))
            (root / 'metrics.jsonl').write_text('{bad}\n{"event":"epoch","epoch":1}\n')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                load_log(root, 1)

    def test_resume_retains_early_stopping_counter(self):
        self.config['training'].update(epochs=5, early_stopping={'patience': 2, 'min_delta': 1e9})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.run_fit(root / 'full')
            with self.assertRaises(KeyboardInterrupt):
                self.run_fit(root / 'resume', interrupt=2)
            payload = read_checkpoint(root / 'resume/last.pt')
            self.assertEqual(payload['resume_state']['early']['bad_epochs'], 1)
            resumed = self.run_fit(root / 'resume', resume=payload)
            self.assertEqual(resumed['status'], 'early_stopped')
            self.assertEqual(resumed['epochs_run'], full['epochs_run'])
            self.assertEqual(resumed['epochs_run'], 3)
            self.assertEqual(resumed['history'], full['history'])
class AdapterDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_root_adapter_keeps_full_field_and_stage_projections_do_not_resize(self):
        from model.card.card import InputAdapter
        student = tiny_model()
        x = torch.rand(1, 48, 96, 16)
        x[:, :, :16, :3] = 1
        observed = []
        hook = student.layers[0].proj.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
        head, features = student(x, return_features=True)
        hook.remove()
        expected = InputAdapter.resize(x.permute(0, 3, 1, 2), 32)
        torch.testing.assert_close(observed[0], expected, atol=0, rtol=0)
        self.assertEqual(tuple(head.shape), (1, 32, 32, 4))
        self.assertTrue(torch.all(observed[0][:, :3, :, :4] > .99))
        self.assertTrue(all(module.size is None for module in student.modules()
                            if isinstance(module, InputAdapter) and module is not student.layers[0]))
        self.assertEqual(set(features), {'E2', 'VIT', 'D1', 'D3'})

    def test_one_selected_stage_controls_capture_projection_and_loss_average(self):
        student = EchoRenderStudent(tiny_model().config, distillation_layers=['RGB composition', 'D3'])
        criterion = DistillationLoss({**load_training_config()['loss'], 'rgb_lpips': 0.}, channels={'D3': 4}, stages=['RGB composition', 'D3'])
        head, features = student(torch.rand(1, 32, 32, 16), return_features=True)
        self.assertEqual(set(features), {'D3'})
        self.assertEqual(set(criterion.projections), {'D3'})
        rgb = torch.rand(1, 3, 192, 192)
        target_features = {'D3': torch.rand(1, 4, 4, 4)}
        _, parts = criterion(head, torch.zeros_like(head), features, target_features, rgb, rgb,
                            image_hw=(32, 32), network_hw=(32, 32), clean_source=rgb)
        self.assertAlmostEqual(parts['mid'], criterion.config['feature_weights']['D3'] * parts['D3'], places=7)
        self.assertNotIn('E1', parts)
        self.assertNotIn('vit_feature', parts)
        for stages in (['D3', 'D3'], ['unknown'], []):
            with self.assertRaises(ValueError):
                EchoRenderStudent(tiny_model().config, distillation_layers=stages)

    def test_config_rejects_retired_crop_invalid_adapter_and_weight_controls(self):
        import yaml
        original = load_training_config()
        original.pop('config_path')
        cases = [('training', 'crop_size', 0), ('training', 'adapter_size', 192),
                 ('training', 'adapter_size', True), ('training', 'distillation_layers', ['E2', 'E2']),
                 ('training', 'distillation_layers', ['e2']), ('training', 'distillation_layers', []),
                 ('loss', 'teacher_change', {'alpha': -1, 'tau': .1}),
                 ('loss', 'teacher_change', {'alpha': 1, 'tau': 0})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for section, key, value in cases:
                config = deepcopy(original)
                config[section][key] = value
                path.write_text(yaml.safe_dump(config))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_training_config(path)

    def test_teacher_change_uses_clean_reference_and_detached_bounded_weights(self):
        criterion = DistillationLoss({**load_training_config()['loss'], 'rgb_lpips': 0.})
        teacher = torch.zeros(1, 3, 8, 8, requires_grad=True)
        clean = torch.zeros_like(teacher, requires_grad=True)
        with torch.no_grad():
            teacher[:, :, :, :4] = .2
        weight = criterion.importance(teacher, clean)
        self.assertFalse(weight.requires_grad)
        self.assertEqual(float(weight.min()), 1)
        maximum = 1 + criterion.config['teacher_change']['alpha']
        self.assertEqual(float(weight.max()), maximum)
        prediction = torch.zeros_like(teacher, requires_grad=True)
        error = (prediction - teacher.detach()).square()
        actual = criterion.weighted_mean(error, weight)
        expected = .04 * maximum / (maximum + 1)
        self.assertAlmostEqual(float(actual.detach()), expected, places=7)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(clean.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_zero_importance_strength_matches_official_structural_metrics(self):
        from pytorch_msssim import ssim, ms_ssim
        config = {**load_training_config()['loss'], 'rgb_lpips': 0.}
        config['teacher_change']['alpha'] = 0
        criterion = DistillationLoss(config)
        torch.manual_seed(17)
        for height, width in ((192, 192), (193, 225)):
            source = torch.rand(2, 3, height, width)
            prediction = (source * .95 + .02).clamp(0, 1)
            terms = criterion.image_terms(prediction, source, torch.zeros_like(source))
            torch.testing.assert_close(terms['rgb_mse'], F.mse_loss(prediction, source))
            torch.testing.assert_close(terms['rgb_ssim'], 1 - ssim(prediction, source, data_range=1))
            torch.testing.assert_close(terms['rgb_ms_ssim'], 1 - ms_ssim(prediction, source, data_range=1))

    def test_local_structural_scores_prioritize_teacher_changed_error_regions(self):
        criterion = DistillationLoss({**load_training_config()['loss'], 'rgb_lpips': 0.})
        target = torch.full((1, 3, 192, 192), .5)
        prediction = target.clone()
        prediction[..., :96] = .3
        clean_error = target.clone()
        clean_elsewhere = target.clone()
        clean_error[..., :96] = .3
        clean_elsewhere[..., 96:] = .3
        emphasized = criterion.image_terms(prediction, target, clean_error)
        other = criterion.image_terms(prediction, target, clean_elsewhere)
        for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim'):
            self.assertGreater(float(emphasized[name]), float(other[name]))

    def test_cached_targets_retain_full_unaugmented_reference(self):
        config = load_training_config()
        config['training']['adapter_size'] = 32
        item = sample(frames=1, size=48)
        target = teacher_target(FixtureTeacher(), make_frame(item, 0, config))
        torch.testing.assert_close(target['source_rgb'][0], item['images'][0], atol=0, rtol=0)
        self.assertEqual(tuple(target['head'].shape), (1, 32, 32, 4))
        self.assertEqual(tuple(target['rgb'].shape), (1, 3, 48, 48))


class HeadPerceptualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        config = load_training_config()
        config['loss'].update(ssim_window=3, ms_ssim_weights=[1.], rgb_lpips=0., appearance=0.)
        return config

    def head_inputs(self):
        head = torch.ones(1, 64, 64, 4, requires_grad=True)
        with torch.no_grad():
            head[..., 3] = 2
        teacher_head = torch.zeros_like(head, requires_grad=True)
        features = {'D3': torch.rand(1, 4, 8, 8, requires_grad=True)}
        teacher_features = {'D3': torch.rand(1, 4, 8, 8, requires_grad=True)}
        rgb = torch.rand(1, 3, 64, 64, requires_grad=True)
        teacher_rgb = torch.rand_like(rgb, requires_grad=True)
        return head, teacher_head, features, teacher_features, rgb, teacher_rgb

    def test_raw_rgb_and_gate_weights_control_separate_gradients(self):
        config = self.config()['loss']
        config.update(distillation=1., mid=0., output=1., reconstruction=0., output_rgb=.75, output_gate=.05)
        criterion = DistillationLoss(config, channels={'D3': 4}, stages=['RGB composition', 'OutputHead', 'D3'])
        arguments = self.head_inputs()
        total, terms = criterion(*arguments, image_hw=(64, 64), network_hw=(64, 64), clean_source=arguments[-1].detach())
        self.assertAlmostEqual(terms['output_rgb'], 1.)
        self.assertAlmostEqual(terms['output_gate'], 4.)
        self.assertAlmostEqual(total.item(), .95, places=6)
        self.assertAlmostEqual(terms['weighted_output_rgb'], .75, places=6)
        self.assertAlmostEqual(terms['weighted_output_gate'], .2, places=6)
        self.assertAlmostEqual(sum(v for k, v in terms.items() if k.startswith('weighted_')), total.item(), places=6)
        total.backward()
        torch.testing.assert_close(arguments[0].grad[..., :3], torch.full_like(arguments[0][..., :3], 1.5 / (64 * 64 * 3)))
        torch.testing.assert_close(arguments[0].grad[..., 3], torch.full_like(arguments[0][..., 3], .2 / (64 * 64)))
        self.assertIsNone(arguments[1].grad)
        self.assertIsNone(arguments[-1].grad)
        criterion.config['output_gate'] = 0.
        arguments = self.head_inputs()
        total, _ = criterion(*arguments, image_hw=(64, 64), network_hw=(64, 64), clean_source=arguments[-1].detach())
        total.backward()
        self.assertEqual(float(arguments[0].grad[..., 3].abs().sum()), 0.)
        self.assertGreater(float(arguments[0].grad[..., :3].abs().sum()), 0.)

    def test_zero_lpips_weight_does_not_load_network(self):
        with patch('tool.metrics.pretrained_lpips', side_effect=AssertionError('unexpected model load')):
            criterion = DistillationLoss(self.config()['loss'])
            rgb = torch.rand(1, 3, 32, 32, requires_grad=True)
            self.assertIsNone(criterion.perceptual)
            self.assertEqual(float(criterion.image_terms(rgb, rgb, rgb)['rgb_lpips']), 0.)

    def test_loss_config_rejects_invalid_head_weights_and_lpips(self):
        import yaml
        original = load_training_config()
        original.pop('config_path')
        for key, value in (('output_rgb', -1), ('output_gate', True), ('rgb_lpips', -0.1), ('lpips_net', 'random')):
            config = deepcopy(original)
            config['loss'][key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(config))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_training_config(path)
        self.assertIn('rgb_lpips', TrainingSchedule(original, total_steps=2).values(epoch=1, step=0)['loss_weights'])

    def test_real_pretrained_lpips_gradient_freeze_mode_and_checkpoint(self):
        import os
        from tool.metrics import pretrained_lpips
        from .training_config import PROJECT_ROOT
        from .trainer import save_checkpoint
        from .resume import read_checkpoint
        os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.cache/torch'))
        config = self.config()
        config['loss'].update(distillation=0., reconstruction=2., rgb_mse=0., rgb_ssim=0., rgb_ms_ssim=0., rgb_lpips=.05)
        criterion = DistillationLoss(config['loss'], channels={'D3': 4}, stages=['RGB composition', 'D3']).train()
        self.assertFalse(criterion.perceptual.training)
        self.assertTrue(all(not p.requires_grad for p in criterion.perceptual.parameters()))
        arguments = self.head_inputs()
        before = {k: v.clone() for k, v in criterion.perceptual.state_dict().items()}
        loss, terms = criterion(*arguments, image_hw=(64, 64), network_hw=(64, 64), clean_source=arguments[-1].detach())
        with torch.no_grad():
            expected = criterion.perceptual(arguments[-2] * 2 - 1, arguments[-1] * 2 - 1).mean()
        torch.testing.assert_close(loss.detach(), .1 * expected)
        self.assertAlmostEqual(terms['weighted_rgb_lpips'], loss.item(), places=7)
        loss.backward()
        self.assertGreater(float(arguments[-2].grad.abs().sum()), 0.)
        self.assertIsNone(arguments[-1].grad)
        self.assertTrue(all(p.grad is None for p in criterion.perceptual.parameters()))
        for name, value in criterion.perceptual.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        student = tiny_model()
        trainable = [p for module in (student, criterion) for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=.001)
        perceptual_ids = {id(p) for p in criterion.perceptual.parameters()}
        self.assertFalse(any(id(p) in perceptual_ids for group in optimizer.param_groups for p in group['params']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_checkpoint(path, student, criterion, optimizer, epoch=1, step=1, config=config,
                            controls={}, blend_scale=.5)
            payload = read_checkpoint(path)
            restored = DistillationLoss(config['loss'], channels={'D3': 4}, stages=['RGB composition', 'D3'])
            restored.load_state_dict(payload['distillation'], strict=True)
            torch.testing.assert_close(restored.perceptual_loss(arguments[-2], arguments[-1]), expected)
            self.assertFalse(restored.perceptual.training)
            self.assertTrue(all(not p.requires_grad for p in restored.perceptual.parameters()))
        with self.assertRaisesRegex(ValueError, 'at least 64'):
            criterion.perceptual_loss(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))

    def test_real_lpips_student_optimizer_update_and_loss_logging(self):
        import os
        from .training_config import PROJECT_ROOT
        os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.cache/torch'))
        config = self.config()
        config['loss']['rgb_lpips'] = .05
        config['training'].update(adapter_size=64, distillation_layers=['RGB composition', 'D3'])
        config['rendering']['train_min_frames'] = 2
        student = tiny_model('raft_small', adapter_size=64, distillation_layers=['RGB composition', 'D3'])
        criterion = DistillationLoss(config['loss'], stages=['RGB composition', 'D3'], channels={'D3': 4})
        parameters = [p for module in (student, criterion) for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, **config['training']['optimizer'])
        teacher = FixtureTeacher()
        class SelectedTeacher(FixtureTeacher):
            def forward(self, x, return_features=False):
                result = super().forward(x, return_features=return_features)
                return (result[0], {'D3': result[1]['D3']}) if return_features else result
        teacher = SelectedTeacher()
        samples = [sample(frames=2, size=64, name=str(i)) for i in range(2)]
        before = {n: v.clone() for n, v in student.named_parameters()}
        controls = TrainingSchedule(config, total_steps=2).values(epoch=1, step=0)
        result = train_batch(teacher, student, criterion, {'sequences': samples}, optimizer,
                             config=config, controls=controls, blend_scale=.5, micro_batch_size=2)
        self.assertTrue(any(not torch.equal(v, before[n]) for n, v in student.named_parameters()))
        self.assertGreater(result['components']['rgb_lpips'], 0.)
        self.assertGreater(result['components']['weighted_rgb_lpips'], 0.)
        self.assertAlmostEqual(result['loss'], sum(v for k, v in result['components'].items() if k.startswith('weighted_')), places=6)
        self.assertFalse(criterion.perceptual.training)
        self.assertTrue(all(p.grad is None for p in criterion.perceptual.parameters()))

    def test_real_lpips_resume_matches_uninterrupted_training(self):
        import os
        from .training_config import PROJECT_ROOT
        os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.cache/torch'))
        def run(directory, *, interrupt=False, resume=None):
            config = self.config()
            config['loss']['rgb_lpips'] = .05
            config['runtime']['output_dir'] = str(directory)
            config['training'].update(adapter_size=64, distillation_layers=['RGB composition', 'D3'], profiles=['neutral'],
                epochs=2, micro_batch_size=2, releas=False, early_stopping={'patience': 0, 'min_delta': 0.})
            config['rendering']['profiles'] = {'neutral': config['rendering']['profiles']['neutral']}
            config['rendering']['train_min_frames'] = 2
            config['logging']['progress'] = config['benchmark']['enabled'] = False
            random.seed(77)
            np.random.seed(77)
            torch.manual_seed(77)
            student = tiny_model('raft_small', adapter_size=64, distillation_layers=['RGB composition', 'D3'])
            criterion = DistillationLoss(config['loss'], stages=['RGB composition', 'D3'], channels={'D3': 4})
            parameters = [p for module in (student, criterion) for p in module.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(parameters, **config['training']['optimizer'])
            items = [sample(frames=2, size=64, name=str(i)) for i in range(2)]
            loaders = [DataLoader(items, batch_size=2, shuffle=True, collate_fn=collate_sequences,
                                 generator=torch.Generator().manual_seed(4)) for _ in range(2)]
            class SelectedTeacher(FixtureTeacher):
                def forward(self, x, return_features=False):
                    result = super().forward(x, return_features=return_features)
                    return (result[0], {'D3': result[1]['D3']}) if return_features else result
            def save(*args, **kwargs):
                save_checkpoint(*args, **kwargs)
                if args[0].name == 'last.pt' and kwargs['epoch'] == 1:
                    raise KeyboardInterrupt('test interruption')
            with redirect_stdout(io.StringIO()), patch('tool.trainer.save_checkpoint', save if interrupt else save_checkpoint):
                return fit(SelectedTeacher, student, criterion, *loaders, optimizer, Metrics(), config,
                           device='cpu', resume=resume)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(root / 'full')
            with self.assertRaises(KeyboardInterrupt):
                run(root / 'resume', interrupt=True)
            first = read_checkpoint(root / 'resume/last.pt')
            run(root / 'resume', resume=first)
            full = read_checkpoint(root / 'full/last.pt')
            restored = read_checkpoint(root / 'resume/last.pt')
            for group in ('student', 'distillation'):
                for name in full[group]:
                    torch.testing.assert_close(full[group][name], restored[group][name], rtol=0, atol=0)
            self.assertEqual(full['global_step'], restored['global_step'])
            self.assertEqual(full['resume_state']['early'], restored['resume_state']['early'])
            self.assertEqual(full['resume_state']['history'], restored['resume_state']['history'])
