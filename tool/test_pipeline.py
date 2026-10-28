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
from torch.utils.data import DataLoader
from mlxdlss.features import NetworkGeometry

from model.card.loss import DistillationLoss
from .dataloader import collate_sequences
from .distillation import (StudentAugmentation, batch_augmentation, clip_groups, train_batch,
                           make_frame, cache_key, module_digest)
from .temporal import (prepare_student_frames, prepare_student_frame, student_inputs, compose_tensor,
                       truncate_tensor, truncate_half, teacher_target, crop_rgb)
from .teacher_features import TeacherFeatureCache, TargetPrefetch, teacher_temporal_signature
from .training_config import load_training_config, TrainingSchedule
from .trainer import fit, save_checkpoint
from .resume import read_checkpoint, training_lock, validate_resume, load_log
from .test_training import tiny_model, FixtureTeacher
from .test_temporal import sample, Metrics


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)
        self.config = deepcopy(load_training_config())
        self.config['training'].update(crop_size=0, epochs=3, releas=False, micro_batch_size=2,
            profiles=['neutral'], early_stopping={'patience': 0, 'min_delta': 0.})
        self.config['rendering']['profiles'] = {'neutral': self.config['rendering']['profiles']['neutral']}
        self.config['logging']['progress'] = self.config['benchmark']['enabled'] = False
        self.config['loss'].update(ssim_window=3, ms_ssim_weights=[1.])
        self.geometry = patch('tool.temporal.network_geometry', lambda w, h, minimum: NetworkGeometry.identity(w, h))
        self.geometry.start()
        self.addCleanup(self.geometry.stop)

    def criterion(self):
        return DistillationLoss(self.config['loss'], channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))

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
        student = tiny_model('feat')
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
        actual = batch_augmentation(samples, lengths, plan, augmentation, .25, 0, 2)
        expected = {}
        for begin in range(0, len(samples), 2):
            for position in range(max(lengths[begin:begin + 2])):
                for index in range(begin, min(begin + 2, len(samples))):
                    if index in plan and position < lengths[index]:
                        expected[index, position] = reference.apply_rgb(crop_rgb(samples[index]['images'][position], 0), plan[index], .25)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        torch.testing.assert_close(augmentation.generator.get_state(), reference.generator.get_state())
        groups = clip_groups(lengths, 2)
        self.assertEqual(sorted(i for group in groups for i in group), list(range(len(samples))))
        old_waves = sum(max(lengths[i:i + 2]) for i in range(0, len(samples), 2))
        self.assertLessEqual(sum(max(lengths[i] for i in group) for group in groups), old_waves)

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
        signature, previous = teacher_temporal_signature()
        self.assertEqual(signature, 'aa88a5a513a76f83d1e33cf293330a4dd1fd664ee9aaaeaafdeb792834a4ebb3')
        self.assertIsNotNone(previous)
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.directory.mkdir()
            cache.prepare('fixture', .5)
            payload = {'rendering': {'profiles': {'neutral': {'tone': 1}}}, 'source': {'tool.temporal': signature}}
            alternate = {**payload, 'source': {'tool.temporal': previous}}
            old = TeacherFeatureCache(cache.directory)
            old_hash = old.configure_signatures(alternate)
            fingerprint = cache.configure_signatures(payload, alternate)
            item = sample(frames=1)
            key = cache_key(item, 0)
            cache.store('train', key, {'head': torch.zeros(1)})
            path = cache.path('train', key)
            name = path.relative_to(cache.directory).as_posix()
            manifest = {'fingerprint': old_hash, 'profile_signatures': old.profile_signatures,
                        'files': {name: path.stat().st_size}}
            modified = path.stat().st_mtime_ns
            self.assertEqual(cache.compatible_files(manifest, fingerprint), manifest['files'])
            self.assertEqual(path.stat().st_mtime_ns, modified)
            changed = deepcopy(payload)
            changed['rendering']['profiles']['neutral']['tone'] = 2
            different = cache.configure_signatures(changed)
            self.assertEqual(cache.compatible_files(manifest, different), {})

    def test_cached_and_online_steps_match_and_regrouping_preserves_update(self):
        config = deepcopy(self.config)
        config['rendering']['train_min_frames'] = 10
        config['augmentation']['enabled'] = True
        samples = [sample(frames=n, name=str(i)) for i, n in enumerate((1, 3, 2, 4))]
        controls = TrainingSchedule(config, total_steps=3).values(epoch=1, step=0)
        student, criterion = tiny_model('feat'), self.criterion()
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
                        optimizer, config=config, crop_size=0, controls=controls, augmentation=StudentAugmentation(config),
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
        student, criterion = tiny_model('feat'), self.criterion()
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

    def test_current_checkpoint_without_rng_restores_completed_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with self.assertRaises(KeyboardInterrupt):
                self.run_fit(output, interrupt=True)
            payload = read_checkpoint(output / 'last.pt')
            payload.pop('resume_state')
            result = self.run_fit(output, resume=payload)
            self.assertEqual(result['global_step'], 3)
            self.assertEqual([row['epoch'] for row in result['history']], [1, 2, 3])
            rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            self.assertFalse(next(row for row in rows if row.get('event') == 'resume')['rng_restored'])

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
