"""Sequence/control/cache integration tests; fixtures do not claim image quality."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .training_config import load_training_config, TrainingSchedule
from .dataloader import load_data_config, build_dataloader
from .temporal import (prepare_student_frame, prepare_frame, student_inputs, compose_tensor, teacher_target, crop_rgb,
                       compose_output, frame_guides, write_stage_report, FORMAT)
from .distillation import StudentAugmentation, train_batch, evaluate, cache_key, make_frame, module_digest
from .teacher_features import TeacherFeatureCache
from .test_training import tiny_model, FixtureTeacher
from .test_teacher_infer import write_flow
from model.card.loss import DistillationLoss
from mlxdlss.features import NetworkGeometry


def sample(profile='neutral', frames=3, size=32, name='demo'):
    generator = torch.Generator().manual_seed(25)
    rgb = torch.rand(3, size, size, generator=generator) * .5 + .2
    return {'images': [rgb.clone() + i * .001 for i in range(frames)],
        'sample_ids': [f'training/clean/{name}/frame_{i + 1:04}' for i in range(frames)],
        'clip_id': f'training/clean/{name}/frame_0001', 'scene': name, 'profile': profile,
        'motions': [None] + [np.zeros((size, size, 2), np.float32) for _ in range(frames - 1)],
        'confidences': [None] + [np.ones((size, size, 1), np.float32) for _ in range(frames - 1)]}


class Metrics:
    def __call__(self, prediction, target):
        mse = float((prediction - target).square().mean())
        return {'psnr': -10 * np.log10(max(mse, 1e-12)), 'ssim': 1 - mse,
                'lpips': float((prediction - target).abs().mean())}
    def per_image(self, prediction, target):
        return [self(prediction[i:i + 1], target[i:i + 1]) for i in range(len(prediction))]
    def eval(self):
        return self
    def requires_grad_(self, value):
        return self
    def cpu(self):
        return self
    def to(self, device):
        return self


class Monitor:
    def sample(self):
        return None
    def display(self):
        return 'N/A'


class ClipDataset:
    def __init__(self, root, profiles=('standard', 'neutral')):
        self.root, self.source, self.render_pass = Path(root), 'training', 'clean'
        self.profiles = profiles
        self.clips = [('demo', tuple(Path(root) / f'frame_{i + 1:04}.png' for i in range(3)))]
        self.items = [(0, p) for p in profiles]
    def __len__(self):
        return len(self.items)
    def metadata(self, index):
        return {k: v for k, v in sample(profile=self.items[index][1]).items()
                if k not in ('images', 'motions', 'confidences')}
    def __getitem__(self, index):
        return sample(profile=self.items[index][1])


class Loader:
    batch_size, num_workers = 2, 0
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield {'sequences': [self.dataset[i] for i in range(len(self.dataset))]}


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)
        self.config = deepcopy(load_training_config())
        self.config['training'].update(crop_size=0, epochs=4, releas=False)
        self.config['logging']['progress'] = False
        self.config['benchmark']['enabled'] = False
        self.config['loss'].update(ssim_window=3, ms_ssim_weights=[1.])
        self.geometry = patch('tool.temporal.network_geometry', lambda w, h, minimum: NetworkGeometry.identity(w, h))
        self.geometry.start()
        self.addCleanup(self.geometry.stop)

    def criterion(self):
        return DistillationLoss(self.config['loss'], channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))

    def test_student_input_keeps_rgb_history_style_tone_and_rejects_teacher_layout(self):
        student = tiny_model().eval()
        rgb = crop_rgb(sample()['images'][0], 0)
        outputs, inputs = [], []
        for profile in self.config['rendering']['profiles']:
            teacher_frame = prepare_frame(rgb, self.config, profile)
            with patch('tool.temporal.make_features', side_effect=AssertionError('Teacher preprocessing used by Student')):
                frame = prepare_student_frame(rgb, self.config, profile, student)
            x = student_inputs(frame, 'cpu')
            self.assertEqual(x.shape[-1], 8)
            self.assertEqual(teacher_frame.inputs.shape[-1], 16)
            torch.testing.assert_close(x, teacher_frame.inputs[..., 4:12], atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'Student frame'):
                student_inputs(teacher_frame, 'cpu')
            outputs.append(student(x))
            inputs.append(x)
        self.assertTrue(all(not torch.equal(inputs[0], value) for value in inputs[1:]))
        self.assertGreater(float((outputs[0] - outputs[-1]).detach().abs().max()), 1e-6)
        changed = x.clone()
        changed[..., 3:6] += .01
        self.assertFalse(torch.equal(student(changed), outputs[-1]))
        self.assertFalse(any('control_layers' in key or 'history_projection' in key for key in student.state_dict()))
        with self.assertRaises(ValueError):
            student(torch.zeros(1, 32, 32, 3))
        with self.assertRaises(TypeError):
            student(x, controls=torch.zeros(1, 8))
        bchw = deepcopy(student.config)
        bchw['input_layout'] = 'BCHW'
        other = type(student)(bchw).eval()
        other.load_state_dict(student.state_dict(), strict=True)
        torch.testing.assert_close(other(student_inputs(frame, 'cpu', 'BCHW')).permute(0, 2, 3, 1), outputs[-1])

    def test_student_retained_channels_match_teacher_with_real_padding(self):
        from teacher_infer import network_geometry
        student = tiny_model('feat')
        rgb = np.random.default_rng(3).random((37, 53, 3), dtype=np.float32)
        with patch('tool.temporal.network_geometry', network_geometry):
            for profile in self.config['rendering']['profiles']:
                teacher = prepare_frame(rgb, self.config, profile)
                frame = prepare_student_frame(rgb, self.config, profile, student)
                x = student_inputs(frame, 'cpu')
                self.assertEqual(tuple(x.shape), (1, 320, 320, 8))
                torch.testing.assert_close(x, teacher.inputs[..., 4:12], atol=0, rtol=0)
        cinematic = prepare_student_frame(rgb, self.config, 'cinematic', student)
        neutral = prepare_student_frame(rgb, self.config, 'neutral', student)
        torch.testing.assert_close(cinematic.inputs[..., :7], neutral.inputs[..., :7], atol=0, rtol=0)
        self.assertTrue(torch.all(cinematic.inputs[..., 7] == 1))
        self.assertTrue(torch.all(neutral.inputs[..., 7] == 0))

    def test_predicted_flow_aligns_history_before_input_and_rejects_invalid_pixels(self):
        from mlxdlss.features import scaled_color
        student = tiny_model('feat')
        rgb = np.full((32, 32, 3), .2, np.float32)
        history = np.broadcast_to(np.arange(32, dtype=np.float32)[None, :, None] / 32, rgb.shape).copy()
        field = torch.zeros(1, 2, 32, 32)
        field[:, 0] = 1 / 32
        logit = torch.full((1, 1, 32, 32), float(np.log(4)))
        with patch.object(student.motion, 'forward', return_value=(field, logit)):
            frame = prepare_student_frame(rgb, self.config, 'neutral', student, previous=rgb, history=history)
        expected = rgb + .8 * (np.concatenate((history[:, 1:], history[:, -1:]), axis=1) - rgb)
        expected[:, -1] = rgb[:, -1]
        x = student_inputs(frame, 'cpu')
        torch.testing.assert_close(x[0, ..., :3], torch.from_numpy(scaled_color(rgb)), atol=0, rtol=0)
        torch.testing.assert_close(x[0, ..., 3:6], torch.from_numpy(scaled_color(expected)), atol=0, rtol=0)
        self.assertGreater(float((x[..., :3] - x[..., 3:6]).abs().sum()), 0)

    def test_selected_profiles_and_early_stopping_config_are_validated(self):
        import yaml
        from .training_config import selected_data_config
        config = deepcopy(self.config)
        config['training']['profiles'] = ['neutral', 'standard']
        self.assertEqual(selected_data_config(config).profiles, ('neutral', 'standard'))
        for key, value in (('profiles', []), ('profiles', ['bad']), ('profiles', ['neutral', 'neutral']),
                           ('profiles', 'neutral'), ('early_stopping', {'patience': -1, 'min_delta': 0}),
                           ('early_stopping', {'patience': True, 'min_delta': 0}),
                           ('early_stopping', {'patience': 1, 'min_delta': -1})):
            changed = deepcopy(config)
            changed.pop('config_path')
            changed['training'][key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(changed))
                with self.assertRaises(ValueError):
                    load_training_config(path)

    def test_profile_cache_adds_only_missing_modes_and_preserves_existing_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            teacher = FixtureTeacher()
            digest = module_digest(teacher)
            config = deepcopy(self.config)
            try:
                one = Loader(ClipDataset(directory, profiles=('standard',)))
                fingerprint = cache.fingerprint(digest, one, one, config, torch.device('cpu'))
                cache.prepare(fingerprint, .5)
                cache.build(teacher, one, one, config)
                cache.seal(fingerprint)
                original = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in cache.directory.rglob('targets.pt')}
                two = Loader(ClipDataset(directory, profiles=('standard', 'neutral')))
                expanded = cache.fingerprint(digest, two, two, config, torch.device('cpu'))
                self.assertEqual(expanded, fingerprint)
                self.assertFalse(cache.reuse(expanded))
                cache.prepare(expanded, .5)
                with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
                    cache.build(teacher, two, two, config)
                    self.assertEqual(forward.call_count, 6)
                cache.seal(expanded)
                for path, (modified, data) in original.items():
                    self.assertEqual(path.stat().st_mtime_ns, modified)
                    self.assertEqual(path.read_bytes(), data)
                self.assertEqual(len(list(cache.directory.rglob('targets.pt'))), 12)
                cache.fingerprint(digest, one, one, config, torch.device('cpu'))
                self.assertTrue(cache.reuse(fingerprint))
                self.assertEqual(len(cache.read_manifest()['files']), 12)
                changed = deepcopy(config)
                changed['rendering']['profiles']['neutral']['intensity'] = 1.
                new_signature = cache.fingerprint(digest, two, two, changed, torch.device('cpu'))
                self.assertFalse(cache.reuse(new_signature))
                cache.prepare(new_signature, .5)
                with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
                    cache.build(teacher, two, two, changed)
                    self.assertEqual(forward.call_count, 6)
                cache.seal(new_signature)
                for path, (modified, data) in original.items():
                    self.assertEqual(path.stat().st_mtime_ns, modified)
                    self.assertEqual(path.read_bytes(), data)
            finally:
                cache.close(delete=False)

    def test_current_cache_index_gains_profile_signatures_without_reextracting(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            teacher = FixtureTeacher()
            loader = Loader(ClipDataset(directory))
            try:
                signature = cache.fingerprint(module_digest(teacher), loader, loader, self.config, torch.device('cpu'))
                cache.prepare(signature, .5)
                cache.build(teacher, loader, loader, self.config)
                cache.seal(signature)
                index = cache.directory / 'manifest.json'
                manifest = json.loads(index.read_text())
                manifest.pop('profile_signatures')
                index.write_text(json.dumps(manifest))
                times = {p: p.stat().st_mtime_ns for p in cache.directory.rglob('targets.pt')}
                self.assertTrue(cache.reuse(signature))
                self.assertEqual(set(cache.read_manifest()['profile_signatures']), {'standard', 'neutral'})
                self.assertTrue(all(p.stat().st_mtime_ns == modified for p, modified in times.items()))
            finally:
                cache.close(delete=False)

    def test_early_stop_runs_normal_completion_reports_and_best_last_exports(self):
        from .trainer import fit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(self.config)
            config['runtime']['output_dir'] = str(root / 'run')
            config['training'].update(epochs=5, profiles=['neutral'], early_stopping={'patience': 1, 'min_delta': 0.})
            student, criterion = tiny_model(), self.criterion()
            loader = Loader(ClipDataset(root, profiles=('neutral',)))
            teacher = FixtureTeacher()
            fixed = evaluate(teacher, student, criterion, loader, Metrics(), Monitor(), config=config,
                epoch=1, controls={'loss_weights': config['loss']}, blend_scale=.5)
            optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
            completed, cleanups = [], []
            with patch('tool.trainer.FEATURE_CACHE_DIR', root / 'feats'), \
                 patch('tool.trainer.evaluate', return_value=fixed), redirect_stdout(io.StringIO()):
                result = fit(FixtureTeacher, student, criterion, loader, loader, optimizer, Metrics(), config,
                    device='cpu', report_dir=root / 'report', on_complete=completed.append,
                    cache_cleanup=lambda path: cleanups.append(path) or False)
            self.assertEqual(result['status'], 'early_stopped')
            self.assertEqual(result['epochs_run'], 2)
            self.assertEqual(result['best_epoch'], 1)
            self.assertTrue(result['early_stopping']['triggered'])
            self.assertEqual(len(completed), 1)
            self.assertEqual(torch.load(root / 'run/last.pt', weights_only=True)['epoch'], 2)
            payload = torch.load(root / 'run/best.pt', weights_only=True)
            self.assertEqual(list(payload['train_config']['rendering']['profiles']), ['neutral'])
            self.assertTrue((root / 'report/training_metrics.png').is_file())
            self.assertTrue((root / 'report/profile_metrics.png').is_file())
            self.assertEqual(set(p.name for p in (root / 'run').glob('*.pt')), {'best.pt', 'last.pt'})

    def test_raw_motion_recovers_backward_translation_without_guides(self):
        from model.motion import raw_motion
        rng = np.random.default_rng(45)
        previous = rng.random((96, 96, 3), dtype=np.float32)
        current = np.roll(previous, (2, 4), axis=(0, 1))
        motion, confidence = raw_motion(previous, current)
        pixels = motion[20:-20, 20:-20] * 96
        np.testing.assert_allclose(np.median(pixels, axis=(0, 1)), [-4, -2], atol=.4)
        self.assertGreater(float(confidence[20:-20, 20:-20].mean()), .8)

    def test_warp_direction_boundaries_and_displacement_gradients(self):
        from model.motion import warp_history
        source = torch.arange(64, dtype=torch.float32).view(1, 1, 1, 64).expand(1, 3, 32, 64)
        field = torch.zeros(1, 2, 32, 64)
        field[:, 0] = 2 / 64
        field.requires_grad_()
        aligned, valid = warp_history(source, field, torch.ones(1, 1, 32, 64))
        torch.testing.assert_close(aligned[..., :-2], source[..., 2:], atol=2e-5, rtol=0)
        self.assertEqual(float(valid[..., -2:].sum()), 0)
        aligned[..., 3:-3].mean().backward()
        self.assertGreater(float(field.grad[:, 0].abs().sum()), 0)

    def test_feature_motion_is_independent_trainable_and_not_ground_truth_driven(self):
        student = tiny_model('feat')
        self.assertEqual(set(dict(student.motion.named_children())), {'encoder', 'decoder'})
        self.assertEqual(student.motion.encoder[2].out_channels, 64)
        prior = crop_rgb(sample()['images'][0], 0)
        current = np.roll(prior, 2, axis=1)
        target = np.zeros((*prior.shape[:2], 2), np.float32)
        target[..., 0] = -2 / prior.shape[1]
        mask = np.ones((*prior.shape[:2], 1), np.float32)
        optimizer = torch.optim.Adam(student.motion.parameters(), lr=.003)
        errors = []
        for _ in range(12):
            optimizer.zero_grad()
            frame = prepare_student_frame(current, self.config, 'neutral', student,
                previous=prior, history=prior, motion=target, confidence=mask)
            errors.append(float(frame.motion_loss.detach()))
            frame.motion_loss.backward()
            optimizer.step()
        self.assertLess(errors[-1], errors[0])
        self.assertGreater(float(student.motion.encoder[0].weight.grad.abs().sum()), 0)
        a = prepare_student_frame(current, self.config, 'neutral', student,
            previous=prior, history=prior, motion=target, confidence=mask)
        b = prepare_student_frame(current, self.config, 'neutral', student,
            previous=prior, history=prior, motion=target * -10, confidence=mask * 0)
        torch.testing.assert_close(a.history, b.history, atol=0, rtol=0)
        torch.testing.assert_close(a.confidence, b.confidence, atol=0, rtol=0)
        b = prepare_student_frame(current * .5, self.config, 'neutral', student,
            previous=prior, history=prior)
        self.assertGreater(float((a.history - b.history).detach().abs().max()), 1e-6)
        x = student_inputs(a, 'cpu')
        head = student(x)
        output, _ = compose_tensor(head, a, .5)
        student.zero_grad()
        output.mean().backward()
        self.assertGreater(float(student.motion.decoder[-1].weight.grad.abs().sum()), 0)

    def test_feature_mode_updates_and_single_frame_has_no_missing_gradients(self):
        config = deepcopy(self.config)
        config['rendering']['train_min_frames'] = 2
        student, criterion = tiny_model('feat'), self.criterion()
        teacher = FixtureTeacher()
        before = module_digest(teacher)
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.01)
        controls = TrainingSchedule(config, total_steps=2).values(epoch=1, step=0)
        for frames in (1, 2):
            batch = {'sequences': [sample(frames=frames, name=str(i)) for i in range(2)]}
            result = train_batch(teacher, student, criterion, batch, optimizer, config=config,
                crop_size=0, controls=controls, blend_scale=.5, micro_batch_size=2)
            self.assertGreaterEqual(result['components']['motion'], 0)
            self.assertTrue(all(p.grad is not None for p in student.motion.parameters()))
        self.assertEqual(module_digest(teacher), before)
        self.assertGreater(float(student.motion.decoder[-1].weight.detach().abs().sum()), 0)
        frame = prepare_student_frame(crop_rgb(sample()['images'][0], 0), config, 'neutral', student)
        self.assertIsNone(frame.history)

    def test_feature_checkpoint_reloads_exactly_and_rejects_retired_formats(self):
        from .trainer import save_checkpoint, load_student_state
        student, criterion = tiny_model('feat').eval(), self.criterion()
        optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
        rgb = crop_rgb(sample()['images'][0], 0)
        frame = prepare_student_frame(rgb, self.config, 'neutral', student, previous=rgb, history=rgb)
        inputs = student_inputs(frame, 'cpu')
        expected = student(inputs).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            save_checkpoint(path, student, criterion, optimizer, epoch=1, step=1, config=self.config,
                controls={'loss_weights': self.config['loss']}, blend_scale=.5)
            payload = torch.load(path, weights_only=True)
            restored = type(student)(payload['student_config']).eval()
            load_student_state(restored, path)
            self.assertEqual(restored.motion_mode, 'feat')
            self.assertEqual(restored.input_channels, 8)
            torch.testing.assert_close(restored(inputs), expected, atol=0, rtol=0)
            payload['format'] = 'echorender-bhwc16-motion-v4'
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_student_state(restored, path)
        retired = deepcopy(student.config)
        retired['input_channels'] = 16
        with self.assertRaises(ValueError):
            type(student)(retired)

    def test_rgb_only_prediction_has_no_dependency_on_dataset_flow_in_either_mode(self):
        from .prediction_report import predict_clip
        config = deepcopy(self.config)
        config['reporting'].update(crop_size=0, inference_size=0)
        rgb_only = sample(frames=2)
        rgb_only['motions'] = [None, None]
        rgb_only['confidences'] = [None, None]
        for mode in ('raw', 'feat'):
            student = tiny_model(mode).eval()
            with patch('tool.temporal.frame_guides', side_effect=AssertionError('Student must not read flow')):
                _, prediction, info = predict_clip(student, rgb_only, 'original', 0, 'neutral', config,
                    torch.device('cpu'), .5, False)
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue(info['history_used'])
            self.assertEqual(info['motion_source'], mode)

    def test_test_sequences_need_only_rgb_and_label_estimated_reference_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'test/clean/demo'
            folder.mkdir(parents=True)
            for index in (1, 2):
                Image.fromarray(np.full((32, 32, 3), 100 + index, np.uint8)).save(folder / f'frame_{index:04}.png')
            config = deepcopy(self.config)
            config['dataset']['root'] = str(root)
            loader = build_dataloader(load_data_config(config), split='test')
            item = loader.dataset[0]
            self.assertEqual(item['motion_reference'], 'estimated_from_rgb')
            self.assertEqual(len(item['images']), 2)
            self.assertEqual(item['motions'][1].shape, (32, 32, 2))

    def test_composition_matches_teacher_host_for_all_profiles_and_history(self):
        rgb = crop_rgb(sample()['images'][0], 0)
        for profile in self.config['rendering']['profiles']:
            for temporal in (False, True):
                motion = np.zeros((*rgb.shape[:2], 2), np.float32)
                confidence = np.ones((*rgb.shape[:2], 1), np.float32) * .7
                frame = prepare_frame(rgb, self.config, profile, history=rgb * .8 if temporal else None,
                                      motion=motion, confidence=confidence)
                head = (torch.rand(1, *rgb.shape[:2], 4) * .2).requires_grad_()
                actual, history = compose_tensor(head, frame, .5)
                expected, expected_history = compose_output(head.detach().numpy()[0], rgb, frame.controls,
                    blend_scale=.5, history=frame.history, confidence=frame.confidence,
                    history_strength=1., **{k: frame.settings[k] for k in
                    ('intensity', 'detail_strength', 'colour_strength', 'detail_radius')})
                np.testing.assert_allclose(actual.detach().numpy()[0].transpose(1, 2, 0), expected, atol=2e-6, rtol=1e-5)
                np.testing.assert_allclose(history, expected_history, atol=5e-4)
                actual.mean().backward()
                self.assertTrue(torch.isfinite(head.grad).all())
                self.assertGreater(float(head.grad.abs().sum()), 0)

    def test_schedule_caps_corrupted_clips_and_keeps_originals(self):
        self.config['training']['epochs'] = 100
        aug = StudentAugmentation(self.config)
        self.assertEqual(aug.ratio(1), .1)
        self.assertEqual(aug.ratio(50), .25)
        self.assertEqual(aug.ratio(74), .25)
        self.assertEqual(aug.ratio(75), 0)
        for epoch in (1, 25, 50, 75):
            ratio = aug.ratio(epoch)
            for count in (1, 3, 4, 16, 19):
                plan = aug.plan(count, ratio)
                self.assertLessEqual(len(plan), count * ratio + 1e-8)
                self.assertLess(len(plan), count)
        lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6]
        plan = aug.plan(len(lengths), .25, lengths)
        self.assertLessEqual(sum(lengths[i] for i in plan), sum(lengths) * .25)
        rgb = np.full((32, 32, 3), .5, np.float32)
        for mode in ('brightness', 'contrast', 'sharpness'):
            actual = aug.apply_rgb(rgb, mode, .25)
            self.assertGreater(float(actual.std()), .01)
        self.assertEqual(aug.plan(16, 0), {})

    def test_microbatch_gradients_and_online_cache_targets_match(self):
        config = self.config
        config['rendering']['train_min_frames'] = 2
        batch = {'sequences': [sample(frames=2, name=str(i)) for i in range(3)]}
        a, criterion = tiny_model(), self.criterion()
        b, other_criterion = deepcopy(a), deepcopy(criterion)
        targets = {}
        teacher = FixtureTeacher()
        for s in batch['sequences']:
            history = None
            for i in range(2):
                target = teacher_target(teacher, make_frame(s, i, config, history))
                targets[cache_key(s, i)] = target
                history = target['history']
        cache = SimpleNamespace(load_payload=lambda split, key: targets[key])
        controls = TrainingSchedule(config, total_steps=2).values(epoch=1, step=0)
        values = []
        for student, loss, cached, micro in ((a, criterion, None, 1), (b, other_criterion, cache, 2)):
            optimizer = torch.optim.SGD([*student.parameters(), *loss.parameters()], lr=.001)
            values.append(train_batch(teacher if cached is None else None, student, loss, batch, optimizer,
                config=config, crop_size=0, controls=controls, augmentation=StudentAugmentation(config),
                augmentation_ratio=0, target_cache=cached, blend_scale=.5, micro_batch_size=micro))
        self.assertAlmostEqual(values[0]['loss'], values[1]['loss'], places=5)
        for actual, expected in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-6)
        self.assertEqual(values[1]['frames'], 6)
        self.assertIn(2, values[1]['micro_batch_sizes'])

    def test_evaluation_pairs_same_frame_and_reports_profiles_and_stages(self):
        student, criterion, teacher = tiny_model(), self.criterion(), FixtureTeacher()
        batch = {'sequences': [sample(profile=p) for p in self.config['rendering']['profiles']]}
        values = evaluate(teacher, student, criterion, [batch], Metrics(), Monitor(), config=self.config,
            epoch=1, controls={'loss_weights': self.config['loss']}, blend_scale=.5)
        self.assertEqual(values['images'], 12)
        self.assertEqual(len(values['history_stages']), 12)
        for row in values['history_stages']:
            if row['frame_index'] == 1:
                self.assertEqual(row['delta_psnr'], 0)
                self.assertEqual(row['history_used_fraction'], 0)
            else:
                self.assertEqual(row['history_used_fraction'], 1)
                self.assertEqual(row['mean_history_frames'], row['frame_index'] - 1)
                self.assertGreater(row['valid_pixels'], 0)
        self.assertEqual(set(values['profiles']), set(self.config['rendering']['profiles']))
        with tempfile.TemporaryDirectory() as directory:
            write_stage_report(values, directory)
            self.assertTrue((Path(directory) / 'temporal.csv').is_file())
            self.assertTrue((Path(directory) / 'temporal.png').is_file())

    def test_batched_validation_matches_serial_scores_and_keeps_history_order(self):
        student, criterion, teacher = tiny_model(), self.criterion(), FixtureTeacher()
        batch = {'sequences': [sample(profile=p) for p in self.config['rendering']['profiles']]}
        config = deepcopy(self.config)
        results = []
        calls = []
        hook = student.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
        try:
            for maximum in (1, 4):
                config['training']['micro_batch_size'] = maximum
                results.append(evaluate(teacher, student, criterion, [batch], Metrics(), Monitor(), config=config,
                    epoch=1, controls={'loss_weights': config['loss']}, blend_scale=.5))
        finally:
            hook.remove()
        self.assertIn(4, calls)
        self.assertEqual(results[0]['images'], results[1]['images'])
        for first, second in zip(results[0]['history_stages'], results[1]['history_stages']):
            for key in ('loss', 'psnr', 'ssim', 'lpips', 'single_psnr', 'mean_history_frames'):
                self.assertAlmostEqual(first[key], second[key], delta=2e-4)

    def test_vectorized_image_metrics_match_individual_images(self):
        from .metrics import ImageMetrics
        class Perceptual(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).abs().mean((1, 2, 3), keepdim=True)
        metrics = ImageMetrics.__new__(ImageMetrics)
        torch.nn.Module.__init__(metrics)
        metrics.perceptual = Perceptual()
        metrics.ssim_window = 3
        prediction, target = torch.rand(4, 3, 64, 64), torch.rand(4, 3, 64, 64)
        batched = metrics.per_image(prediction, target)
        individual = [metrics(prediction[i:i + 1], target[i:i + 1]) for i in range(4)]
        for actual, expected in zip(batched, individual):
            for key in actual:
                self.assertAlmostEqual(actual[key], expected[key], delta=2e-6)

    def test_cache_decodes_each_physical_clip_once_per_split(self):
        with tempfile.TemporaryDirectory() as directory:
            class Counting(ClipDataset):
                reads = 0
                def __getitem__(self, index):
                    self.reads += 1
                    return super().__getitem__(index)
            dataset = Counting(directory, profiles=tuple(self.config['rendering']['profiles']))
            loader = Loader(dataset)
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            try:
                cache.prepare('speed', .5)
                cache.build(FixtureTeacher(), loader, loader, self.config)
                self.assertEqual(dataset.reads, 2)
                self.assertEqual(len(cache._progress['files']), 4 * 3 * 2)
            finally:
                cache.close(delete=False)

    def test_forward_benchmark_uses_8_student_and_16_teacher_channels(self):
        self.config['benchmark'].update(enabled=True, warmup=0, repeats=1)
        result = evaluate(FixtureTeacher(), tiny_model(), self.criterion(),
            [{'sequences': [sample(frames=1)]}], Metrics(), Monitor(), config=self.config, epoch=1,
            controls={'loss_weights': self.config['loss']}, blend_scale=.5)
        self.assertGreater(result['benchmark']['teacher_latency_ms'], 0)
        self.assertGreater(result['benchmark']['student_latency_ms'], 0)
        self.assertEqual(result['benchmark']['student_input_shape'][-1], 8)

    def test_cache_groups_frame_payloads_and_rejects_old_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            cache.open()
            try:
                cache.prepare('a', .5)
                s = sample(frames=1)
                key = cache_key(s, 0)
                target = teacher_target(FixtureTeacher(), make_frame(s, 0, self.config))
                target['history'] = torch.from_numpy(target['history'])
                cache.store('train', key, target)
                path = cache.path('train', key)
                self.assertEqual(path.relative_to(cache.directory).as_posix(),
                    'neutral/train/training/demo/frame_0001/targets.pt')
                cache.seal('a')
                self.assertTrue(cache.reuse('a'))
                self.assertFalse(cache.reuse('old-single-image'))
                cached = cache.load_payload('train', key)
                self.assertEqual(cached['frame_index'], 1)
                self.assertEqual(cached['profile'], 'neutral')
                torch.testing.assert_close(cached['head'], target['head'])
                with self.assertRaises(RuntimeError):
                    TeacherFeatureCache(cache.directory).open()
            finally:
                cache.close(delete=False)

    def test_scene_reset_and_invalid_motion_do_not_use_bad_history(self):
        s = sample(frames=2)
        s['confidences'][1][:] = 0
        first = make_frame(s, 0, self.config)
        target = teacher_target(FixtureTeacher(), first)
        second = make_frame(s, 1, self.config, target['history'])
        np.testing.assert_array_equal(second.inputs[..., 4:7], second.inputs[..., 7:10])
        disabled = deepcopy(self.config)
        disabled['rendering']['history_strength'] = 0
        self.assertIsNone(make_frame(s, 1, disabled, target['history']).history)
        student = tiny_model('feat')
        current = crop_rgb(s['images'][1], 0)
        frame = prepare_student_frame(current, self.config, 'neutral', student,
            previous=current, history=target['history'])
        frame.confidence = torch.zeros_like(frame.confidence)
        x = student_inputs(frame, 'cpu')
        torch.testing.assert_close(x[..., :3], x[..., 3:6], atol=0, rtol=0)

    def test_loader_orders_consecutive_clips_and_expands_four_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in ('train_scene', 'val_scene'):
                for name in ('clean', 'flow', 'invalid', 'occlusions'):
                    (root / 'training' / name / scene).mkdir(parents=True)
                for i in range(1, 14):
                    name = f'frame_{i:04}'
                    Image.fromarray(np.full((32, 32, 3), i, np.uint8)).save(root / 'training/clean' / scene / (name + '.png'))
                    for kind in ('invalid', 'occlusions'):
                        Image.fromarray(np.zeros((32, 32), np.uint8)).save(root / 'training' / kind / scene / (name + '.png'))
                    write_flow(root / 'training/flow' / scene / (name + '.flo'), np.zeros((32, 32, 2), np.float32))
            c = deepcopy(self.config)
            c['dataset']['root'] = str(root)
            c['dataset']['splits']['train']['scenes'] = ['train_scene']
            c['dataset']['splits']['val']['scenes'] = ['val_scene']
            loader = build_dataloader(load_data_config(c))
            self.assertEqual(len(loader.dataset), 8)
            self.assertEqual([len(paths) for _, paths in loader.dataset.clips], [10, 3])
            self.assertEqual(loader.dataset[0]['profile'], 'standard')
            self.assertEqual(loader.dataset[3]['profile'], 'neutral')
            c['rendering']['temporal'] = False
            self.assertEqual(len(build_dataloader(load_data_config(c)).dataset), 13 * 4)
            c['dataset']['render_pass'] = 'final'
            with self.assertRaises(ValueError):
                load_data_config(c)

    def test_rendering_schema_rejects_retired_knobs_and_invalid_lengths(self):
        import yaml
        for key, value in (('max_frames', 11), ('max_frames', 0), ('train_min_frames', 11),
                           ('motion_mode', 'identity'), ('control_mask', None)):
            c = deepcopy(self.config)
            c.pop('config_path')
            c['rendering'][key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.yaml'
                path.write_text(yaml.safe_dump(c))
                with self.assertRaises(ValueError):
                    load_training_config(path)

    def test_parallel_cache_resume_does_not_repeat_complete_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = Loader(ClipDataset(directory))
            cache = TeacherFeatureCache(Path(directory) / 'feats')
            teacher = FixtureTeacher()
            cache.open()
            try:
                fingerprint = cache.fingerprint(module_digest(teacher), loader, loader, self.config, torch.device('cpu'))
                cache.prepare(fingerprint, .5)
                with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
                    cache.build(teacher, loader, loader, self.config)
                    self.assertEqual(forward.call_count, 12)
                cache.seal(fingerprint)
                self.assertTrue(cache.reuse(fingerprint))
                missing = cache.path('val', cache_key(loader.dataset[1], 1))
                missing.unlink()
                self.assertFalse(cache.reuse(fingerprint))
                cache.prepare(fingerprint, .5)
                with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
                    cache.build(teacher, loader, loader, self.config)
                    self.assertEqual(forward.call_count, 1)
                cache.seal(fingerprint)
                self.assertTrue(cache.reuse(fingerprint))
                changed = deepcopy(self.config)
                changed['rendering']['profiles']['neutral']['intensity'] = 1.
                self.assertNotEqual(cache.fingerprint(module_digest(teacher), loader, loader, changed, torch.device('cpu')), fingerprint)
            finally:
                cache.close(delete=False)

    def test_full_fit_releases_teacher_saves_best_last_and_applies_cosine(self):
        from .trainer import fit
        import weakref
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(self.config)
            config['runtime']['output_dir'] = str(root / 'run')
            config['training'].update(epochs=2, releas=True, max_steps=2, micro_batch_size=2)
            config['rendering']['train_min_frames'] = 3
            student, criterion = tiny_model(), self.criterion()
            optimizer = torch.optim.SGD([*student.parameters(), *criterion.parameters()], lr=.001)
            loader = Loader(ClipDataset(root))
            objects, completed = [], []
            def create_teacher():
                teacher = FixtureTeacher()
                objects.append(weakref.ref(teacher))
                return teacher
            with patch('tool.trainer.FEATURE_CACHE_DIR', root / 'feats'), redirect_stdout(io.StringIO()):
                result = fit(create_teacher, student, criterion, loader, loader, optimizer, Metrics(), config,
                    device='cpu', cache_cleanup=lambda path: False, report_dir=root / 'report', on_complete=completed.append)
            self.assertEqual(result['global_step'], 2)
            self.assertTrue(result['teacher_released'])
            self.assertTrue(result['teacher_unchanged'])
            self.assertTrue(result['student_parameters_updated'])
            self.assertIsNone(objects[0]())
            self.assertEqual(optimizer.param_groups[0]['lr'], config['schedule']['eta_min'])
            self.assertEqual(set(p.name for p in (root / 'run').glob('*.pt')), {'best.pt', 'last.pt'})
            payload = torch.load(root / 'run/best.pt', weights_only=True)
            self.assertEqual(payload['format'], FORMAT)
            self.assertEqual(len(completed), 1)
            self.assertTrue((root / 'report/temporal_latest.csv').is_file())
            self.assertTrue((root / 'report/training_metrics.png').is_file())

    def test_standalone_validation_keeps_training_index_and_reuses_evaluation_targets(self):
        import val
        from . import trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student, criterion = tiny_model(), self.criterion()
            payload = {'format': FORMAT, 'epoch': 1, 'student_config': student.config,
                'student': student.state_dict(), 'distillation': criterion.state_dict(),
                'teacher_blend_scale': .5, 'train_config': self.config}
            cache_root = root / 'feats'
            cache_root.mkdir()
            (cache_root / 'manifest.json').write_text('{"existing_training_index":true}')
            training_index = (cache_root / 'manifest.json').read_bytes()
            loader = Loader(ClipDataset(root))
            def factory():
                return FixtureTeacher()
            with patch.object(trainer, 'FEATURE_CACHE_DIR', cache_root), \
                 patch('val.DistillationLoss', side_effect=lambda config, **kwargs: self.criterion()), \
                 patch('val.ImageMetrics', return_value=Metrics()), \
                 patch.object(FixtureTeacher, 'forward', autospec=True, side_effect=FixtureTeacher.forward) as forward:
                first = val.evaluate_checkpoint(payload, factory, {'val': loader}, self.config, torch.device('cpu'))
                calls = forward.call_count
                second = val.evaluate_checkpoint(payload, factory, {'val': loader}, self.config, torch.device('cpu'))
                self.assertEqual(forward.call_count, calls)
            self.assertEqual(first['val']['image_mean'], second['val']['image_mean'])
            self.assertEqual((cache_root / 'manifest.json').read_bytes(), training_index)
            self.assertTrue((cache_root / 'evaluation.json').is_file())

    def test_evaluation_cache_retains_previous_profiles_when_adding_another(self):
        import val
        from . import trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student, criterion = tiny_model(), self.criterion()
            payload = {'format': FORMAT, 'epoch': 1, 'student_config': student.config,
                'student': student.state_dict(), 'distillation': criterion.state_dict(),
                'teacher_blend_scale': .5, 'train_config': self.config}
            with patch.object(trainer, 'FEATURE_CACHE_DIR', root / 'feats'), \
                 patch('val.DistillationLoss', side_effect=lambda config, **kwargs: self.criterion()), \
                 patch('val.ImageMetrics', return_value=Metrics()), \
                 patch.object(FixtureTeacher, 'forward', autospec=True, side_effect=FixtureTeacher.forward) as forward:
                for profiles in (('standard',), ('standard', 'neutral'), ('standard',)):
                    config = deepcopy(self.config)
                    config['training']['profiles'] = list(profiles)
                    config['rendering']['profiles'] = {p: config['rendering']['profiles'][p] for p in profiles}
                    loader = Loader(ClipDataset(root, profiles=profiles))
                    before = forward.call_count
                    val.evaluate_checkpoint(payload, FixtureTeacher, {'val': loader}, config, torch.device('cpu'))
                    self.assertEqual(forward.call_count - before, 0 if profiles == ('standard',) and before else 3)
                manifest = json.loads((root / 'feats/evaluation.json').read_text())
                self.assertEqual(set(manifest['profile_signatures']), {'standard', 'neutral'})
                self.assertEqual(len(manifest['files']), 6)

    def test_validation_limit_retains_all_profiles_of_the_selected_clip(self):
        import val
        from .dataloader import collate_sequences
        loader = torch.utils.data.DataLoader(ClipDataset('.', profiles=tuple(self.config['rendering']['profiles'])),
            batch_size=4, collate_fn=collate_sequences)
        limited = val.limit_loader(loader, 1)
        self.assertEqual(len(limited.dataset), 4)
        self.assertEqual({s['profile'] for s in next(iter(limited))['sequences']}, set(self.config['rendering']['profiles']))

    def test_prediction_grids_include_all_profiles_and_preserve_single_mode_augmentation(self):
        from .prediction_report import export_predictions
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = tiny_model('feat')
            weights = root / 'best.pt'
            torch.save({'format': FORMAT, 'epoch': 1, 'student_config': student.config,
                'student': student.state_dict(), 'teacher_blend_scale': .5, 'train_config': self.config}, weights)
            config = deepcopy(self.config)
            config['reporting'].update(samples_per_mode=1, strengths=[.25, 1.], inference_size=0)
            dataset = ClipDataset(root, profiles=tuple(config['rendering']['profiles']))
            # One-frame reports exercise the legitimate temporal first-frame path.
            original = sample
            class OneFrame(ClipDataset):
                def __getitem__(self, index):
                    return original(profile=self.items[index][1], frames=1)
            dataset = OneFrame(root, profiles=dataset.profiles)
            with redirect_stdout(io.StringIO()):
                manifest = export_predictions(weights, FixtureTeacher, {'val': Loader(dataset)}, config,
                                              root / 'report', 'cpu')
            self.assertEqual(manifest['status'], 'completed')
            grids = manifest['grids']
            self.assertEqual(len(grids), 2 * 4 * 5)
            self.assertEqual({g['profile'] for g in grids}, set(config['rendering']['profiles']))
            self.assertEqual(len(list((root / 'report/preview').rglob('*.png'))), 5)
            for grid in grids:
                self.assertIn('total infer', grid['title'])
                self.assertIn('avg infer', grid['title'])
                self.assertTrue((root / 'report' / grid['path']).is_file())
            rows = [json.loads(line) for line in (root / 'report/predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['model_input_shape'][-1] == (8 if r['model'] == 'student' else 16) for r in rows))


if __name__ == '__main__':
    unittest.main()
