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
from .temporal import (prepare_frame, student_arguments, compose_tensor, teacher_target, crop_rgb,
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

    def test_profile_controls_and_rgb_history_have_effect_without_extra_channels(self):
        student = tiny_model().eval()
        rgb = crop_rgb(sample()['images'][0], 0)
        outputs, vectors = [], []
        for profile in self.config['rendering']['profiles']:
            frame = prepare_frame(rgb, self.config, profile)
            x, c, h = student_arguments(frame, 'cpu')
            self.assertEqual(x.shape[-1], 3)
            self.assertEqual(c.shape[-1], 16)
            outputs.append(student(x, controls=c, history=h))
            vectors.append(c)
        self.assertTrue(all(not torch.equal(vectors[0], c) for c in vectors[1:]))
        self.assertGreater(float((outputs[0] - outputs[-1]).detach().abs().max()), 1e-6)
        self.assertFalse(torch.equal(student(x, controls=c, history=h + .1), outputs[-1]))
        with self.assertRaises(ValueError):
            student(torch.zeros(1, 32, 32, 16), controls=c)

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

    def test_forward_benchmark_receives_external_controls(self):
        self.config['benchmark'].update(enabled=True, warmup=0, repeats=1)
        result = evaluate(FixtureTeacher(), tiny_model(), self.criterion(),
            [{'sequences': [sample(frames=1)]}], Metrics(), Monitor(), config=self.config, epoch=1,
            controls={'loss_weights': self.config['loss']}, blend_scale=.5)
        self.assertGreater(result['benchmark']['teacher_latency_ms'], 0)
        self.assertGreater(result['benchmark']['student_latency_ms'], 0)
        self.assertEqual(result['benchmark']['student_input_shape'][-1], 3)

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
        x, c, history = student_arguments(second, 'cpu')
        np.testing.assert_allclose(history[0].permute(1, 2, 0).numpy(), second.small, atol=0)

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
            student = tiny_model()
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
            self.assertTrue(all(r['model_input_shape'][-1] == (3 if r['model'] == 'student' else 16) for r in rows))


if __name__ == '__main__':
    unittest.main()
