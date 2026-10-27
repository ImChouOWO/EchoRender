"""Teacher inference contract, Sintel guide alignment and function API checks."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import teacher_infer as infer
from mlxdlss.composition import compose_head
from mlxdlss.features import make_features
from .training_config import load_training_config


def write_flow(path, flow):
    with path.open('wb') as stream:
        stream.write(b'PIEH')
        stream.write(np.array([flow.shape[1], flow.shape[0]], dtype='<i4').tobytes())
        stream.write(flow.astype('<f4').tobytes())


class TeacherFixture(nn.Module):
    blend_scale = .5

    def __init__(self, *args, device='cpu'):
        super().__init__()
        self.register_buffer('anchor', torch.tensor(.125, device=device))
        self.seen = []

    @property
    def device(self):
        return self.anchor.device

    def forward(self, inputs):
        self.seen.append(inputs.clone())
        head = torch.zeros((*inputs.shape[:-1], 4), device=self.device)
        head[..., :3] = self.anchor
        return head


class TeacherInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


    def test_flow_reader_roundtrip_and_rejects_bad_files(self):
        flow = np.random.default_rng(42).normal(size=(5, 9, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'test.flo'
            write_flow(path, flow)
            np.testing.assert_array_equal(infer.read_flow(path), flow)
            for contents in (b'bad!', b'PIEH', path.read_bytes()[:-4]):
                path.write_bytes(contents)
                with self.assertRaises(ValueError):
                    infer.read_flow(path)


    def test_forward_translation_is_inverted_in_current_pixel_coordinates(self):
        flow = np.broadcast_to(np.array([2., -1.], np.float32), (8, 12, 2)).copy()
        motion, confidence = infer.invert_forward_flow(flow, np.ones((8, 12), bool))
        np.testing.assert_allclose(motion[:-1, 2:], np.broadcast_to([-2/12, 1/8], (7, 10, 2)), atol=1e-8)
        np.testing.assert_array_equal(confidence[:-1, 2:], 1)
        np.testing.assert_array_equal(confidence[-1], 0)
        np.testing.assert_array_equal(confidence[:, :2], 0)


    def test_holes_collisions_and_nonfinite_flow_reject_history(self):
        yy, xx = np.indices((5, 9))
        flow = np.stack((4 - xx, np.zeros_like(yy)), axis=-1).astype(np.float32)
        motion, confidence = infer.invert_forward_flow(flow, np.ones((5, 9), bool))
        np.testing.assert_array_equal(confidence, 0)
        self.assertTrue(np.isfinite(motion).all())
        flow[:] = np.nan
        motion, confidence = infer.invert_forward_flow(flow, np.ones((5, 9), bool))
        np.testing.assert_array_equal(motion, 0)
        np.testing.assert_array_equal(confidence, 0)


    def test_dataset_masks_follow_forward_source_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('flow', 'occlusions', 'invalid'):
                (root / 'training' / name / 'demo').mkdir(parents=True)
            first, second = Path('frame_0001.png'), Path('frame_0002.png')
            flow = np.broadcast_to(np.array([1., 0.], np.float32), (5, 9, 2)).copy()
            write_flow(root/'training/flow/demo/frame_0001.flo', flow)
            mask = np.zeros((5, 9), np.uint8)
            mask[2, 3] = 255
            Image.fromarray(mask).save(root/'training/occlusions/demo/frame_0001.png')
            Image.fromarray(np.zeros_like(mask)).save(root/'training/invalid/demo/frame_0001.png')
            mask[:] = 0
            mask[1, 7] = 255
            Image.fromarray(mask).save(root/'training/invalid/demo/frame_0002.png')
            motion, confidence, guides = infer.dataset_motion(root, 'training', 'demo', first, second, (5, 9))
            self.assertEqual(confidence[2, 4, 0], 0)
            self.assertEqual(confidence[2, 3, 0], 1)
            self.assertEqual(confidence[1, 7, 0], 0)
            self.assertAlmostEqual(float(motion[2, 3, 0]), -1/9)
            self.assertIn('frame_0001.flo', guides['flow'])


    def test_still_uses_intel_geometry_features_and_ignores_temporal_gate(self):
        teacher = TeacherFixture()
        rgb = np.random.default_rng(42).random((31, 47, 3), dtype=np.float32)
        prediction, history, metadata = infer.infer_image(teacher, rgb)
        geometry = infer.network_geometry(47, 31)
        expected_features = make_features(rgb, geometry=geometry)
        np.testing.assert_array_equal(teacher.seen[0].numpy()[0], expected_features)
        head = teacher(torch.from_numpy(expected_features).unsqueeze(0))[0].numpy()[:31, :47]
        expected = compose_head(head, rgb)
        np.testing.assert_array_equal(prediction, expected)
        self.assertEqual(history.shape, rgb.shape)
        self.assertFalse(metadata['history_used'])

    def test_geometry_matches_intel_padding_boundaries(self):
        for (w, h), expected in { (512, 512): (576, 512), (1280, 720): (1344, 768),
                                 (1024, 768): (1088, 768), (320, 180): (320, 320)}.items():
            geometry = infer.network_geometry(w, h)
            self.assertEqual((geometry.network_width, geometry.network_height), expected)

    def test_controls_change_condition_channels_and_mask_meaning(self):
        teacher = TeacherFixture()
        rgb = np.full((8, 12, 3), .25, np.float32)
        controls, automatic = infer.network_controls('natural', style_index=2, tone=.5, structure=1.5)
        mask = np.broadcast_to([0., .5, .25], rgb.shape).astype(np.float32)
        prediction, _, _ = infer.infer_image(teacher, rgb, controls=controls, automatic_mask=automatic,
                                            control_mask=mask, noise_index=3)
        seen = teacher.seen[-1].numpy()[0]
        np.testing.assert_array_equal(seen[..., 10], 2/128)
        np.testing.assert_array_equal(seen[..., 11], .25)
        np.testing.assert_array_equal(seen[..., 12], .375)
        np.testing.assert_array_equal(prediction, rgb)
        with self.assertRaisesRegex(ValueError, 'not both'):
            infer.network_controls(skin_structure=2, control_mask=Path('mask.png'))

    def test_temporal_confidence_zero_replaces_history_and_disables_blend(self):
        teacher = TeacherFixture()
        rgb = np.full((8, 12, 3), .25, np.float32)
        prediction, _, metadata = infer.infer_image(teacher, rgb, history=np.full_like(rgb, .8),
            motion=np.zeros((8, 12, 2), np.float32), confidence=np.zeros((8, 12, 1), np.float32))
        seen = teacher.seen[-1].numpy()
        np.testing.assert_array_equal(seen[..., 7:10], seen[..., 4:7])
        np.testing.assert_array_equal(prediction, rgb + .125 * .25)
        self.assertTrue(metadata['history_used'])
        self.assertEqual(metadata['history_valid_fraction'], 0)

    def test_inspection_does_not_load_teacher(self):
        config = load_training_config()
        with tempfile.TemporaryDirectory() as temporary:
            config['dataset']['root'] = temporary
            with patch('teacher_infer.EchoRenderTeacher') as teacher, redirect_stdout(io.StringIO()) as output:
                result = infer.run_inference(config=config, inspect_data=True)
                teacher.assert_not_called()
            self.assertEqual(result, json.loads(output.getvalue()))
            self.assertEqual(result['flow']['files'], 0)
            self.assertIn('not an additional', result['albedo']['use'])

    def test_scene_functions_use_selected_motion_and_increment_output_names(self):
        config = deepcopy(load_training_config())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config['dataset']['root'] = str(root/'dataset')
            config['runtime']['output_dir'] = str(root/'output/train')
            config['runtime']['device'] = 'cpu'
            for name in ('clean', 'flow', 'invalid', 'occlusions'):
                (root/'dataset/training'/name/'demo').mkdir(parents=True)
            for number in (1, 2):
                frame = f'frame_{number:04}.png'
                Image.fromarray(np.full((8, 12, 3), 64, np.uint8)).save(root/'dataset/training/clean/demo'/frame)
                Image.fromarray(np.zeros((8, 12), np.uint8)).save(root/'dataset/training/invalid/demo'/frame)
            Image.fromarray(np.zeros((8, 12), np.uint8)).save(root/'dataset/training/occlusions/demo/frame_0001.png')
            write_flow(root/'dataset/training/flow/demo/frame_0001.flo', np.zeros((8, 12, 2), np.float32))
            for mode in ('sintel',):
                with patch('teacher_infer.EchoRenderTeacher', TeacherFixture), redirect_stdout(io.StringIO()):
                    result = infer.run_inference(config=config, scene='demo', temporal=True,
                        motion_mode=mode, project=mode, max_frames=2)
                    infer.run_inference(config=config, scene='demo', temporal=True,
                        motion_mode=mode, project=mode, max_frames=2)
                self.assertEqual([row['history_used'] for row in result['frames']], [False, True])
                self.assertEqual([row['noise_index'] for row in result['frames']], [0, 0])
                self.assertEqual(result['frames'][1]['history_valid_fraction'], 1)
                self.assertEqual(bool(result['frames'][1]['guides']), mode == 'sintel')
                for project in (mode, mode + '_2'):
                    self.assertTrue((root/'output/infer'/project/'report.json').is_file())
                    self.assertEqual(len(list((root/'output/infer'/project).glob('*.png'))), 2)

    def test_function_validation_happens_before_loading_teacher(self):
        for options in ({'project': '../escape'}, {'tone': float('nan')}, {'structure': 3},
                        {'max_frames': 1.5}, {'max_frames': 11}, {'max_frames': 0}, {'render_pass': 'final'},
                        {'render_scale': 0}, {'motion_mode': 'identity'}, {'motion_mode': 'unknown'}):
            with self.subTest(options=options), self.assertRaises(ValueError), \
                 patch('teacher_infer.EchoRenderTeacher') as teacher:
                infer.run_inference(scene='demo', **options)
                teacher.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            infer.run_inference(input_image='image.png', temporal=True)

    def test_demo_is_a_parameterized_function_without_cli(self):
        with patch('teacher_infer.run_inference', return_value={'done': True}) as run:
            self.assertEqual(infer.demo(), {'done': True})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs['scene'], 'alley_1')
        self.assertEqual(run.call_args.kwargs['device'], 'cuda:1')
        self.assertEqual(run.call_args.kwargs['motion_mode'], 'sintel')
        self.assertFalse(hasattr(infer, 'parser'))
        self.assertFalse(hasattr(infer, 'TeacherInference'))

    def test_history_is_saved_before_intensity_and_style_grade(self):
        rgb = np.full((3, 5, 3), .25, np.float32)
        head = np.full((3, 5, 4), .125, np.float32)
        controls, _ = infer.network_controls('natural')
        first, history_a = infer.compose_output(head, rgb, controls, blend_scale=.5, intensity=0)
        second, history_b = infer.compose_output(head, rgb, controls, blend_scale=.5, intensity=2)
        np.testing.assert_array_equal(first, rgb)
        np.testing.assert_array_equal(history_a, history_b)
        np.testing.assert_array_equal(history_a, .28125)
        self.assertGreater(float(np.abs(second-first).max()), 0)

    def test_cinematic_grade_uses_hsl_saturation_and_tone(self):
        controls, _ = infer.network_controls('cinematic')
        red = np.array([[[1., 0., 0.]]], np.float32)
        np.testing.assert_allclose(infer.style_grade(red, controls), [[[.925, .075, .075]]], atol=1e-7)
        controls['local_tone_strength'] = 0
        np.testing.assert_array_equal(infer.style_grade(red, controls), red)

    def test_truncated_history_never_rounds_away_from_zero(self):
        image = np.random.default_rng(42).uniform(-1, 1, 10000).astype(np.float32)
        stored = infer.truncate_half(image)
        self.assertTrue((np.abs(stored) <= np.abs(image)).all())
        np.testing.assert_array_equal(stored.astype(np.float16).astype(np.float32), stored)


if __name__ == '__main__':
    unittest.main()
