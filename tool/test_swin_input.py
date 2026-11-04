"""Fixed RGB maps, named front/head supervision and unchanged Teacher output."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.card.card import InputAdapter, SwinBlock
from model.card.loss import DistillationLoss, normalized_feature_mse
from model.student import EchoRenderStudent
from .temporal import rgb_feature_maps, prepare_student_frame, student_inputs
from .teacher_features import FeatureRecordingBackend
from .training_config import load_training_config
from .test_training import tiny_model


class SwinInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gaussian_detail_and_signed_gradients_match_reference(self):
        rgb = np.random.default_rng(12).random((19, 23, 3), dtype=np.float32)
        actual = rgb_feature_maps(torch.from_numpy(rgb).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
        low = cv2.GaussianBlur(rgb, (5, 5), 1., borderType=cv2.BORDER_REPLICATE)
        gray = rgb @ np.array([.299, .587, .114], np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REPLICATE)
        from mlxdlss.features import scaled_color
        np.testing.assert_allclose(actual[..., :3], scaled_color(low), atol=1e-7)
        np.testing.assert_allclose(actual[..., 3:6], (rgb - low) * .125, atol=3e-8)
        np.testing.assert_allclose(actual[..., 6:], np.stack((gx, gy), -1) * .125, atol=2e-8)
        self.assertLess(actual[..., 3:].min(), 0.)
        self.assertGreater(actual[..., 3:].max(), 0.)
        constant = rgb_feature_maps(torch.full((1, 3, 7, 9), .25))
        torch.testing.assert_close(constant[:, 3:], torch.zeros_like(constant[:, 3:]), atol=1e-8, rtol=0)

    def test_maps_follow_augmented_input_and_are_reused(self):
        config = load_training_config()
        student = tiny_model()
        original = np.random.default_rng(3).random((32, 32, 3), dtype=np.float32)
        augmented = np.clip(original * 1.25, 0, 1)
        frame = prepare_student_frame(augmented, config, 'neutral', student)
        first = student_inputs(frame, 'cpu')
        torch.testing.assert_close(first[..., :8], frame.inputs)
        expected = rgb_feature_maps(torch.from_numpy(augmented).permute(2, 0, 1)[None]).permute(0, 2, 3, 1)
        torch.testing.assert_close(first[..., 8:], expected, atol=0, rtol=0)
        with patch('tool.temporal.rgb_feature_maps', side_effect=AssertionError('recomputed fixed maps')):
            torch.testing.assert_close(student_inputs(frame, 'cpu'), first)
        self.assertEqual(first.shape[-1], 16)

    def test_root_has_swin_and_keeps_existing_e0_and_output_skip_grid(self):
        student = tiny_model(distillation_layers=['InputAdapter', 'E0', 'OutputHead'])
        self.assertIsInstance(student.layers[0].block, SwinBlock)
        self.assertTrue(all(not m.has_attention for m in student.modules()
                            if isinstance(m, InputAdapter) and m is not student.layers[0]))
        head, features = student(torch.rand(1, 39, 53, 16), return_features=True)
        self.assertEqual(features['InputAdapter'].shape, (1, 4, 32, 32))
        self.assertEqual(features['E0'].shape, (1, 4, 32, 32))
        self.assertEqual(head.shape, (1, 32, 32, 4))
        self.assertEqual(student.sources[-2], [-1, 0])
        retired = deepcopy(student.config)
        retired['backbone'][0][3] = [4]
        with self.assertRaisesRegex(ValueError, 'Swin InputAdapter'):
            EchoRenderStudent(retired)

    def test_named_head_is_selectable_and_never_duplicated_as_feature(self):
        config = load_training_config()['loss']
        config.update(rgb_lpips=0., distillation=1., mid=1., output=1.)
        sh = torch.ones(1, 8, 8, 4, requires_grad=True)
        th = torch.zeros_like(sh, requires_grad=True)
        sf = {'InputAdapter': torch.rand(1, 4, 8, 8, requires_grad=True)}
        tf = {'InputAdapter': torch.rand(1, 4, 8, 8, requires_grad=True)}
        rgb = torch.zeros(1, 3, 8, 8)
        args = (sh, th, sf, tf, rgb, rgb)
        kwargs = dict(image_hw=(8, 8), network_hw=(8, 8), clean_source=rgb)
        for include_head in (False, True):
            selected = ['InputAdapter'] + (['OutputHead'] if include_head else [])
            criterion = DistillationLoss(config, channels={'InputAdapter': 4}, stages=selected)
            loss, terms = criterion(*args, **kwargs)
            self.assertEqual(set(criterion.projections), {'InputAdapter'})
            self.assertEqual(terms['output_rgb'], float(include_head))
            self.assertEqual(terms['output_gate'], float(include_head))
            self.assertAlmostEqual(loss.item(), sum(v for k, v in terms.items() if k.startswith('weighted_')), places=6)
            loss.backward()
            self.assertIsNone(th.grad)
            self.assertIsNone(tf['InputAdapter'].grad)
        normalized = normalized_feature_mse(sf['InputAdapter'], tf['InputAdapter'])
        cosine = (2 * (1 - F.cosine_similarity(sf['InputAdapter'], tf['InputAdapter'].detach(), dim=1)) / 4).mean()
        torch.testing.assert_close(normalized, cosine)

    def test_teacher_front_tap_quantizes_skip_without_changing_forward(self):
        backend = FeatureRecordingBackend.__new__(FeatureRecordingBackend)
        nn.Module.__init__(backend)
        backend._captures = {}
        backend.stages = ('InputAdapter',)
        backend._window_names = {}
        value = torch.rand(1, 8, 8, 32, requires_grad=True)
        with patch('mlxdlss.model.NeuralRenderingModel._window', return_value=value):
            result = backend._window(value, 0, head_count=1)
        self.assertIs(result, value)
        from mlxdlss.model import e4m3_round_trip
        torch.testing.assert_close(backend._captures['InputAdapter'], e4m3_round_trip(value).permute(0, 3, 1, 2), atol=0, rtol=0)
        self.assertFalse(backend._captures['InputAdapter'].requires_grad)


if __name__ == '__main__':
    unittest.main()
