"""Single-image contracts, plus opt-in tests using a real external checkpoint."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch

from infer import main
from model.teacher import EchoRenderTeacher
from tool.teacher_pipeline import TeacherPipeline, compose_image, load_rgb, prepare_image, save_rgb


class ImageContractTests(unittest.TestCase):
    def test_first_frame_conditions_and_noise(self):
        rgb = np.linspace(0, 1, 17 * 23 * 3, dtype=np.float32).reshape(17, 23, 3)
        first = prepare_image(rgb)
        same = prepare_image(rgb)
        changed_noise = prepare_image(rgb, noise_index=1)
        self.assertEqual(tuple(first.features.shape), (1, 320, 320, 16))
        torch.testing.assert_close(first.features, same.features, rtol=0, atol=0)
        torch.testing.assert_close(first.features[..., 4:7], first.features[..., 7:10], rtol=0, atol=0)
        self.assertTrue(torch.all(first.features[..., 3] == 1))
        self.assertTrue(torch.all(first.features[..., 13:15] == -1))
        self.assertTrue(torch.all(first.features[..., 15] == 0))
        self.assertFalse(torch.equal(first.features[..., :3], changed_noise.features[..., :3]))
        torch.testing.assert_close(first.features[..., 3:], changed_noise.features[..., 3:], rtol=0, atol=0)
        np.testing.assert_array_equal(first.rgb, rgb)

    def test_non_multiple_extent_is_padded_then_cropped(self):
        prepared = prepare_image(np.full((321, 513, 3), 0.5, dtype=np.float32))
        self.assertEqual(tuple(prepared.features.shape), (1, 384, 576, 16))
        head = torch.zeros(1, 384, 576, 4)
        output = compose_image(head, prepared, blend_scale=0.75)
        self.assertEqual(output.shape, (321, 513, 3))
        np.testing.assert_array_equal(output, prepared.rgb)

    def test_invalid_images_and_controls(self):
        for rgb in (np.zeros((0, 2, 3)), np.zeros((2, 2)), np.zeros((2, 2, 4)),
                    np.full((2, 2, 3), np.nan), np.full((2, 2, 3), 2)):
            with self.subTest(shape=rgb.shape), self.assertRaises(ValueError):
                prepare_image(rgb)
        with self.assertRaises(ValueError):
            prepare_image(np.zeros((2, 2, 3)), noise_index=-1)
        prepared = prepare_image(np.zeros((2, 2, 3)))
        head = torch.zeros(1, 320, 320, 4)
        for intensity in (-1, 2, np.nan):
            with self.subTest(intensity=intensity), self.assertRaises(ValueError):
                compose_image(head, prepared, blend_scale=0.75, intensity=intensity)
        with self.assertRaises(ValueError):
            compose_image(head[..., :3], prepared, blend_scale=0.75)
        head[..., 0] = float('nan')
        with self.assertRaises(ValueError):
            compose_image(head, prepared, blend_scale=0.75)

    def test_gate_changes_rgb_even_without_previous_image(self):
        prepared = prepare_image(np.full((2, 3, 3), 0.5, dtype=np.float32))
        head = torch.full((1, 320, 320, 4), 0.5)
        head[..., 3] = -100
        open_residual = compose_image(head, prepared, blend_scale=0.75)
        head[..., 3] = 100
        held_image = compose_image(head, prepared, blend_scale=0.75)
        np.testing.assert_allclose(open_residual, 0.625, rtol=0, atol=1e-6)
        np.testing.assert_allclose(held_image, 0.53125, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(compose_image(head, prepared, blend_scale=0.75, intensity=0), prepared.rgb)

    def test_composition_matches_pinned_reference(self):
        from mlxdlss.temporal import compose_temporal
        rng = np.random.default_rng(7)
        prepared = prepare_image(rng.random((9, 13, 3), dtype=np.float32))
        head = torch.from_numpy(rng.normal(size=(1, 320, 320, 4)).astype(np.float32))
        expected = compose_temporal(
            prepared.geometry.crop(head[0].numpy()), prepared.rgb,
            prepared.geometry.crop(prepared.features[0].numpy()), blend_scale=0.73974609375,
            intensity=0.7,
        )
        actual = compose_image(head, prepared, blend_scale=0.73974609375, intensity=0.7)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

    def test_image_io_and_grayscale_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'image.png'
            rgb = np.full((3, 5, 3), 128 / 255, dtype=np.float32)
            save_rgb(rgb, path)
            np.testing.assert_array_equal(load_rgb(path), rgb)
            with Image.open(path) as saved:
                self.assertEqual(saved.mode, 'RGB')
                self.assertEqual(saved.size, (5, 3))
            gray = Path(directory) / 'gray.png'
            Image.fromarray(np.full((4, 6), 64, dtype=np.uint8)).save(gray)
            self.assertEqual(load_rgb(gray).shape, (4, 6, 3))

    def test_cli_protects_input_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            path.write_bytes(b'original')
            arguments = ['--input', str(path), '--weights', 'unused.safetensors', '--output', str(path)]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), b'original')
            output = Path(directory) / 'existing.png'
            output.write_bytes(b'existing')
            arguments[-1] = str(output)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(arguments)
            self.assertEqual(output.read_bytes(), b'existing')

    def test_missing_and_packed_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                EchoRenderTeacher(Path(directory) / 'missing.safetensors', device='cpu')
            path = Path(directory) / 'packed.safetensors'
            save_file({'block0.layer0.layer': torch.zeros(8, dtype=torch.uint8)}, str(path),
                      metadata={'format': 'dlssnr-packed'})
            with self.assertRaisesRegex(ValueError, 'logical'):
                EchoRenderTeacher(path, device='cpu')


@unittest.skipUnless(os.environ.get('ECHORENDER_TEACHER_WEIGHTS'), 'Set ECHORENDER_TEACHER_WEIGHTS for real checkpoint tests')
class RealTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = EchoRenderTeacher(os.environ['ECHORENDER_TEACHER_WEIGHTS'])

    def test_real_inference_is_frozen_stateless_and_usable_as_target(self):
        rgb = np.linspace(0, 1, 32 * 40 * 3, dtype=np.float32).reshape(32, 40, 3)
        pipeline = TeacherPipeline(self.teacher)
        self.teacher.train()
        self.assertFalse(self.teacher.training)
        self.assertFalse(self.teacher.model.training)
        first = pipeline.enhance(rgb)
        pipeline.enhance(np.ones_like(rgb))
        again = pipeline.enhance(rgb)
        torch.testing.assert_close(first.raw_head, again.raw_head, rtol=0, atol=0)
        np.testing.assert_array_equal(first.image, again.image)
        self.assertEqual(first.image.shape, rgb.shape)
        self.assertTrue(np.isfinite(first.image).all())
        self.assertGreater(float(np.abs(first.image - rgb).mean()), 1e-5)
        self.assertFalse(first.raw_head.requires_grad)
        # A real Teacher target must be usable by autograd in the next project step.
        student_head = torch.nn.Conv2d(16, 4, 1)
        predicted = student_head(prepare_image(rgb).features.permute(0, 3, 1, 2))
        loss = torch.nn.functional.mse_loss(predicted, first.raw_head.permute(0, 3, 1, 2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(student_head.weight.grad).all())
        self.assertGreater(student_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(not value.requires_grad for value in self.teacher.model.buffers()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
