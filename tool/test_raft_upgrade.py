"""Contracts and real CUDA checks for the RAFT/encoder-distillation upgrade."""
from copy import deepcopy
import os
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from model.student import EchoRenderStudent
from model.card.loss import DistillationLoss
from model.card.mamba import selective_scan
from model.motion import RaftSmallMotion
from .training_config import load_training_config
from .temporal import prepare_frame, teacher_target
from .test_training import tiny_model


class UpgradeContracts(unittest.TestCase):
    def test_selected_encoder_stages_and_direct_adapter_skip(self):
        config = load_training_config()
        self.assertEqual(config['training']['distillation_layers'],
                         ['InputAdapter', 'E0', 'E1', 'E2', 'E3', 'E4', 'VIT', 'OutputHead', 'RGB composition'])
        model = tiny_model(distillation_layers=config['training']['distillation_layers'])
        self.assertEqual(model.sources[-2], [-1, 0])
        self.assertEqual(set(model.feature_layers), {'InputAdapter', 'E0', 'E1', 'E2', 'E3', 'E4', 'VIT'})
        # Isolate the direct adapter path from all other graph paths.
        captured = {}
        hook = model.layers[0].register_forward_hook(lambda module, args, output: captured.update(adapter=output))
        head, features = model(torch.randn(1, 32, 32, 16), return_features=True)
        hook.remove()
        captured['adapter'].retain_grad()
        head.square().mean().backward()
        self.assertGreater(float(captured['adapter'].grad.abs().sum()), 0.)
        self.assertEqual(head.shape, (1, 32, 32, 4))

    def test_rgb_composition_selection_controls_rgb_loss(self):
        config = load_training_config()['loss']
        config.update(rgb_lpips=0., ssim_window=3, ms_ssim_weights=[1.])
        args = (torch.zeros(1, 64, 64, 4, requires_grad=True), torch.zeros(1, 64, 64, 4),
                {}, {}, torch.ones(1, 3, 64, 64, requires_grad=True) * .5, torch.zeros(1, 3, 64, 64))
        criterion = DistillationLoss(config, stages=['RGB composition'], channels={}, student_channels={})
        loss, parts = criterion(*args, image_hw=(64, 64), network_hw=(64, 64), clean_source=args[-1])
        self.assertGreater(parts['rgb_mse'], 0.)
        self.assertEqual(parts['mid'], 0.)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(criterion.supervise_rgb)


@unittest.skipUnless(os.environ.get('ECHORENDER_GPU_TESTS') == '1' and torch.cuda.is_available(),
                     'Set ECHORENDER_GPU_TESTS=1; tests use physical GPU 1')
class UpgradeCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
        torch.cuda.set_device(cls.device)
        torch.set_num_threads(4)

    def tearDown(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_fused_scan_matches_reference_forward_and_all_gradients(self):
        torch.manual_seed(12)
        shapes = ((2, 19, 8), (2, 19, 8), (8, 4), (2, 19, 4), (2, 19, 4), (8,))
        values = [torch.randn(shape) for shape in shapes]
        values[1] = values[1].sigmoid() * .1
        values[2] = -values[2].abs()
        cpu = [x.clone().requires_grad_() for x in values]
        cuda = [x.to(self.device).requires_grad_() for x in values]
        expected, actual = selective_scan(*cpu), selective_scan(*cuda)
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-4)
        weight = torch.randn_like(expected)
        (expected * weight).sum().backward()
        (actual * weight.to(self.device)).sum().backward()
        for reference, candidate in zip(cpu, cuda):
            torch.testing.assert_close(candidate.grad.cpu(), reference.grad, atol=4e-5, rtol=5e-4)

    def test_real_raft_small_backward_flow_supervision_and_bf16(self):
        model = RaftSmallMotion(128).to(self.device).train()
        previous = torch.rand(1, 3, 128, 128, device=self.device)
        current = previous.roll(2, dims=-1)
        seen = []
        hook = model.raft.register_forward_pre_hook(lambda module, args: seen.extend([x.detach().clone() for x in args]))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            field, logits = model(previous, current)
        hook.remove()
        torch.testing.assert_close(seen[0], current * 2 - 1)
        torch.testing.assert_close(seen[1], previous * 2 - 1)
        self.assertEqual(field.shape, (1, 2, 128, 128))
        self.assertEqual(logits.shape, (1, 1, 128, 128))
        target = torch.zeros_like(field)
        target[:, 0] = -2 / 128
        loss = F.smooth_l1_loss(field * 128, target * 128) + F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.raft.update_block.flow_head.conv2.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.confidence[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(model.raft.corr_block.corr_pyramid, [])

    def test_real_teacher_captures_all_encoder_targets(self):
        from model.teacher import EchoRenderTeacher
        config = load_training_config()
        selected = config['training']['distillation_layers']
        teacher = EchoRenderTeacher(config['model']['teacher_weights'], device=self.device, stages=selected)
        rgb = np.full((320, 320, 3), .4, dtype=np.float32)
        with torch.inference_mode():
            target = teacher_target(teacher, prepare_frame(rgb, config, 'neutral'))
        self.assertEqual({k: v.shape[1] for k, v in target['features'].items()},
                         dict(InputAdapter=32, E0=32, E1=64, E2=128, E3=256, E4=512, VIT=1024))
        self.assertEqual(target['features']['E0'].shape[-2:], (160, 160))
        self.assertEqual(target['rgb'].shape, (1, 3, 320, 320))
        self.assertTrue(all(not v.requires_grad for v in target['features'].values()))

    def test_real_student_bf16_encoder_features_and_checkpoint_contract(self):
        config = load_training_config()
        model = EchoRenderStudent(config['model']['student_config'], precision='bf16',
            distillation_layers=config['training']['distillation_layers']).to(self.device).eval()
        with torch.inference_mode():
            head, features = model(torch.rand(1, 320, 320, 16, device=self.device), return_features=True)
        self.assertEqual(head.dtype, torch.bfloat16)
        self.assertEqual(model.sources[27], [-1, 0])
        self.assertEqual({k: v.shape[1] for k, v in features.items()},
                         dict(InputAdapter=32, E0=32, E1=32, E2=64, E3=128, E4=512, VIT=1024))
        self.assertTrue(all(torch.isfinite(v).all() for v in features.values()))
        restored = EchoRenderStudent(model.config)
        restored.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
        self.assertEqual(restored.precision, 'bf16')
        self.assertEqual(restored.motion_mode, 'raft_small')


if __name__ == '__main__':
    unittest.main()
