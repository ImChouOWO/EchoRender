import unittest
import torch
from model.card.mamba import Mamba, MambaStage, selective_scan
from model.model import Model

class MambaTests(unittest.TestCase):
    def test_scan_known_recurrence(self):
        u = torch.tensor([[[2.], [3.]]])
        delta = torch.ones_like(u)
        A = torch.tensor([[-0.69314718056]])
        B = C = torch.ones(1, 2, 1)
        y = selective_scan(u, delta, A, B, C, torch.zeros(1))
        torch.testing.assert_close(y, torch.tensor([[[2.], [4.]]]))

    def test_scan_gradcheck(self):
        shapes = [(1, 3, 2), (1, 3, 2), (2, 2), (1, 3, 2), (1, 3, 2), (2,)]
        args = tuple(torch.randn(s, dtype=torch.float64, requires_grad=True) * 0.1 for s in shapes)
        self.assertTrue(torch.autograd.gradcheck(selective_scan, args))

    def test_causality_and_state_dict(self):
        torch.manual_seed(1)
        m = Mamba(4, d_state=3, d_conv=3)
        x = torch.randn(2, 7, 4)
        altered = x.clone(); altered[:, 4:] += 10
        torch.testing.assert_close(m(x)[:, :4], m(altered)[:, :4])
        other = Mamba(4, d_state=3, d_conv=3)
        other.load_state_dict(m.state_dict())
        torch.testing.assert_close(m(x), other(x))
        dt = torch.nn.functional.softplus(m.dt_proj.bias)
        self.assertTrue(((dt >= .001) & (dt <= .1)).all())

    def test_image_modes_backward(self):
        for scan in ('raster', 'cross'):
            stage = MambaStage(3, 4, 2, 3, 3, 2, scan)
            x = torch.randn(2, 3, 3, 5, requires_grad=True)
            y = stage(x)
            self.assertEqual(y.shape, (2, 4, 3, 5))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for p in stage.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_yaml_registry(self):
        for name in ('MambaStage', 'SSMStage'):
            cfg = {'input_channels': 3, 'output_channels': 2, 'backbone': [
                [-1, 1, name, [4, 1, 3, 3, 2, 'cross']],
                [-1, 1, 'OutputHead', [2]]], 'head': []}
            self.assertEqual(Model(cfg)(torch.randn(1, 3, 5, 3)).shape, (1, 3, 5, 2))

if __name__ == '__main__':
    unittest.main()
