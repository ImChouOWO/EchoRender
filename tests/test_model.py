import unittest
from copy import deepcopy
import torch
import yaml
from model.model import Model, DEFAULT_CONFIG
from model.card.card import SwinBlock

class ModelTests(unittest.TestCase):
    def small_config(self, layout='BHWC'):
        with DEFAULT_CONFIG.open(encoding='utf-8-sig') as f:
            cfg = yaml.safe_load(f)
        cfg['input_layout'] = layout
        for row in cfg['backbone'] + cfg['head']:
            if row[2] in ('SwinStage', 'ViTStage'):
                row[3][0] = max(4, row[3][0] // 8)
                row[3][1] = 2
                row[3][2] = max(1, row[3][0] // 4)
            elif row[2] != 'OutputHead' and row[3]:
                row[3][0] = max(4, row[3][0] // 8)
        return cfg

    def test_forward_backward_odd_shape(self):
        torch.manual_seed(7)
        model = Model(self.small_config())
        x = torch.randn(1, 17, 19, 16, requires_grad=True)
        y, features = model(x, return_features=True)
        self.assertEqual(y.shape, (1, 17, 19, 4))
        self.assertEqual(len(features), 28)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_layout_and_roundtrip(self):
        a = Model(self.small_config()).eval()
        b = Model(self.small_config('BCHW')).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(1, 16, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(a(x).permute(0, 3, 1, 2), b(x.permute(0, 3, 1, 2)))

    def test_repeats_and_concat(self):
        cfg = {'input_channels': 16, 'output_channels': 4, 'backbone': [
            [-1, 2, 'InputAdapter', [8]], [[-1, 0], 1, 'Concat', []],
            [-1, 1, 'OutputHead', [4]]], 'head': []}
        self.assertEqual(Model(cfg)(torch.randn(1, 3, 5, 16)).shape, (1, 3, 5, 4))

    def test_invalid_graph(self):
        for row in ([99, 1, 'InputAdapter', [4]], [-1, 0, 'InputAdapter', [4]], [-1, 1, '__import__', [4]]):
            with self.assertRaises(ValueError):
                Model({'backbone': [row]})

    def test_full_config_build(self):
        # Validate actual YAML with full widths/depths without allocating weights.
        with torch.device('meta'):
            model = Model()
        self.assertEqual(model.channels[-1], 4)
        self.assertEqual(len(model.layers), 28)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
