import tempfile
import unittest
import torch
from torch import nn
from tool.teacher_features import TorchFeatureExtractor, TeacherFeatures

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.teacher = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        self.x = torch.randn(2, 3, requires_grad=True)
        self.extractor = TorchFeatureExtractor(self.teacher, ['0', '2'])

    def test_online_cached_equal(self):
        with tempfile.TemporaryDirectory() as d:
            p = TeacherFeatures(self.extractor, ['0', '2'], cache_dir=d, teacher_id='test')
            live = p.get(self.x)
            self.assertFalse(live['0'].requires_grad)
            self.assertEqual(p.precompute([('sample', self.x)]), 1)
            self.assertEqual(p.precompute([('sample', self.x)]), 0)
            p.set_mode('cached')
            p.extractor = None
            for k, v in p.get(self.x, sample_id='sample').items():
                torch.testing.assert_close(v, live[k])
            with self.assertRaises(ValueError):
                p.get(self.x + 1, sample_id='sample')
            with self.assertRaises(FileNotFoundError):
                p.get(self.x, sample_id='missing')

    def test_restore_and_hooks(self):
        self.teacher.train()
        self.teacher[1].eval()
        self.extractor(self.x)
        self.assertTrue(self.teacher.training)
        self.assertFalse(self.teacher[1].training)
        self.assertFalse(self.teacher[0]._forward_hooks)
        with self.assertRaises(ValueError):
            TorchFeatureExtractor(self.teacher, ['unknown'])

    def test_namespace_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = TeacherFeatures(self.extractor, ['0', '2'], cache_dir=d, teacher_id='A')
            p.precompute([('sample', self.x)])
            with self.assertRaises(ValueError):
                p.precompute([('sample', self.x + 1)])
            p.precompute([('sample', self.x + 1)], overwrite=True)
            p.set_mode('cached').get(self.x + 1, sample_id='sample')
            q = TeacherFeatures(None, ['0', '2'], mode='cached', cache_dir=d, teacher_id='B')
            with self.assertRaises(FileNotFoundError):
                q.get(self.x, sample_id='sample')

    def test_student_gradients(self):
        p = TeacherFeatures(self.extractor, ['0', '2'], teacher_id='test')
        target = p.get(self.x)['2']
        student = nn.Linear(3, 2)
        (student(self.x) - target).square().mean().backward()
        self.assertIsNotNone(student.weight.grad)
        self.assertTrue(all(v.grad is None for v in self.teacher.parameters()))

if __name__ == '__main__':
    unittest.main()
