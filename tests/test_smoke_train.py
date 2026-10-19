import unittest

import torch
from torch import nn

from model.student import EchoRenderStudent
from smoke_train import DEFAULT_STUDENT_CONFIG, run_smoke


class FixedTeacher(nn.Module):
    """Small deterministic target for testing the step's control flow only.

    Actual checkpoint execution is checked separately by the real smoke CLI.
    """
    def __init__(self, valid_size=None):
        super().__init__()
        self.register_buffer('gain', torch.tensor(1.0))
        self.valid_size = valid_size

    @property
    def device(self):
        return self.gain.device

    @torch.no_grad()
    def forward(self, features):
        if self.valid_size:
            head = features.new_full((*features.shape[:-1], 4), 100.0)
            height, width = self.valid_size
            head[:, :height, :width] = 1.0
            return head
        return features[..., :4] * self.gain


def small_student(layout='BHWC'):
    return EchoRenderStudent({'input_channels': 16, 'output_channels': 4, 'input_layout': layout,
                              'backbone': [[-1, 1, 'InputAdapter', [8]], [-1, 1, 'OutputHead', [4]]]})


class SmokeStepTests(unittest.TestCase):
    def test_ragged_batch_one_update_both_layouts(self):
        for layout in ('BHWC', 'BCHW'):
            with self.subTest(layout=layout):
                teacher, student = FixedTeacher(), small_student(layout)
                batch = {'images': [torch.rand(3, 11, 13), torch.rand(3, 7, 9)], 'sample_ids': ['a', 'b']}
                result = run_smoke(teacher, student, batch)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['optimizer_steps'], 1)
                self.assertEqual(result['batch_size'], 2)
                self.assertTrue(result['teacher_unchanged'])
                self.assertTrue(result['teacher_frozen'])
                self.assertTrue(result['student_parameters_updated'])
                self.assertEqual(result['student_gradient_tensors'], len(list(student.parameters())))

    def test_padding_is_excluded_from_loss(self):
        student = small_student()
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.zero_()
        batch = {'images': [torch.full((3, 11, 13), 0.5)], 'sample_ids': ['valid-area']}
        result = run_smoke(FixedTeacher(valid_size=(11, 13)), student, batch)
        self.assertEqual(result['loss'], 1.0)

    def test_default_smoke_student_preserves_graph_and_interface(self):
        with torch.device('meta'):
            student = EchoRenderStudent(DEFAULT_STUDENT_CONFIG)
        self.assertEqual(len(student.layers), 28)
        self.assertEqual(student.input_channels, 16)
        self.assertEqual(student.channels[-1], 4)
        self.assertLess(sum(parameter.numel() for parameter in student.parameters()), 1_000_000)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
