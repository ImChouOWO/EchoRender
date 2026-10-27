"""Standalone validation CLI integration and checkpoint-objective tests."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import val
from model.card.loss import DistillationLoss
from .training_config import load_training_config


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_rejects_unsafe_project_and_nonpositive_limit(self):
        for option in (['--project', '../escape'], ['--project', 'good', '--max-clips', '0']):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                val.main(['--weights', 'best.pt', *option])
            self.assertEqual(error.exception.code, 2)

    def test_relative_output_is_project_rooted_and_absolute_output_is_preserved(self):
        import yaml
        from tool.training_config import PROJECT_ROOT
        config = load_training_config()
        config.pop('config_path')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'train.yaml'
            config['runtime']['output_dir'] = 'output/example'
            path.write_text(yaml.safe_dump(config))
            self.assertEqual(load_training_config(path)['runtime']['output_dir'], str(PROJECT_ROOT / 'output/example'))
            absolute = str(Path(temporary) / 'custom' / 'run')
            config['runtime']['output_dir'] = absolute
            path.write_text(yaml.safe_dump(config))
            self.assertEqual(load_training_config(path)['runtime']['output_dir'], absolute)

    def test_limit_loader_bounds_number_of_samples(self):
        from torch.utils.data import DataLoader
        loader = DataLoader(list(range(8)), batch_size=4)
        limited = val.limit_loader(loader, 3)
        self.assertEqual(len(limited.dataset), 3)
        self.assertEqual(next(iter(limited)).tolist(), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
