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
from .test_training import tiny_model, FixtureTeacher, identity_prepare
from .training_config import load_training_config


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        config = load_training_config()
        config['training']['crop_size'] = 32
        config['loss'].update(ssim_window=3, ms_ssim_weights=[.3, .3, .4])
        config['logging']['progress'] = False
        student = tiny_model()
        criterion = DistillationLoss(config['loss'], channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4))
        payload = {'format': 'echorender-ssm-v2', 'student': student.state_dict(),
                   'distillation': criterion.state_dict(), 'student_config': student.config,
                   'train_config': deepcopy(config), 'epoch': 2, 'teacher_blend_scale': .5}
        batch = {'images': [torch.rand(3, 32, 32)], 'sample_ids': ['training/clean/scene/frame_0001'],
                 'scenes': ['scene']}
        return config, payload, batch

    def test_clean_validation_uses_checkpoint_model_and_projections(self):
        config, payload, batch = self.fixture()
        class Metrics(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, prediction, target):
                return {'psnr': 20., 'ssim': .7, 'lpips': .3}
        with patch('val.ImageMetrics', Metrics), \
             patch('val.DistillationLoss', side_effect=lambda config, **kw: DistillationLoss(
                 config, channels=dict.fromkeys(('e1', 'e2', 'e3', 'vit'), 4), **kw)), \
             patch('tool.distillation.prepare_image', identity_prepare), redirect_stdout(io.StringIO()):
            results = val.evaluate_checkpoint(payload, FixtureTeacher, {'val': [batch]}, config, torch.device('cpu'))
        self.assertEqual(results['val']['images'], 1)
        self.assertEqual(results['val']['image_mean'], {'psnr': 20., 'ssim': .7, 'lpips': .3})
        self.assertGreater(results['val']['components']['total'], 0)

    def test_cli_names_output_and_records_missing_test_and_selected_epoch(self):
        config, payload, batch = self.fixture()
        payload['train_config']['augmentation']['noise_std'] = .01
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / 'last.pt'
            torch.save(payload, weights)
            config['runtime']['output_dir'] = str(root / 'output' / 'train')
            config['model']['teacher_weights'] = str(root / 'teacher.safetensors')
            Path(config['model']['teacher_weights']).touch()
            def loader(data, *, split):
                if split == 'test':
                    raise FileNotFoundError('test/clean missing')
                return [batch]
            common = [patch('val.PROJECT_ROOT', root), patch('val.load_training_config', return_value=config),
                      patch('val.build_dataloader', side_effect=loader),
                      patch('val.evaluate_checkpoint', return_value={'val': {'image_mean': {'psnr': 20.}}}),
                      patch('val.export_predictions', return_value={'status': 'completed'})]
            with common[0], common[1], common[2], common[3], common[4] as exporter, redirect_stdout(io.StringIO()):
                for _ in range(2):
                    self.assertEqual(val.main(['--weights', str(weights), '--project', 'demo', '--device', 'cpu', '--imgsz', '64']), 0)
            for folder in ('demo', 'demo_2'):
                result = json.loads((root / 'output/val' / folder / 'validation.json').read_text())
                self.assertEqual(result['checkpoint_epoch'], 2)
                self.assertEqual(result['comparison_inference_size'], 64)
                self.assertIn('test', result['missing_splits'])
                self.assertIsNone(result['history_source'])
            self.assertEqual(exporter.call_count, 2)
            self.assertEqual(exporter.call_args.args[3]['augmentation']['noise_std'], .20)
            self.assertEqual(exporter.call_args.kwargs['missing_splits'], {'test': 'test/clean missing'})

    def test_cli_rejects_unsafe_project_and_nonpositive_limit(self):
        for option in (['--project', '../escape'], ['--project', 'good', '--max-images', '0']):
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
