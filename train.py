"""Train the SSM student using config/train.yaml (all paths relative to YAML)."""
import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch

from model.student import EchoRenderStudent
from model.teacher import EchoRenderTeacher, resolve_device
from model.card.loss import DistillationLoss
from tool.dataloader import load_data_config, build_dataloader
from tool.training_config import load_training_config, DEFAULT_TRAIN_CONFIG, reserve_output_dir
from tool.metrics import ImageMetrics
from tool.trainer import fit, load_student_state
from tool.teacher_features import ask_delete_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_TRAIN_CONFIG)
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config)
        data = load_data_config(config)
        output = Path(config['runtime']['output_dir'])
        if output.is_relative_to(data.root):
            raise ValueError('output_dir must be outside the dataset')
        if not Path(config['model']['teacher_weights']).is_file():
            raise FileNotFoundError(f"Set model.teacher_weights to a logical checkpoint: {config['model']['teacher_weights']}")
        torch.set_num_threads(config['runtime']['cpu_threads'])
        random.seed(data.seed)
        np.random.seed(data.seed % 2**32)
        torch.manual_seed(data.seed)
        train_loader = build_dataloader(data, split='train')
        val_loader = build_dataloader(data, split='val')
        device = resolve_device(config['runtime']['device'])
        def create_teacher():
            teacher = EchoRenderTeacher(config['model']['teacher_weights'], device='cpu')
            if teacher.model.topology.name != 'ht':
                raise ValueError('This student targets the HT Teacher (71 blocks), not small/custom topology')
            return teacher
        student = EchoRenderStudent(config['model']['student_config'])
        if student.input_channels != 16 or student.channels[-1] != 4:
            raise ValueError('Student must have 16 input and 4 output channels')
        student_channels = {name: student.channels[index] for name, index in student.FEATURE_LAYERS.items()}
        if config['model']['student_weights']:
            load_student_state(student, config['model']['student_weights'])
        criterion = DistillationLoss(config['loss'], student_channels=student_channels)
        optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()),
                                    **config['training']['optimizer'])
        # Keep official pretrained LPIPS downloads in an existing ignored cache.
        os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent / '.cache/torch'))
        metrics = ImageMetrics(net=config['validation']['lpips_net'],
                               ssim_window=config['loss']['ssim_window'])
        output = reserve_output_dir(output)
        config['runtime']['output_dir'] = str(output)
        print(f'Output: {output}', flush=True)
        print(f'Training on {device}; train={len(train_loader.dataset)}, val={len(val_loader.dataset)}', flush=True)
        print(f"Batch: effective={config['loader']['batch_size']}, micro={config['training']['micro_batch_size']}", flush=True)
        fit(create_teacher, student, criterion, train_loader, val_loader, optimizer, metrics, config, device=device, cache_cleanup=ask_delete_cache)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'Training failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
