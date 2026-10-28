"""Train the SSM Student; output paths resolve from the project root."""
import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch

from model.student import EchoRenderStudent
from model.teacher import EchoRenderTeacher, resolve_device
from model.card.loss import DistillationLoss
from tool.dataloader import build_dataloader
from tool.training_config import load_training_config, DEFAULT_TRAIN_CONFIG, reserve_output_dir, PROJECT_ROOT, selected_data_config
from tool.metrics import ImageMetrics
from tool.trainer import fit, load_student_state
from tool.resume import read_checkpoint, validate_resume
from tool.teacher_features import ask_delete_cache
from tool.prediction_report import export_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument('--resume', type=Path, help='Resume an unfinished run from its last.pt checkpoint')
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config)
        resume = None
        if args.resume is not None:
            path = args.resume.expanduser()
            path = (PROJECT_ROOT / path).resolve()
            resume = read_checkpoint(path)
            if config['model']['student_weights']:
                raise ValueError('Use either --resume or model.student_weights, not both')
            config['runtime']['output_dir'] = str(path.parent)
        data = selected_data_config(config)
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
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        def create_teacher():
            teacher = EchoRenderTeacher(config['model']['teacher_weights'], device='cpu')
            if teacher.model.topology.name != 'ht':
                raise ValueError('This student targets the HT Teacher (71 blocks), not small/custom topology')
            return teacher
        student = EchoRenderStudent(config['model']['student_config'])
        student_channels = {name: student.channels[index] for name, index in student.FEATURE_LAYERS.items()}
        if config['model']['student_weights']:
            load_student_state(student, config['model']['student_weights'])
        if resume is not None:
            validate_resume(resume, config, student)
        criterion = DistillationLoss(config['loss'], student_channels=student_channels)
        optimizer = torch.optim.SGD(list(student.parameters()) + list(criterion.parameters()),
                                    **config['training']['optimizer'])
        # Keep official pretrained LPIPS downloads in an existing ignored cache.
        os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent / '.cache/torch'))
        metrics = ImageMetrics(net=config['validation']['lpips_net'],
                               ssim_window=config['loss']['ssim_window'])
        output = output if resume is not None else reserve_output_dir(output)
        config['runtime']['output_dir'] = str(output)
        print(f'Output: {output}', flush=True)
        print(f'Training on {device}; train clips/profiles={len(train_loader.dataset)}, val={len(val_loader.dataset)}', flush=True)
        print(f'Student motion: {student.motion_mode} | profiles: ' + ', '.join(config['training']['profiles']), flush=True)
        print(f"Batch: effective={config['loader']['batch_size']} clips, micro={config['training']['micro_batch_size']} concurrent frames", flush=True)
        report_dir = None
        on_complete = None
        if config['reporting']['enabled']:
            project_name = config['reporting']['project_name'] or output.name
            previous_report = (resume.get('resume_state') or {}).get('report_dir') if resume is not None else None
            report_dir = Path(previous_report) if previous_report else output.parent / 'val' / project_name
            if resume is None:
                report_dir = reserve_output_dir(report_dir)
            print(f'Reports: {report_dir}', flush=True)
            def on_complete(summary):
                loaders, missing = {'val': val_loader}, {}
                try:
                    loaders['test'] = build_dataloader(data, split='test')
                except (FileNotFoundError, ValueError) as exc:
                    missing['test'] = str(exc)
                    print(f'[Report]: Test split unavailable: {exc}', flush=True)
                export_predictions(summary['best_checkpoint'], create_teacher, loaders, config, report_dir,
                                   device, missing_splits=missing)
        fit(create_teacher, student, criterion, train_loader, val_loader, optimizer, metrics, config,
            device=device, cache_cleanup=ask_delete_cache, report_dir=report_dir, on_complete=on_complete, resume=resume)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'Training failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
