"""Best-checkpoint selection and end-of-training statistics and plots."""
import csv
import json
import math
from pathlib import Path
from statistics import fmean

from tqdm.auto import tqdm


DIRECTIONS = {'loss': -1, 'psnr': 1, 'ssim': 1, 'lpips': -1}


def validation_values(validation):
    return {'loss': validation['components']['total'], **validation['image_mean']}


def is_better(candidate, previous, metric):
    return previous is None or DIRECTIONS[metric] * candidate[metric] > DIRECTIONS[metric] * previous[metric]


def best_message(epoch, current, previous):
    changes = []
    for name, direction in DIRECTIONS.items():
        if previous is None:
            change = 'initial'
        elif not all(math.isfinite(value) for value in (current[name], previous[name])):
            change = 'N/A'
        elif previous[name] == 0:
            change = '0.00%' if current[name] == 0 else 'N/A'
        else:
            change = f'{direction * (current[name] - previous[name]) / abs(previous[name]) * 100:+.2f}%'
        changes.append(f'{name.upper() if name != "loss" else name}={change}')
    return f'[Best]: save best weight | epoch={epoch} | ' + ' | '.join(changes)


def metric_statistics(history, best_epoch):
    series = {'train_loss': [row['train_loss'] for row in history]}
    for name in DIRECTIONS:
        series[name] = [validation_values(row['validation'])[name] for row in history]
    best_index = next(i for i, row in enumerate(history) if row['epoch'] == best_epoch)
    result = {}
    for name, values in series.items():
        direction = DIRECTIONS.get(name, -1)
        index = max(range(len(values)), key=lambda i: direction * values[i])
        result[name] = {'max': max(values), 'min': min(values), 'mean': fmean(values),
                        'best': values[index], 'best_epoch': history[index]['epoch'],
                        'at_best_checkpoint': values[best_index]}
    return result


def print_statistics(statistics):
    tqdm.write('\n[Done]: Training performance (epoch means; loss = validation loss)')
    tqdm.write(f'{"Metric":<14}{"Max":>13}{"Min":>13}{"Mean":>13}{"Best":>13}{"Epoch":>8}')
    for name, values in statistics.items():
        tqdm.write(f'{name:<14}' + ''.join(f'{values[key]:>13.5f}' for key in ('max', 'min', 'mean', 'best'))
                   + f'{values["best_epoch"]:>8}')


def write_training_report(summary, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .trainer import json_safe

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    statistics = summary['statistics']
    (directory / 'statistics.json').write_text(json.dumps(json_safe({
        'reference': 'teacher_rgb', 'aggregation': 'validation image means per epoch',
        'best_epoch': summary['best_epoch'], 'best_metric': summary['best_metric'],
        'metrics': statistics}), indent=2, allow_nan=False) + '\n')
    with (directory / 'statistics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'max', 'min', 'mean', 'best', 'best_epoch', 'at_best_checkpoint'])
        for name, values in statistics.items():
            writer.writerow([name, *values.values()])
    history = summary['history']
    epochs = [row['epoch'] for row in history]
    with (directory / 'epochs.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'psnr', 'ssim', 'lpips'])
        for row in history:
            values = validation_values(row['validation'])
            writer.writerow([row['epoch'], row['train_loss'], *[values[name] for name in DIRECTIONS]])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, DIRECTIONS):
        values = [validation_values(row['validation'])[name] for row in history]
        finite = [value if math.isfinite(value) else float('nan') for value in values]
        axis.plot(epochs, finite, marker='o', label='Validation')
        if name == 'loss':
            axis.plot(epochs, [row['train_loss'] for row in history], marker='o', label='Train')
        if any(not math.isfinite(value) for value in values):
            axis.text(.02, .97, 'Infinite values omitted from plot; see statistics.json',
                      transform=axis.transAxes, va='top', fontsize=8)
        axis.axvline(summary['best_epoch'], color='gray', linestyle='--', label='Best checkpoint')
        axis.set(title={'loss': 'Loss', 'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'lpips': 'LPIPS'}[name],
                 xlabel='Epoch')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        axis.set_xticks(epochs)
    figure.suptitle('Training performance - Teacher RGB reference', fontweight='bold')
    figure.savefig(directory / 'training_metrics.png', dpi=160)
    plt.close(figure)
