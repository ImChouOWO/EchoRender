"""Fixed-column training progress, with device-wide utilization and CUDA memory."""
import torch
from tqdm.auto import tqdm


BAR_FORMAT = '{desc} {bar:10} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'


def progress_bar(loader, *, enabled, validation=False):
    columns = ('Epoch', 'GPU_mem', 'GPU', 'Loss', 'PSNR', 'SSIM', 'LPIPS', 'Images') if validation else (
        'Epoch', 'GPU_mem', 'GPU', 'Loss', 'AvgLoss', 'LR', 'Images', 'Size')
    if enabled:
        tqdm.write(''.join(f'{name:>11}' for name in columns))
    return tqdm(loader, bar_format=BAR_FORMAT, dynamic_ncols=True, disable=not enabled)


def progress_row(epoch, epochs, device, monitor, values):
    device = torch.device(device)
    memory = f'{torch.cuda.memory_reserved(device) / 2**30:.2f}G' if device.type == 'cuda' else '0G'
    fields = [f'{epoch}/{epochs}', memory, monitor.display(), *values]
    return ''.join(f'{str(value):>11}' for value in fields)


def network_size(image, crop_size, rendering):
    from teacher_infer import network_geometry, render_extent
    h, w = [min(crop_size, size) if crop_size else size for size in image.shape[-2:]]
    w, h = render_extent(w, h, rendering['render_scale'], rendering['min_extent'])
    geometry = network_geometry(w, h, rendering['min_extent'])
    h, w = geometry.network_height, geometry.network_width
    return str(h) if h == w else f'{h}x{w}'
