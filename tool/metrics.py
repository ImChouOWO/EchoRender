"""Teacher-reference RGB metrics; pretrained LPIPS is never replaced by random weights."""
import torch
from torch import nn
from pytorch_msssim import ssim


class ImageMetrics(nn.Module):
    def __init__(self, *, net='alex', ssim_window=11):
        super().__init__()
        import lpips
        try:
            self.perceptual = lpips.LPIPS(net=net, verbose=False).eval().requires_grad_(False)
        except Exception as exc:
            raise RuntimeError('Pretrained LPIPS could not load. Set TORCH_HOME to a writable '
                               'cache and allow the official PyTorch model download; '
                               'random-backbone LPIPS is not a valid substitute.') from exc
        self.ssim_window = ssim_window

    @torch.no_grad()
    def per_image(self, prediction, target):
        if min(prediction.shape[-2:]) < 64:
            raise ValueError('LPIPS evaluation requires real RGB images at least 64x64')
        mse = (prediction - target).square().flatten(1).mean(1)
        psnr = -10 * torch.log10(mse)
        structural = ssim(prediction, target, data_range=1, win_size=self.ssim_window,
                          size_average=False)
        perceptual = self.perceptual(prediction * 2 - 1, target * 2 - 1).flatten(1).mean(1)
        values = torch.stack((psnr, structural, perceptual), dim=1).cpu().tolist()
        return [dict(zip(('psnr', 'ssim', 'lpips'), row)) for row in values]

    def forward(self, prediction, target):
        values = self.per_image(prediction, target)
        return {key: sum(row[key] for row in values) / len(values) for key in values[0]}
