"""Differentiable Lab, overlapping-ROI and frequency supervision of composed RGB."""
import torch
from torch import nn
from torch.nn import functional as F


APPEARANCE_TERMS = ('lab_ab', 'lab_l', 'roi_contrast', 'roi_l', 'roi_c', 'roi_s',
                    'low_frequency', 'high_frequency')
APPEARANCE_SCORES = ('roi_l_score', 'roi_c_score', 'roi_s_score',
                     'lab_l_dark_mae', 'lab_l_bright_mae', 'lab_l_dark_fraction')


def rgb_to_lab(rgb):
    """Nonlinear sRGB [0,1] -> D65/2-degree L*, a*, b*, computed in FP32."""
    rgb = rgb.float()
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).clamp_min(0).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor([[.412453, .357580, .180423],
                            [.212671, .715160, .072169],
                            [.019334, .119193, .950227]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    xyz = xyz / rgb.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    threshold = (6 / 29) ** 3
    curve = torch.where(xyz > threshold, xyz.clamp_min(threshold).pow(1 / 3),
                        xyz * (29 / 6) ** 2 / 3 + 4 / 29)
    x, y, z = curve.unbind(1)
    return torch.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), 1)


def roi_means(value, size, stride):
    """Pool without materializing patches; append boundary-aligned ROIs for full coverage."""
    height, width = value.shape[-2:]
    if min(height, width) < size:
        raise ValueError(f'Appearance loss requires image height/width >= ROI size {size}')

    def columns(rows):
        result = F.avg_pool2d(rows, size, stride)
        if (width - size) % stride:
            result = torch.cat((result, F.avg_pool2d(rows[..., -size:], size, stride)), -1)
        return result

    result = columns(value)
    if (height - size) % stride:
        result = torch.cat((result, columns(value[..., -size:, :])), -2)
    return result


class AppearanceLoss(nn.Module):
    """Lab MAE uses L*/100 and a/b/128; ROI scores use normalized L*.

    Each ROI contributes equally within an image. Frequency losses are RGB L1,
    while dark/bright diagnostic MAE retains L* units. No spatial importance mask
    is applied here; the existing reconstruction loss owns Teacher-change weights.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        radius = config['gaussian_size'] // 2
        positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-positions.square() / (2 * config['gaussian_sigma'] ** 2))
        self.register_buffer('kernel', (kernel / kernel.sum()).view(1, 1, 1, -1), persistent=False)

    def frequency_bands(self, rgb):
        radius = self.config['gaussian_size'] // 2
        kernel = self.kernel.to(device=rgb.device, dtype=rgb.dtype).expand(rgb.shape[1], 1, 1, -1)
        low = F.conv2d(F.pad(rgb, (radius, radius, 0, 0), mode='replicate'), kernel, groups=rgb.shape[1])
        low = F.conv2d(F.pad(low, (0, 0, radius, radius), mode='replicate'), kernel.transpose(-1, -2), groups=rgb.shape[1])
        return low, rgb - low

    def roi_terms(self, prediction, target):
        # Center before second moments to keep flat regions and their gradients stable.
        centers = [value.mean((-2, -1), keepdim=True) for value in (prediction, target)]
        a, b = prediction - centers[0], target - centers[1]
        moments = roi_means(torch.cat((a, b, a.square(), b.square(), a * b), 1),
                            self.config['roi_size'], self.config['roi_stride'])
        ma, mb, aa, bb, ab = moments.chunk(5, 1)
        va, vb = (aa - ma.square()).clamp_min(0), (bb - mb.square()).clamp_min(0)
        sa, sb = va.clamp_min(1e-12).sqrt(), vb.clamp_min(1e-12).sqrt()
        covariance = ab - ma * mb
        mu_a, mu_b = ma + centers[0], mb + centers[1]
        l = ((2 * mu_a * mu_b + .01**2) / (mu_a.square() + mu_b.square() + .01**2)).clamp(0, 1)
        c = ((2 * sa * sb + .03**2) / (va + vb + .03**2)).clamp(0, 1)
        s = ((covariance + .03**2 / 2) / (sa * sb + .03**2 / 2)).clamp(-1, 1)
        return {'roi_contrast': (sa - sb).abs().mean(),
                'roi_l': (1 - l).mean(), 'roi_c': (1 - c).mean(), 'roi_s': (1 - s).mean(),
                'roi_l_score': l.mean().detach(), 'roi_c_score': c.mean().detach(),
                'roi_s_score': s.mean().detach()}

    def forward(self, prediction, target):
        # Never transform the augmented source or raw head here: both arguments are final RGB.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction, target = prediction.float(), target.detach().float()
            p_lab, t_lab = rgb_to_lab(prediction), rgb_to_lab(target)
            p_l, t_l = p_lab[:, :1] / 100, t_lab[:, :1] / 100
            terms = {'lab_ab': ((p_lab[:, 1:] - t_lab[:, 1:]) / 128).abs().mean(),
                     'lab_l': (p_l - t_l).abs().mean(), **self.roi_terms(p_l, t_l)}
            low_p, high_p = self.frequency_bands(prediction)
            low_t, high_t = self.frequency_bands(target)
            terms.update(low_frequency=F.l1_loss(low_p, low_t), high_frequency=F.l1_loss(high_p, high_t))
            # Teacher L* fixes membership; empty bright/dark groups report zero, never NaN.
            error = (p_lab[:, :1] - t_lab[:, :1]).detach().abs()
            dark = t_lab[:, :1] < self.config['brightness_split']
            for name, mask in (('dark', dark), ('bright', ~dark)):
                counts = mask.sum((-2, -1))
                means = (error * mask).sum((-2, -1)) / counts.clamp_min(1)
                terms[f'lab_l_{name}_mae'] = (means.sum() / (counts > 0).sum().clamp_min(1)).detach()
            terms['lab_l_dark_fraction'] = dark.float().mean().detach()
            return terms
