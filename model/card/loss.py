"""All training losses; Teacher targets are always detached.

Lmid=(E1+E2+E3+ViT)/4, Ld=.5 Lmid+.5 Lout, Ltotal=3 Ld+Lrgb.
Lrgb=.5 MSE+.2(1-SSIM)+.3(1-MS-SSIM), on composed RGB in [0,1].
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ssim, ms_ssim

STAGE_CHANNELS = {'e1': 64, 'e2': 128, 'e3': 256, 'vit': 1024}
TEACHER_STRIDES = {'e1': 4, 'e2': 8, 'e3': 16, 'vit': 64}


def align_features(student, teacher, *, image_hw, network_hw, stride):
    """Resize student onto Teacher's physical grid; crop internal end padding.

    Teacher's split/global stages include extra end padding. Resizing directly
    to the full captured tensor would spatially stretch real image content.
    Edge cells intersecting real pixels are retained; padding-only cells omitted.
    """
    if student.ndim != 4 or teacher.ndim != 4 or student.shape[:2] != teacher.shape[:2]:
        raise ValueError('Feature alignment expects matching BCHW batch/channels')
    physical_hw = tuple(math.ceil(size / stride) for size in network_hw)
    valid_hw = tuple(math.ceil(size / stride) for size in image_hw)
    if any(t < p for t, p in zip(teacher.shape[-2:], physical_hw)):
        raise ValueError('Teacher capture is smaller than its documented spatial grid')
    student = F.interpolate(student, size=physical_hw, mode='bilinear', align_corners=False)
    h, w = valid_hw
    return student[..., :h, :w], teacher.detach()[..., :h, :w]


def normalized_feature_mse(student, teacher, eps=1e-6):
    return F.mse_loss(F.normalize(student, dim=1, eps=eps),
                      F.normalize(teacher.detach(), dim=1, eps=eps))


def token_affinity_loss(student, teacher, *, eps=1e-6, chunk_size=128):
    """Exact cosine-affinity MSE, excluding diagonal, using bounded row chunks."""
    s = F.normalize(student.flatten(2).transpose(1, 2), dim=-1, eps=eps)
    t = F.normalize(teacher.detach().flatten(2).transpose(1, 2), dim=-1, eps=eps)
    batch, tokens, _ = s.shape
    if tokens < 2:
        return s.sum() * 0  # no off-diagonal relations for a single token
    total = s.new_zeros(())
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        difference = s[:, start:end] @ s.transpose(1, 2) - t[:, start:end] @ t.transpose(1, 2)
        rows = torch.arange(end - start, device=s.device)
        cols = torch.arange(start, end, device=s.device)
        mask = torch.ones_like(difference)
        mask[:, rows, cols] = 0
        total = total + (difference.square() * mask).sum()
    return total / (batch * tokens * (tokens - 1))


class DistillationLoss(nn.Module):
    def __init__(self, config, channels=None, *, student_channels=None):
        super().__init__()
        self.config = dict(config)
        channels = channels or STAGE_CHANNELS
        student_channels = channels if student_channels is None else student_channels
        if set(student_channels) != set(channels):
            raise ValueError('Student and Teacher feature stage names must match')
        self.projections = nn.ModuleDict()
        for name, width in channels.items():
            source_width = student_channels[name]
            if type(source_width) is not int or source_width < 1:
                raise ValueError(f'{name} student feature width must be a positive integer')
            if not config['projection'] and source_width != width:
                raise ValueError(f'{name}: loss.projection must be true for different Student/Teacher widths')
            if config['projection']:
                projection = nn.Conv2d(source_width, width, 1, bias=False)
                with torch.no_grad():
                    projection.weight.copy_(torch.eye(width, source_width).reshape(width, source_width, 1, 1))
                self.projections[name] = projection
            else:
                self.projections[name] = nn.Identity()

    def image_terms(self, prediction, target):
        c = self.config
        target = target.detach()
        levels = len(c['ms_ssim_weights'])
        minimum = (c['ssim_window'] - 1) * 2 ** (levels - 1)
        if min(prediction.shape[-2:]) <= minimum:
            raise ValueError(f'MS-SSIM needs real image height/width > {minimum}; '
                             'use larger images or explicitly reduce scales/window in train.yaml')
        # pytorch-msssim checks the five-scale minimum even for fewer weights.
        # Multi-scale computation below uses its official SSIM primitive and
        # the same pooling/product definition, supporting configured scale count.
        structural = ssim(prediction, target, data_range=1, win_size=c['ssim_window'],
                          size_average=True)
        multiscale = self._multiscale(prediction, target)
        return {'rgb_mse': F.mse_loss(prediction, target),
                'rgb_ssim': 1 - structural, 'rgb_ms_ssim': 1 - multiscale}

    def _multiscale(self, prediction, target):
        # Use the public implementation for the standard five-scale definition.
        c = self.config
        if len(c['ms_ssim_weights']) == 5:
            return ms_ssim(prediction, target, data_range=1, size_average=True,
                           win_size=c['ssim_window'], weights=c['ms_ssim_weights'])
        from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
        window = _fspecial_gauss_1d(c['ssim_window'], 1.5).to(prediction)
        window = window.repeat(prediction.shape[1], 1, 1, 1)
        values = []
        for level in range(len(c['ms_ssim_weights'])):
            score, contrast = _ssim(prediction, target, data_range=1, win=window, size_average=False)
            values.append(torch.relu(score if level == len(c['ms_ssim_weights']) - 1 else contrast))
            if level < len(c['ms_ssim_weights']) - 1:
                padding = [size % 2 for size in prediction.shape[-2:]]
                prediction = F.avg_pool2d(prediction, 2, padding=padding)
                target = F.avg_pool2d(target, 2, padding=padding)
        weights = prediction.new_tensor(c['ms_ssim_weights']).view(-1, 1, 1)
        return torch.prod(torch.stack(values) ** weights, dim=0).mean()

    def forward(self, student_head, teacher_head, student_features, teacher_features,
                prediction_rgb, target_rgb, *, image_hw, network_hw, weights=None, tensor_terms=False):
        c = self.config if weights is None else {**self.config, **weights}
        terms = {}
        for name, stride in TEACHER_STRIDES.items():
            s, t = align_features(self.projections[name](student_features[name]), teacher_features[name],
                                  image_hw=image_hw, network_hw=network_hw, stride=stride)
            if name == 'vit':
                terms['vit_feature'] = F.mse_loss(s, t)
                terms['vit_affinity'] = token_affinity_loss(
                    s, t, eps=c['norm_eps'], chunk_size=c['affinity_chunk_size'])
                terms['vit'] = c['vit_feature'] * terms['vit_feature'] + c['vit_affinity'] * terms['vit_affinity']
            else:
                terms[name] = normalized_feature_mse(s, t, c['norm_eps'])
        h, w = image_hw
        s, t = student_head[:, :h, :w], teacher_head.detach()[:, :h, :w]
        terms['output'] = F.mse_loss(s, t)
        terms['output_rgb'] = F.mse_loss(s[..., :3], t[..., :3])
        terms['output_gate'] = F.mse_loss(s[..., 3:], t[..., 3:])
        terms['mid'] = sum(c[name] * terms[name] for name in TEACHER_STRIDES) / 4
        terms['distillation'] = c['mid'] * terms['mid'] + c['output'] * terms['output']
        terms.update(self.image_terms(prediction_rgb, target_rgb))
        terms['reconstruction'] = sum(c[name] * terms[name] for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim'))
        terms['total'] = c['distillation'] * terms['distillation'] + c['reconstruction'] * terms['reconstruction']
        # Retain both raw subterms and effective contributions for diagnosis.
        for name in ('e1', 'e2', 'e3'):
            terms[f'weighted_{name}'] = c['distillation'] * c['mid'] * c[name] * terms[name] / 4
        for name in ('vit_feature', 'vit_affinity'):
            terms[f'weighted_{name}'] = c['distillation'] * c['mid'] * c['vit'] * c[name] * terms[name] / 4
        terms['weighted_output'] = c['distillation'] * c['output'] * terms['output']
        for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim'):
            terms[f'weighted_{name}'] = c['reconstruction'] * c[name] * terms[name]
        detached = {name: value.detach() for name, value in terms.items()}
        if tensor_terms:
            return terms['total'], detached
        values = torch.stack(list(detached.values())).cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('Non-finite loss component')
        return terms['total'], dict(zip(detached, values))
