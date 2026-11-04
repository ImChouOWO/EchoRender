"""Selected-stage distillation and clean-source Teacher-change weighted RGB losses."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_msssim.ssim import _fspecial_gauss_1d, gaussian_filter
from .appearance import AppearanceLoss, APPEARANCE_TERMS, APPEARANCE_SCORES

STAGE_CHANNELS = {'InputAdapter': 32, 'E0': 32, 'E1': 64, 'E2': 128, 'E3': 256, 'E4': 512, 'VIT': 1024,
                  'D1': 512, 'D2': 256, 'D3': 128}
TEACHER_STRIDES = {'InputAdapter': 1, 'E0': 2, 'E1': 4, 'E2': 8, 'E3': 16, 'E4': 32, 'VIT': 64, 'D1': 32, 'D2': 16, 'D3': 8}


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
    if student.shape[-2:] != physical_hw:
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
    def __init__(self, config, channels=None, *, stages=None, student_channels=None):
        super().__init__()
        self.config = dict(config)
        self.appearance = AppearanceLoss(config['appearance_config'])
        self.perceptual = None
        if config['rgb_lpips'] > 0:
            from tool.metrics import pretrained_lpips
            self.perceptual = pretrained_lpips(config['lpips_net'])
        from model.student import DEFAULT_STAGES, HEAD, COMPOSITION
        selected = tuple(DEFAULT_STAGES if stages is None else stages)
        self.stages = tuple(name for name in selected if name in STAGE_CHANNELS)
        self.supervise_head = HEAD in selected
        self.supervise_rgb = COMPOSITION in selected
        if not selected or len(set(selected)) != len(selected) or any(n not in (*STAGE_CHANNELS, HEAD, COMPOSITION) for n in selected):
            raise ValueError('Invalid distillation stages')
        channels = ({name: STAGE_CHANNELS[name] for name in self.stages} if channels is None else channels)
        if set(channels) != set(self.stages):
            raise ValueError('Teacher channels must match selected distillation stages')
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

    def train(self, mode=True):
        super().train(mode)
        if self.perceptual is not None:
            self.perceptual.eval()
        return self

    def perceptual_loss(self, prediction, target):
        if self.perceptual is None:
            return prediction.new_zeros(())
        if min(prediction.shape[-2:]) < 64:
            raise ValueError('LPIPS loss requires real RGB images at least 64x64')
        # Freeze the network, not its input gradient. Teacher RGB is a fixed target.
        return self.perceptual(prediction * 2 - 1, target.detach() * 2 - 1).mean()

    def importance(self, target, clean_source):
        if clean_source is None or clean_source.shape != target.shape:
            raise ValueError('Teacher-change weighting requires matching unaugmented source RGB')
        c = self.config['teacher_change']
        delta = (target.detach() - clean_source.detach()).abs().mean(1, keepdim=True)
        return (1 + c['alpha'] * (delta / c['tau']).clamp(0, 1)).detach()

    @staticmethod
    def weighted_mean(value, importance):
        # Normalize each image independently so image area cannot change its batch weight.
        return ((value * importance).sum((-2, -1)) /
                importance.sum((-2, -1)).clamp_min(1e-12)).mean()

    def local_scores(self, prediction, target, importance):
        window = _fspecial_gauss_1d(self.config['ssim_window'], 1.5).to(prediction)
        window = window.repeat(prediction.shape[1], 1, 1, 1)
        mu1, mu2 = gaussian_filter(prediction, window), gaussian_filter(target, window)
        v1 = gaussian_filter(prediction.square(), window) - mu1.square()
        v2 = gaussian_filter(target.square(), window) - mu2.square()
        covariance = gaussian_filter(prediction * target, window) - mu1 * mu2
        contrast = (2 * covariance + .03**2) / (v1 + v2 + .03**2)
        score = (2 * mu1 * mu2 + .01**2) / (mu1.square() + mu2.square() + .01**2) * contrast
        # Filter importance onto the same valid local windows; never mask RGB before SSIM.
        local_weight = gaussian_filter(importance, window[:1])
        def reduce(value):
            return (value * local_weight).sum((-2, -1)) / local_weight.sum((-2, -1)).clamp_min(1e-12)
        return reduce(score), reduce(contrast)

    def image_terms(self, prediction, target, clean_source=None):
        c = self.config
        target = target.detach()
        importance = self.importance(target, clean_source)
        levels = len(c['ms_ssim_weights'])
        minimum = (c['ssim_window'] - 1) * 2 ** (levels - 1)
        if min(prediction.shape[-2:]) <= minimum:
            raise ValueError(f'MS-SSIM needs real image height/width > {minimum}')
        mse = self.weighted_mean((prediction - target).square(), importance)
        perceptual = self.perceptual_loss(prediction, target)
        score, contrast = self.local_scores(prediction, target, importance)
        structural = score.mean()
        values = []
        for level in range(levels):
            if level:
                score, contrast = self.local_scores(prediction, target, importance)
            values.append(torch.relu(score if level == levels - 1 else contrast))
            if level < levels - 1:
                padding = [size % 2 for size in prediction.shape[-2:]]
                prediction = F.avg_pool2d(prediction, 2, padding=padding)
                target = F.avg_pool2d(target, 2, padding=padding)
                importance = F.avg_pool2d(importance, 2, padding=padding, count_include_pad=False)
        exponents = prediction.new_tensor(c['ms_ssim_weights']).view(-1, 1, 1)
        multiscale = torch.prod(torch.stack(values) ** exponents, dim=0).mean()
        # MSE is evaluated at native output size, before the multi-scale pooling above.
        return {'rgb_mse': mse, 'rgb_ssim': 1 - structural,
                'rgb_ms_ssim': 1 - multiscale, 'rgb_lpips': perceptual}

    def forward(self, student_head, teacher_head, student_features, teacher_features,
                prediction_rgb, target_rgb, *, image_hw, network_hw, clean_source,
                weights=None, tensor_terms=False):
        c = self.config if weights is None else {**self.config, **weights}
        if set(student_features) != set(self.stages) or set(teacher_features) != set(self.stages):
            raise ValueError('Captured features must exactly match selected distillation stages')
        terms = {}
        for name in self.stages:
            s, t = align_features(self.projections[name](student_features[name].float()), teacher_features[name].float(),
                                  image_hw=image_hw, network_hw=network_hw, stride=TEACHER_STRIDES[name])
            if name == 'VIT':
                terms['vit_feature'] = F.mse_loss(s, t)
                terms['vit_affinity'] = token_affinity_loss(
                    s, t, eps=c['norm_eps'], chunk_size=c['affinity_chunk_size'])
                terms[name] = c['vit_feature'] * terms['vit_feature'] + c['vit_affinity'] * terms['vit_affinity']
            else:
                terms[name] = normalized_feature_mse(s, t, c['norm_eps'])
        h, w = image_hw
        s, t = student_head[:, :h, :w].float(), teacher_head.detach()[:, :h, :w].float()
        terms['output_rgb'] = F.mse_loss(s[..., :3], t[..., :3]) if self.supervise_head else s.sum() * 0
        terms['output_gate'] = F.mse_loss(s[..., 3:], t[..., 3:]) if self.supervise_head else s.sum() * 0
        terms['output'] = c['output_rgb'] * terms['output_rgb'] + c['output_gate'] * terms['output_gate']
        count = len(self.stages)
        terms['mid'] = sum((c['feature_weights'][name] * terms[name] for name in self.stages), s.sum() * 0) / max(count, 1)
        terms['distillation'] = c['mid'] * terms['mid'] + c['output'] * terms['output']
        terms.update(self.image_terms(prediction_rgb.float(), target_rgb.float(), clean_source) if self.supervise_rgb else
                     {name: prediction_rgb.sum() * 0 for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim', 'rgb_lpips')})
        terms['reconstruction'] = sum(c[name] * terms[name] for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim', 'rgb_lpips'))
        appearance = (self.appearance(prediction_rgb, target_rgb) if self.supervise_rgb and c['appearance'] > 0 else
                      {name: prediction_rgb.sum() * 0 for name in (*APPEARANCE_TERMS, *APPEARANCE_SCORES)})
        terms.update({f'appearance_{name}': value for name, value in appearance.items()})
        terms['appearance'] = sum(c['appearance_config']['weights'][name] * appearance[name] for name in APPEARANCE_TERMS)
        terms['total'] = (c['distillation'] * terms['distillation'] + c['reconstruction'] * terms['reconstruction']
                          + c['appearance'] * terms['appearance'])
        for name in self.stages:
            factor = c['distillation'] * c['mid'] * c['feature_weights'][name] / count
            if name == 'VIT':
                for subterm in ('vit_feature', 'vit_affinity'):
                    terms[f'weighted_{subterm}'] = factor * c[subterm] * terms[subterm]
            else:
                terms[f'weighted_{name}'] = factor * terms[name]
        for name in ('output_rgb', 'output_gate'):
            terms[f'weighted_{name}'] = c['distillation'] * c['output'] * c[name] * terms[name]
        for name in ('rgb_mse', 'rgb_ssim', 'rgb_ms_ssim', 'rgb_lpips'):
            terms[f'weighted_{name}'] = c['reconstruction'] * c[name] * terms[name]
        for name in APPEARANCE_TERMS:
            terms[f'weighted_appearance_{name}'] = c['appearance'] * c['appearance_config']['weights'][name] * appearance[name]
        detached = {name: value.detach() for name, value in terms.items()}
        if tensor_terms:
            return terms['total'], detached
        values = torch.stack(list(detached.values())).cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('Non-finite loss component')
        return terms['total'], dict(zip(detached, values))
