"""Trainable DLSS-NR topology modules. Tensors inside the graph are BCHW.

This is a structural student baseline, not the recovered NVIDIA execution graph.
It intentionally uses ordinary differentiable PyTorch arithmetic.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .mamba import Mamba, MambaBlock, MambaStage, SSMStage

class InputAdapter(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        return self.proj(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        hidden = int(dim * ratio)
        if hidden < 1:
            raise ValueError('FFN hidden dimension must be positive')
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class CosineAttention(nn.Module):
    def __init__(self, dim, heads, tokens=None):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError('dim must be divisible by positive heads')
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.log_scale = nn.Parameter(torch.full((heads, 1, 1), 2.302585))
        self.bias = nn.Parameter(torch.zeros(heads, tokens, tokens)) if tokens else None

    def forward(self, x, mask=None):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q = F.normalize(q, dim=-1) * self.log_scale.clamp(max=4.605170).exp()
        k = F.normalize(k, dim=-1)
        bias = self.bias
        if mask is not None:
            bias = mask[:, None] if bias is None else bias[None] + mask[:, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window=8, shift=0, ratio=4):
        super().__init__()
        if window < 1 or not 0 <= shift < window:
            raise ValueError('Require window > 0 and 0 <= shift < window')
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CosineAttention(dim, heads, window * window)
        self.ffn = FeedForward(dim, ratio)

    def _partition(self, x):
        b, h, w, c = x.shape
        s = self.window
        return x.reshape(b, h // s, s, w // s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, s * s, c)

    def forward(self, x):
        b, c, h, w = x.shape
        s, shift = self.window, self.shift
        hp, wp = ((h + s - 1) // s) * s, ((w + s - 1) // s) * s
        z = F.pad(x, (0, wp - w, 0, hp - h)).permute(0, 2, 3, 1)
        valid = torch.zeros((1, hp, wp, 1), device=x.device)
        valid[:, :h, :w] = 1
        if shift:
            z = torch.roll(z, (-shift, -shift), (1, 2))
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        # Region mask prevents cyclic wrap-around attention; key mask excludes padding.
        regions = torch.zeros((1, hp, wp, 1), device=x.device)
        if shift:
            slices = (slice(0, -s), slice(-s, -shift), slice(-shift, None))
            for i, hs in enumerate(slices):
                for j, ws in enumerate(slices):
                    regions[:, hs, ws] = i * 3 + j
        labels = self._partition(regions).squeeze(-1)
        allowed = labels[:, :, None] == labels[:, None, :]
        keys = self._partition(valid).squeeze(-1).bool()
        allowed = allowed & keys[:, None, :]
        mask = torch.zeros(allowed.shape, device=x.device, dtype=x.dtype).masked_fill(~allowed, float('-inf')).repeat(b, 1, 1)
        z = self._partition(z)
        z = z + self.attn(self.norm1(z), mask)
        z = z + self.ffn(self.norm2(z))
        z = z.reshape(b, hp // s, wp // s, s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        if shift:
            z = torch.roll(z, (shift, shift), (1, 2))
        return z[:, :h, :w].permute(0, 3, 1, 2).contiguous()


class SwinStage(nn.Module):
    def __init__(self, c1, c2, depth, heads, window=8, ratio=4):
        super().__init__()
        if depth < 1:
            raise ValueError('Stage depth must be positive')
        self.adapter = InputAdapter(c1, c2) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(SwinBlock(c2, heads, window, 0 if i % 2 == 0 else window // 2, ratio) for i in range(depth)))

    def forward(self, x):
        return self.blocks(self.adapter(x))


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, ratio=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.ffn = CosineAttention(dim, heads), FeedForward(dim, ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ViTStage(nn.Module):
    def __init__(self, c1, c2, depth, heads, ratio=4):
        super().__init__()
        if depth < 1:
            raise ValueError('Stage depth must be positive')
        self.adapter = InputAdapter(c1, c2) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(ViTBlock(c2, heads, ratio) for _ in range(depth)))

    def forward(self, x):
        x = self.adapter(x)
        b, c, h, w = x.shape
        return self.blocks(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 2, stride=2)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.proj(F.pad(x, (0, w % 2, 0, h % 2)))


class Upsample(nn.Module):
    """Resize to skip tensor dimensions, then project. Supports odd image sizes."""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1)

    def forward(self, xs):
        x, skip = xs
        return self.proj(F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False))


class Add(nn.Module):
    def forward(self, xs):
        if len(xs) < 2 or any(x.shape != xs[0].shape for x in xs):
            raise ValueError('Add needs at least two tensors of equal shape')
        return torch.stack(xs).sum(0)


class Concat(nn.Module):
    def forward(self, xs):
        return torch.cat(xs, dim=1)


class OutputHead(InputAdapter):
    """Raw RGB residual and temporal gate logit. No output sigmoid."""


MODULES = {cls.__name__: cls for cls in (InputAdapter, SwinStage, ViTStage, Downsample, Upsample, Add, Concat, OutputHead, MambaStage)}

MODULES['SSMStage'] = SSMStage
