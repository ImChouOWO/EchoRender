"""Configurable NAFNet and Restormer heads with sequential RGB supervision.

Blocks follow the official NAFNet (MIT) and Restormer (MIT) implementations;
see THIRD_PARTY.md for attribution. Width/depth are EchoRender configuration.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(
            x.var(1, keepdim=True, unbiased=False) + 1e-6
        ) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.spatial = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2),
            SimpleGate(),
        )
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.project = nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1), SimpleGate(), nn.Conv2d(c, c, 1)
        )
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        z = self.spatial(self.norm1(x))
        y = x + self.project(z * self.sca(z)) * self.beta
        return y + self.ffn(self.norm2(y)) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=16, depths=(1, 2), middle=2):
        super().__init__()
        self.levels = len(depths)
        self.encoders, self.decoders, self.downs, self.ups = (
            nn.ModuleList() for _ in range(4)
        )
        c = width
        for depth in depths:
            self.encoders.append(nn.Sequential(*(NAFBlock(c) for _ in range(depth))))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2))
            c *= 2
        self.middle = nn.Sequential(*(NAFBlock(c) for _ in range(middle)))
        for depth in reversed(depths):
            self.ups.append(
                nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            c //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(c) for _ in range(depth))))

    def forward(self, x):
        h, w = x.shape[-2:]
        unit = 2**self.levels
        x = F.pad(x, (0, (-w) % unit, 0, (-h) % unit))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(up(x) + skip)
        return x[..., :h, :w]


class RestormerBlock(nn.Module):
    def __init__(self, c, heads=1, expansion=2.66):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.qkv = nn.Sequential(
            nn.Conv2d(c, c * 3, 1, bias=False),
            nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False),
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(c, c, 1, bias=False)
        hidden = int(c * expansion)
        self.ffn_in = nn.Conv2d(c, hidden * 2, 1, bias=False)
        self.ffn_dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False
        )
        self.ffn_out = nn.Conv2d(hidden, c, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = (
            t.reshape(b, self.heads, c // self.heads, h * w)
            for t in self.qkv(self.norm1(x)).chunk(3, 1)
        )
        attention = (
            F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        ) * self.temperature
        x = x + self.project((attention.softmax(-1) @ v).reshape(b, c, h, w))
        a, d = self.ffn_dw(self.ffn_in(self.norm2(x))).chunk(2, 1)
        return x + self.ffn_out(F.gelu(a) * d)


class Restormer(nn.Module):
    def __init__(self, width=16, depths=(1, 2), middle=2, heads=(1, 2, 4)):
        super().__init__()
        self.levels = len(depths)
        self.encoders, self.decoders, self.downs, self.ups, self.reduce = (
            nn.ModuleList() for _ in range(5)
        )
        c = width
        for i, depth in enumerate(depths):
            self.encoders.append(
                nn.Sequential(*(RestormerBlock(c, heads[i]) for _ in range(depth)))
            )
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(c, c // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2)
                )
            )
            c *= 2
        self.middle = nn.Sequential(
            *(RestormerBlock(c, heads[-1]) for _ in range(middle))
        )
        for i in reversed(range(len(depths))):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(c, c * 2, 3, padding=1, bias=False), nn.PixelShuffle(2)
                )
            )
            c //= 2
            self.reduce.append(nn.Conv2d(c * 2, c, 1, bias=False))
            self.decoders.append(
                nn.Sequential(*(RestormerBlock(c, heads[i]) for _ in range(depths[i])))
            )

    def forward(self, x):
        h, w = x.shape[-2:]
        unit = 2**self.levels
        x = F.pad(x, (0, (-w) % unit, 0, (-h) % unit))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, reduce, decoder, skip in zip(
            self.ups, self.reduce, self.decoders, reversed(skips)
        ):
            x = decoder(reduce(torch.cat((up(x), skip), 1)))
        return x[..., :h, :w]


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, shift):
        super().__init__()
        self.window, self.heads, self.shift = 8, 4, shift
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        w = self.window
        coords = torch.cartesian_prod(torch.arange(w), torch.arange(w))
        phase = coords[:, 0] % 2 * 2 + coords[:, 1] % 2
        queries = torch.where(phase != 0)[0]
        keys = coords[phase == 0]
        offsets = coords[queries, None] - keys[None] + w - 1
        self.register_buffer("queries", queries, persistent=False)
        self.register_buffer("phases", phase[queries] - 1, persistent=False)
        self.register_buffer(
            "relative_index", offsets[..., 0] * (2 * w - 1) + offsets[..., 1],
            persistent=False,
        )
        self.relative_bias = nn.Parameter(torch.zeros((2 * w - 1) ** 2, self.heads))

    @staticmethod
    def _windows(x, size):
        b, h, w, c = x.shape
        return (
            x.reshape(b, h // size, size, w // size, size, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, -1, size**2, c)
        )

    def forward(self, reference, original, query_bias):
        b, c, h, w = reference.shape
        size, shift = self.window, self.shift
        bottom, right = (-(h + shift)) % size, (-(w + shift)) % size
        hp, wp = h + shift + bottom, w + shift + right

        # Shift the window origin with padding, so opposite edges never wrap.
        pad = (shift, right, shift, bottom)
        query = self._windows(F.pad(reference, pad).movedim(1, -1), size)
        query = self.query(query[:, :, self.queries] + query_bias[self.phases])
        lr_pad = tuple(value // 2 for value in pad)
        keys = self._windows(F.pad(original, lr_pad).movedim(1, -1), size // 2)

        def heads(x):
            return x.flatten(0, 1).unflatten(-1, (self.heads, c // self.heads)).transpose(1, 2)

        query = heads(query)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        bias = bias.to(query.dtype).contiguous()
        if any(pad):
            valid = original.new_ones((1, 1, h // 2, w // 2), dtype=torch.bool)
            valid = self._windows(F.pad(valid, lr_pad).movedim(1, -1), size // 2)
            valid = valid[0, ..., 0]
            bias = bias.masked_fill(~valid[:, None, None, :], -torch.inf)
            bias = bias.repeat(b, 1, 1, 1)
        attended = F.scaled_dot_product_attention(
            query, heads(self.key(keys)), heads(self.value(keys)), attn_mask=bias
        )
        attended = self.output(attended.transpose(1, 2).reshape(b, -1, len(self.queries), c))
        # Scatter only the 48 generated positions; anchors receive zero update.
        windows = attended.new_zeros((*attended.shape[:2], size**2, c))
        windows = windows.index_copy(2, self.queries, attended)
        grid = (
            windows.reshape(b, hp // size, wp // size, size, size, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, hp, wp, c)
        )
        return grid[:, shift:shift + h, shift:shift + w].movedim(-1, 1)


class MaskedResidualUpsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tokens = nn.Parameter(torch.zeros(3, dim))  # horizontal, vertical, diagonal
        self.position = nn.Linear(2, dim, bias=False)
        self.register_buffer(
            "offsets", torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]),
            persistent=False,
        )
        self.standard = WindowCrossAttention(dim, shift=0)
        self.shifted = WindowCrossAttention(dim, shift=4)
        self.rgb = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.rgb[-1].weight)
        nn.init.zeros_(self.rgb[-1].bias)

    @staticmethod
    def _reference(x):
        # Interpolate between anchors without moving the original feature samples.
        right = torch.cat((x[..., 1:], x[..., -1:]), dim=-1)
        x = torch.stack((x, (x + right) * 0.5), dim=-1).flatten(-2)
        below = torch.cat((x[..., 1:, :], x[..., -1:, :]), dim=-2)
        return torch.stack((x, (x + below) * 0.5), dim=-2).flatten(-3, -2)

    def _rgb_update(self, update):
        b, c, h, w = update.shape
        cells = update.reshape(b, c, h // 2, 2, w // 2, 2)
        cells = cells.permute(0, 2, 4, 3, 5, 1).flatten(3, 4)
        # Predict three RGB residuals per cell, then insert the zero anchor.
        residual = F.pad(self.rgb(cells[..., 1:, :]), (0, 0, 1, 0))
        return (
            residual.reshape(b, h // 2, w // 2, 2, 2, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, 3, h, w)
        )

    def forward(self, features, rgb):
        reference = self._reference(features)
        bias = (self.tokens + self.position(self.offsets)).to(reference.dtype)
        first = self.standard(reference, features, bias)
        update = first + self.shifted(reference + first, features, bias)
        anchors = F.pad(features[..., None, :, None], (0, 1, 0, 0, 0, 1))
        anchors = anchors.reshape_as(update)
        base = F.interpolate(rgb, scale_factor=2, mode="bicubic", align_corners=False)
        return anchors + update, base + self._rgb_update(update)


class ReconstructionHead(nn.Module):
    def __init__(
        self, c, scale, width=16, naf_depths=(1, 2), restormer_depths=(1, 2), middle=2
    ):
        super().__init__()
        self.scale = scale
        dim = 32
        self.sr_in = nn.Identity() if c == dim else nn.Conv2d(c, dim, 1)
        self.sr = nn.ModuleList(
            MaskedResidualUpsample(dim) for _ in range({2: 1, 4: 2, 8: 3}[scale])
        )
        self.sr_out = nn.Conv2d(dim, width, 1)
        self.deblur_in = nn.Conv2d(width + 3, width, 3, padding=1)
        self.deblur = NAFNet(width, naf_depths, middle)
        self.deblur_rgb = nn.Conv2d(width, 3, 3, padding=1)
        self.denoise_in = nn.Conv2d(width * 2 + 6, width, 3, padding=1)
        self.denoise = Restormer(
            width,
            restormer_depths,
            middle,
            tuple(2**i for i in range(len(restormer_depths) + 1)),
        )
        self.denoise_rgb = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, features, rgb):
        features, y1 = self.sr_in(features), rgb
        for upsample in self.sr:
            features, y1 = upsample(features, y1)
        f1 = self.sr_out(features)
        f2 = self.deblur(self.deblur_in(torch.cat((f1, y1), 1)))
        y2 = y1 + self.deblur_rgb(f2)
        f3 = self.denoise(self.denoise_in(torch.cat((f1, f2, y1, y2), 1)))
        y3 = y2 + self.denoise_rgb(f3)
        return {"sr": y1, "deblur": y2, "output": y3}
