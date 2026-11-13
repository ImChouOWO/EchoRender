"""Window-routed selective scan using the official Mamba CUDA/reference API.

MambaIRv2-Small supplies state/prompt defaults. Overlap, directional ordering,
zero priors and parameter-conditioned prompts implement EchoRender's design.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def selective_scan(u, delta, A, B, C, D):
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        selective_scan_ref,
    )

    implementation = selective_scan_fn if u.is_cuda else selective_scan_ref
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, A, B, C, D = (v.to(dtype) for v in (u, delta, A, B, C, D))
    return implementation(
        u.transpose(1, 2).contiguous(),
        delta.transpose(1, 2).contiguous(),
        A.contiguous(),
        B.transpose(1, 2).contiguous(),
        C.transpose(1, 2).contiguous(),
        D.contiguous(),
    ).transpose(1, 2)


class Bottleneck(nn.Sequential):
    def __init__(self, c1, c2):
        middle = max(8, c2 // 2)
        super().__init__(
            nn.Conv2d(c1, middle, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(middle, middle, 1),
            nn.GELU(),
            nn.Conv2d(middle, c2, 3, padding=1),
        )


class RoutedScan(nn.Module):
    def __init__(self, dim, state=16, expand=2, tokens=128, rank=64):
        super().__init__()
        self.hidden, self.state = int(dim * expand), state
        self.dt_rank = math.ceil(self.hidden / 16)
        self.project = Bottleneck(dim, self.hidden)
        self.zero_map = nn.Parameter(torch.zeros(1, self.hidden, 1, 1))
        self.router = nn.Sequential(
            nn.Linear(dim, max(1, dim // 3)),
            nn.GELU(),
            nn.Linear(max(1, dim // 3), tokens),
        )
        self.embedding_a = nn.Parameter(
            torch.empty(rank, state).uniform_(-1 / rank, 1 / rank)
        )
        self.embedding_b = nn.Parameter(
            torch.empty(tokens, rank).uniform_(-1 / tokens, 1 / tokens)
        )
        self.parameters_proj = nn.Linear(
            self.hidden, self.dt_rank + 2 * state, bias=False
        )
        # [ranking features, delta_low, B, C] -> a state-sized prompt.
        self.prompt = nn.Conv1d(self.hidden + self.dt_rank + 2 * state, state, 1)
        self.dt_proj = nn.Linear(self.dt_rank, self.hidden)
        nn.init.uniform_(self.dt_proj.weight, -(self.dt_rank**-0.5), self.dt_rank**-0.5)
        dt = torch.exp(
            torch.rand(self.hidden) * math.log(0.1 / 0.001) + math.log(0.001)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(
            torch.arange(1, state + 1).float().log().repeat(self.hidden, 1)
        )
        self.D = nn.Parameter(torch.ones(self.hidden))
        self.norm = nn.LayerNorm(self.hidden)
        self.output = nn.Linear(self.hidden, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        logits = self.router(tokens)
        policy = (
            F.gumbel_softmax(logits, hard=True, dim=-1)
            if self.training
            else F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits)
        )
        order = policy.detach().argmax(-1).argsort(dim=-1, stable=True)
        inverse = order.argsort(-1)

        def reorder(value, indices=order):
            return value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))

        u = reorder((self.project(x) + self.zero_map).flatten(2).transpose(1, 2))
        projected = self.parameters_proj(u)
        low, B, C = projected.split((self.dt_rank, self.state, self.state), -1)
        prompt = self.prompt(torch.cat((u, projected), -1).transpose(1, 2)).transpose(
            1, 2
        )
        prompt = prompt + reorder(policy @ (self.embedding_b @ self.embedding_a))
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.softplus(
                F.linear(
                    low.float(), self.dt_proj.weight.float(), self.dt_proj.bias.float()
                )
            )
            y = selective_scan(
                u.float(),
                delta,
                -self.A_log.float().exp(),
                B.float(),
                C.float() + prompt.float(),
                self.D.float(),
            )
        y = self.output(self.norm(y.to(u.dtype)))
        return reorder(y, inverse).transpose(1, 2).reshape(b, c, h, w)


class SSMBlock(nn.Module):
    def __init__(
        self,
        channels,
        state_dim=16,
        expand=2,
        *,
        rotation="clockwise",
        directions=2,
        index=0,
        window=8,
        stride=4,
        tokens=128,
        prompt_rank=64,
    ):
        super().__init__()
        cycle = (0, 1, 2, 3) if rotation == "clockwise" else (0, 3, 2, 1)
        self.scan_directions = tuple(cycle[(index + i) % 4] for i in range(directions))
        self.window, self.stride = window, stride
        self.norm = nn.GroupNorm(1, channels)
        self.mixer = RoutedScan(channels, state_dim, expand, tokens, prompt_rank)
        self.output_adapter = Bottleneck(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        s, stride = self.window, self.stride
        hp = max(s, math.ceil(max(0, h - s) / stride) * stride + s)
        wp = max(s, math.ceil(max(0, w - s) / stride) * stride + s)
        z = F.pad(self.norm(x), (0, wp - w, 0, hp - h), mode="replicate")
        patches = F.unfold(z, s, stride=stride)
        count = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(b * count, c, s, s)
        total = None
        for direction in self.scan_directions:
            # Rotating the grid changes scan order before stable semantic grouping.
            oriented = torch.rot90(patches, direction, (-2, -1))
            value = torch.rot90(self.mixer(oriented), -direction, (-2, -1))
            total = value if total is None else total + value
        value = total / len(self.scan_directions)
        value = value.reshape(b, count, c * s * s).transpose(1, 2)
        restored = F.fold(value, (hp, wp), s, stride=stride)
        divisor = F.fold(x.new_ones(1, s * s, count), (hp, wp), s, stride=stride)
        return x + self.output_adapter((restored / divisor)[..., :h, :w])


class SSMStage(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        depth=2,
        state_dim=16,
        expand=2,
        *,
        rotation="clockwise",
        directions=2,
        window=8,
        stride=4,
        tokens=128,
        prompt_rank=64,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.adapter = Bottleneck(c1, c2)
        self.input_adapters = nn.ModuleList(Bottleneck(c2, c2) for _ in range(depth))
        self.blocks = nn.ModuleList(
            SSMBlock(
                c2,
                state_dim,
                expand,
                rotation=rotation,
                directions=directions,
                index=i,
                window=window,
                stride=stride,
                tokens=tokens,
                prompt_rank=prompt_rank,
            )
            for i in range(depth)
        )

    def forward(self, x):
        x, previous = self.adapter(x), None
        for adapter, block in zip(self.input_adapters, self.blocks):
            # The previous block's pre-scan high-dimensional input bypasses its scan.
            prepared = adapter(x if previous is None else x + previous)
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                result = checkpoint(block, prepared, use_reentrant=False)
            else:
                result = block(prepared)
            previous, x = prepared, result
        return x
