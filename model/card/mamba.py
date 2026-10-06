"""Pure PyTorch Mamba-1 style Selective SSM, including CPU/Windows support.

Uses input-dependent delta/B/C, negative diagonal A, causal depthwise convolution
and SiLU output gating. Sequential scan is a correctness/research baseline, not
an optimized kernel or an official mamba-ssm package. No streaming cache API.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['Mamba', 'MambaBlock', 'MambaStage', 'SSMStage', 'selective_scan']


def selective_scan(u, delta, A, B, C, D):
    """u/delta: BLD; A: DN; B/C: BLN; D: D. Zero initial state.

Float32 accumulation for half/bfloat16; retain float64 for numerical tests.
Recurrence: h[t] = exp(delta[t]*A)*h[t-1] + delta[t]*B[t]*u[t].
"""
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, A, B, C, D = (v.to(dtype) for v in (u, delta, A, B, C, D))
    batch, length, channels = u.shape
    if length < 1:
        raise ValueError('Sequence must be nonempty')
    state = u.new_zeros(batch, channels, A.shape[-1])
    outputs = []
    for t in range(length):
        dt = delta[:, t, :, None]
        state = torch.exp(dt * A) * state + dt * B[:, t, None, :] * u[:, t, :, None]
        outputs.append((state * C[:, t, None, :]).sum(-1) + D * u[:, t])
    return torch.stack(outputs, dim=1)


class Mamba(nn.Module):
    """Mamba-1 style mixer: [B,L,d_model] -> [B,L,d_model]."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto',
                 dt_min=0.001, dt_max=0.1, bias=False, conv_bias=True):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (d_model, d_state, d_conv, expand)):
            raise ValueError('Model/state/conv dimensions and expand must be positive integers')
        rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        if type(rank) is not int or rank < 1 or not 0 < dt_min < dt_max:
            raise ValueError('Invalid dt_rank or delta initialization range')
        self.d_model, self.d_state = d_model, d_state
        self.d_inner, self.dt_rank = d_model * expand, rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(rank, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -rank**-0.5, rank**-0.5)
        dt = torch.exp(torch.rand(self.d_inner) * math.log(dt_max / dt_min) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] < 1:
            raise ValueError(f'Expected [B,L,{self.d_model}] with L > 0')
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :x.shape[1]]).transpose(1, 2)
        delta_low, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        # Compute time steps and recurrence in full precision even under autocast.
        with torch.autocast(device_type=u.device.type, enabled=False):
            delta = F.softplus(F.linear(delta_low.to(dtype), self.dt_proj.weight.to(dtype), self.dt_proj.bias.to(dtype)))
            y = selective_scan(u, delta, -self.A_log.to(dtype).exp(), B, C, self.D)
            y = y * F.silu(z.to(dtype))
        return self.out_proj(y.to(u.dtype))


class MambaBlock(nn.Module):
    """BCHW residual mixer. raster: flattened image; cross: four axial scans.

cross shares mixer weights between directions and resets state per row/column.
It is a custom image wrapper, not VMamba's exact SS2D implementation.
"""
    def __init__(self, channels, state_dim=16, d_conv=4, expand=2, scan='raster'):
        super().__init__()
        if scan not in ('raster', 'cross'):
            raise ValueError('scan must be raster or cross')
        self.scan = scan
        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(channels, state_dim, d_conv, expand)

    def _bidirectional(self, sequence):
        return (self.mixer(sequence) + self.mixer(sequence.flip(1)).flip(1)) * 0.5

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 1:
            raise ValueError('Expected nonempty BCHW image')
        b, c, h, w = x.shape
        z = self.norm(x.permute(0, 2, 3, 1))
        if self.scan == 'raster':
            y = self.mixer(z.reshape(b, h * w, c)).reshape(b, h, w, c)
        else:
            rows = self._bidirectional(z.reshape(b * h, w, c)).reshape(b, h, w, c)
            cols = self._bidirectional(z.permute(0, 2, 1, 3).reshape(b * w, h, c))
            cols = cols.reshape(b, w, h, c).permute(0, 2, 1, 3)
            y = (rows + cols) * 0.5
        return x + y.permute(0, 3, 1, 2).contiguous()


class MambaStage(nn.Module):
    def __init__(self, c1, c2, depth=2, state_dim=16, d_conv=4, expand=2, scan='raster'):
        super().__init__()
        if type(depth) is not int or depth < 1 or c1 < 1 or c2 < 1:
            raise ValueError('Channels and depth must be positive')
        self.adapter = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(MambaBlock(c2, state_dim, d_conv, expand, scan) for _ in range(depth)))

    def forward(self, x):
        return self.blocks(self.adapter(x))


SSMStage = MambaStage
