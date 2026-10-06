# EchoRender

Trainable PyTorch student baseline for neural rendering, configured in `config/model.yaml`.

## Usage

```python
import torch
from model.student import EchoRenderStudent

model = EchoRenderStudent()  # resolves config relative to the source, not cwd
x = torch.randn(1, 64, 64, 16)  # BHWC by default
head = model(x)  # [1, 64, 64, 4]: RGB residual and temporal gate logit
head.square().mean().backward()
```

Install `requirements.txt` in your training environment. Run checks from the project root:
`python -m unittest discover -s tests -v`.

## YAML contract

Rows use YOLO's `[from, repeats, module, args]` convention. `-1` selects the
previous output (or model input at layer zero); other negative indices are relative;
non-negative indices select earlier graph outputs. Indices span backbone and head.
Module names come from an explicit registry in `model/card/card.py`, without eval.
`repeats` repeats complete modules; stage `depth` counts internal transformer blocks.
Use `input_layout: BCHW` for channel-first inputs and outputs. Features returned by
`model(x, return_features=True)` are always BCHW and remain attached to autograd.

Stage arguments are `[channels, depth, heads, window, ffn_ratio]` for Swin and
`[channels, depth, heads, ffn_ratio]` for ViT. Heads must divide channels.
Upsample takes `[decoder, skip]` sources and matches the skip's spatial dimensions.
Add merges equal-shaped tensors; Concat merges on the channel axis. To substitute
a custom SSM stage, define its class, add it to MODULES, and retain the
`(input_channels, output_channels, *args)` constructor and BCHW forward contract.

## Equivalence boundary

The baseline follows the shared conversation's five encoder/decoder widths,
stage depths, global bottleneck, skip connections and 16-input/4-output interface.
It is a **structural baseline**, not an exact reconstruction of DLSS-NR. LayerNorm,
GELU dense FFN, ordinary residual additions, stride-2 convolutions, bilinear
upsampling and standard PyTorch softmax deliberately replace vendor operations.
The final five-block stage plus separate head is a configurable student choice;
the recovered block 70 includes its head internally. No NVIDIA weight-key mapping
is claimed. Original recovered weights cannot be loaded into this student.
The full baseline is large; lower widths and depths in YAML for a smaller student.
Global bottleneck attention has quadratic token memory cost.

`model/teacher.py` wraps the external recovered `mlxdlss.model.load_model` API,
which must be installed separately. Use `weight/mlxw/dlssnr-logical.safetensors`.
Packed safetensors are opaque vendor storage and are not PyTorch state_dicts.
A safetensors file stores tensors; a matching recovered forward implementation is
still necessary. The teacher is frozen and stays in eval mode. Use
`torch.no_grad()` around teacher inference during distillation.

The model returns the raw head, not a composed RGB frame. Temporal reprojection,
noise/control-channel preparation, vendor padding and output composition belong
to the rendering/data pipeline and are not implemented here.

References:
- https://chatgpt.com/share/6ac4fb26-91f0-83ee-9e3c-26fb50a0e746
- https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/master/docs/ARCHITECTURE.md
