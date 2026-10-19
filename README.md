# EchoRender

Trainable PyTorch student baseline for neural rendering, configured in `config/model.yaml`.

## Python environment

Run from the project root with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

uv installs Python 3.12 and creates a local `.venv`; manual activation is optional
when using `uv run`. Commit `pyproject.toml`, `uv.lock`, and `.python-version` so
collaborators can reproduce the environment. `.venv` is already ignored by Git.
The environment selects PyTorch's official CUDA 12.8 wheels on Windows/Linux,
and PyPI builds on macOS. Use a compatible NVIDIA driver for CUDA execution.
NumPy supports tensor/feature-cache conversion, and Pillow supports image reading.
The recovered Teacher backend is installed from a pinned MLX-DLSS Git revision.
Teacher weights remain external, user-supplied files.

## Single-image Teacher inference

```powershell
uv sync --locked
uv run python infer.py --input input.png --weights "C:/path/to/dlssnr-logical.safetensors" --output outputs/teacher.png --device auto --report outputs/teacher.json
```

This runs the actual recovered Teacher, without a DataLoader or student. The
default device is CUDA when available, otherwise CPU. Input images are converted
to RGB, padded using the pinned backend's reflected geometry (at least 320 pixels
per side, multiples of 64), and cropped back to their original dimensions. The
result is a lossless RGB PNG. Existing outputs require `--overwrite`.

Every call independently builds the 16 input channels with deterministic noise
(`--noise-index 0` by default), standard controls, and current-image values in
the history slots. All four head channels are composed using the checkpoint's
learned blend cap; no previous image is stored. `--intensity` accepts [0,1].
Only the standard style is currently exposed. The model computes in float32
while preserving the recovered backend's explicit FP16/E4M3 rounding points.

`model/teacher.py` owns loading/frozen model execution;
`tool/teacher_pipeline.py` owns image preparation/composition; `infer.py` is the CLI.
See [Teacher implementation and testing](docs/TEACHER_INFERENCE.md) for the
backend revision, validation limits and optional real-weight tests.

## Usage

```python
import torch
from model.student import EchoRenderStudent

model = EchoRenderStudent()  # resolves config relative to the source, not cwd
x = torch.randn(1, 64, 64, 16)  # BHWC by default
head = model(x)  # [1, 64, 64, 4]: RGB residual and temporal gate logit
head.square().mean().backward()
```

Use the uv environment above. `requirements.txt` remains a compatibility list for
other installers; it does not lock exact versions or select the CUDA wheel index.

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

`model/teacher.py` uses the pinned recovered MLX-DLSS graph and validates decoded
logical weights. Packed safetensors are opaque storage and are rejected. The
Teacher remains frozen/eval and its forward runs under `torch.no_grad()` so
its output can be used as an ordinary detached distillation target.

Both model classes return raw four-channel heads. `TeacherPipeline` now supplies
the single-image input preparation and RGB composition for the Teacher; the
student's corresponding image pipeline and training loop remain future work.
The recovered reference graph and rounding are not claimed to be bit-identical
to NVIDIA's kernels, or to the independently updated Intel port. Teacher/student
feature coordinates must still be aligned explicitly for feature distillation.

References:
- https://chatgpt.com/share/6ac4fb26-91f0-83ee-9e3c-26fb50a0e746
- https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/master/docs/ARCHITECTURE.md
