# EchoRender

Single-image knowledge distillation from a frozen recovered DLSS-NR Teacher to a
trainable PyTorch student. Each image is processed independently; no video
sequence, optical flow or cross-image state is used.

## Python environment

Run from the project root with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --locked
```

uv installs Python 3.12 and creates `.venv`. `uv run` uses this environment without
manual activation. `pyproject.toml`, `uv.lock` and `.python-version` define the
shared environment. PyTorch uses CUDA 12.8 wheels on Windows/Linux and PyPI builds
on macOS. The recovered Teacher backend is pinned to an MLX-DLSS Git revision.
Teacher weights and large Sintel downloads remain local and are ignored by Git.

## Training

Place the extracted Sintel dataset at `MPI-Sintel-complete/`, or change
`dataset.root` in `config/data.yaml`. Supply the decoded **logical** Teacher
checkpoint; packed safetensors are unsupported.

```powershell
uv run python train.py --teacher-weights "C:/path/to/dlssnr-logical.safetensors" --epochs 10 --output-dir outputs/run_01
```

`train.py` loads the full `config/model.yaml` student (172,122,446 parameters),
initialized independently from the frozen Teacher. Both models receive the same
16-channel input prepared from the same RGB image. Training uses a shared
320x320 center crop, reflected padding where needed, and MSE between the raw
four-channel heads on valid image pixels. Each batch averages per-image losses
and makes one optimizer update. The Teacher runs under `no_grad` in eval mode.

The default optimizer is SGD with learning rate 0.001 and gradient-norm clipping
at 1.0. After each epoch the complete validation split is evaluated without
updates, then the student checkpoint is saved. The official test split is
reserved for separate evaluation.

Useful training options:

- `--epochs`: number of passes over the configured training split.
- `--max-steps`: optional total optimizer-update budget; validation and checkpoint
  saving still run when training reaches this budget.
- `--lr`, `--grad-clip`: learning rate and gradient-norm cap; a cap of 0 disables clipping.
- `--crop-size`: shared center-crop size; 0 uses original image dimensions.
- `--student-config`, `--data-config`: model and dataset configuration paths.
- `--student-weights`: initialize student weights from a state dict or `last.pt`.
- `--device`: `auto`, `cuda`, `cuda:0` or `cpu`.

Use a new `--output-dir` for each run. Training writes:

- `metrics.jsonl`: per-step training metrics and epoch/validation losses.
- `last.pt`: student weights/config, optimizer state and training progress.
- `summary.json`: settings, losses, progress and frozen-Teacher information.

`--student-weights` starts a new run with matching student weights; it does not
restore optimizer state, data order or epoch progress. Teacher weights are never
copied into training checkpoints. `outputs/` is ignored by Git.

## Dataset and preprocessing

`config/data.yaml` defines the Sintel root, render pass, scene splits, batch size,
workers, shuffle and seed. Dataset paths are resolved relative to this YAML.

- `train`: 19 selected scenes from the official `training` source.
- `val`: held-out alley/temple scenes from the same source.
- `test`: the official `test` source, used separately from training/validation.

`clean` and `final` are rendering passes, independent of the split. Train/val
scene overlap and use of official test data for training are rejected.
`tool/dataloader.py` returns CPU float32 CHW RGB in [0,1], sample IDs, paths,
scenes and sizes. Batches preserve original image sizes as lists. Training
shuffle is seeded; validation/test order is fixed. Cropping and shared Teacher
preparation happen in `tool/distillation.py`, after loading.

## Teacher image inference

```powershell
uv run python infer.py --input input.png --weights "C:/path/to/dlssnr-logical.safetensors" --output outputs/teacher.png
```

`infer.py` runs the Teacher and saves an RGB PNG at the original image dimensions.
It converts input to RGB and pads to the backend's minimum 320/multiple-of-64
geometry. Noise and controls are deterministic; channels 7:10 use the current
image. All four head channels are composed using the checkpoint's learned blend
cap. No previous image is retained. `--intensity` accepts [0,1]; only standard
style is exposed. Existing outputs require `--overwrite`. `--report` optionally
saves inference metadata. See [Teacher implementation](docs/TEACHER_INFERENCE.md).

## Model configuration

Rows in `config/model.yaml` use `[from, repeats, module, args]`. `-1` selects the
previous output (model input at layer zero); other negative indices are relative;
non-negative indices select earlier graph outputs. Indices span backbone/head.
Modules come from the explicit registry in `model/card/card.py`.

Swin arguments are `[channels, depth, heads, window, ffn_ratio]`; ViT omits window.
`repeats` repeats modules; stage depth counts internal blocks. Heads must divide
channels. Upsample takes `[decoder, skip]` and matches skip dimensions. Add merges
equal-shaped tensors; Concat merges channels. `input_layout` supports BHWC or BCHW.
`model(x, return_features=True)` also returns BCHW intermediate features with gradients.

The default student uses Swin/ViT stages. The registered Mamba/SSM stages are
available for explicit model-config changes; see [Mamba stages](model/card/MAMBA.md).
Intermediate-feature extraction/caching is an optional utility documented in
[Teacher features](tool/TEACHER_FEATURES.md); the current training loop uses
raw-head MSE directly. The loss is implemented in `tool/distillation.py:image_loss`.

## Current scope

The implemented path covers image loading, shared preprocessing, Teacher/student
forward passes, raw-head MSE, backward, optimizer updates, validation and student
checkpoint saving. Student-only image inference and full-state training resume
are not implemented. Image-quality metrics, alternative losses and the choice
of SSM replacement are further research work.

The student is a structural baseline, not a numeric reconstruction of DLSS-NR.
It uses standard PyTorch normalization, FFN, sampling and attention operations;
original recovered weights cannot be loaded into it. The Teacher uses the pinned
recovered MLX-DLSS graph and validates logical weights. Agreement with NVIDIA's
kernels or the independently updated Intel port is not claimed. Feature
coordinates must be aligned explicitly before intermediate-feature distillation.

References:

- https://chatgpt.com/share/6ac4fb26-91f0-83ee-9e3c-26fb50a0e746
- https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/master/docs/ARCHITECTURE.md
