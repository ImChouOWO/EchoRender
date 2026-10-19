# EchoRender

Single-image distillation from a frozen recovered DLSS-NR Teacher into a trainable
SSM/Swin/ViT student. Images are independent: no sequence state or optical flow.

## Environment

From the project root, on Windows or Linux:

```shell
uv sync --locked
```

Python 3.12 is pinned. Linux/Windows use matching CUDA 12.8 PyTorch and torchvision
wheels; CPU execution is supported. The MLX-DLSS backend revision is pinned in
`pyproject.toml` and `uv.lock`. Do not update it without rechecking feature capture.
LPIPS uses official pretrained AlexNet by default and bundled LPIPS calibration
weights. The first validation initialization downloads the backbone through
PyTorch, with checksum verification. Set `TORCH_HOME` to a writable persistent
cache if needed; otherwise training uses the ignored `.cache/torch` path.

## Training configuration

Edit **`config/train.yaml`**, then run:

```shell
uv run --locked python train.py --config config/train.yaml
```

All paths resolve relative to the YAML file, not the shell directory. Set:

- `model.teacher_weights`: actual decoded logical safetensors with
  `fully_logical=true`, using the **71-block HT topology**. Packed/small checkpoints
  are unsupported for this student.
- `dataset.root`: extracted MPI-Sintel-complete directory. Defaults to
  `../MPI-Sintel-complete`, relative to `config/`.
- `runtime.output_dir`: a fresh directory for each run.
- `runtime.device`: `auto`, `cpu`, `cuda`, or `cuda:0`.

The old per-parameter training CLI and separate `config/data.yaml` have been
removed. Dataset splits, loader settings, seed, optimizer, crop, losses, metrics,
latency benchmark, logging and schedule all live in `train.yaml`. Model structure
remains in `config/model.yaml`.

The Sintel default is the `clean` pass, with 19 training scenes and four held-out
alley/temple validation scenes. Train/validation scenes cannot overlap; the official
test split is separate. `clean` and `final` are render passes, not an automatic
input/GT pair. **Both RGB reconstruction and evaluation use the Teacher's composed
RGB as reference**, per the current training design. These metrics measure
Teacher fidelity; they do not establish improvement against independent ground truth.

The default crop is a shared 320x320 center crop. Teacher preprocessing reflects
small images to minimum-320/multiple-of-64 geometry, but RGB/head losses exclude
padding. Five-scale MS-SSIM with window 11 requires real crop height/width >160;
small images fail explicitly. Change the configured scales/window if intentionally
using smaller images. Validation LPIPS requires real images at least 64x64.

## Student and losses

E1/E2/E3 use the project's custom pure-PyTorch **SSMStage**, with unchanged widths
64/128/256 and depths 4/6/8. The configured `cross` scan runs shared bidirectional
row/column mixers, resetting state per line. E0/E4 remain Swin, the bottleneck
remains ViT, and decoder stages remain Swin. See [SSM modules](model/card/MAMBA.md).
The sequential selective scan is a research implementation; benchmark actual
latency/memory rather than assuming it is faster than attention.

All losses are defined in **`model/card/loss.py`**:

```text
E1, E2, E3 = channel-L2-normalized feature MSE
ViT        = 0.5 * feature MSE + cosine token-affinity MSE (diagonal excluded)
Lmid       = (E1 + E2 + E3 + ViT) / 4
Lout       = raw four-channel head MSE
Ld         = 0.5 * Lmid + 0.5 * Lout
Lrgb       = 0.5 * RGB MSE + 0.2 * (1 - SSIM) + 0.3 * (1 - MS-SSIM)
Ltotal     = 3 * Ld + Lrgb
```

The Teacher is frozen/eval and runs under `no_grad`. Optional identity-initialized
1x1 student projections accommodate differing feature bases. They are optimized
and saved with the training checkpoint. Spatial grids differ between the recovered
Teacher and structural student; [feature alignment](tool/TEACHER_FEATURES.md)
documents observation points and padding handling.

RGB composition uses the Teacher's learned blend cap for both outputs. Its
forward retains FP16 rounding; backward uses a straight-through estimator for
that rounding, so RGB losses can train the student. Clipping/sigmoid retain their
actual gradients. This is an explicit training surrogate, not NVIDIA numeric equivalence.

## Progress, evaluation and outputs

`tqdm` displays epoch/total, current loss, running image-weighted epoch loss,
iteration rate and **GPU utilization (%)**, not VRAM usage. NVML is polled no more
than once per configured interval and uses the CUDA device UUID. CPU or unavailable
NVML displays `N/A`, never a fabricated 0%. Utilization reflects the whole device.

Each epoch evaluates the full validation split with PSNR, SSIM and pretrained
LPIPS. Both image means and equal-weight scene means are saved. PSNR for identical
images is infinite, serialized as the string `"Infinity"` in strict JSON.

The optional latency benchmark separately measures Teacher and student float32
batch-1 forward passes on the first validation sample's prepared dimensions.
It uses warm-up, CUDA synchronization, eval mode and no feature capture. Timings
exclude loading, preprocessing, composition and backward. They are measured once
per epoch, not inferred from training iteration rate.

Outputs:

- `metrics.jsonl`: every step's raw/weighted loss components, current controls,
  GPU utilization, and epoch validation/benchmark results.
- `last.pt`: v2 student, feature adapters, optimizer, progress and configurations.
- `summary.json`: epoch results, parameter counts and peak CUDA allocation.
- `resolved_train.yaml`: resolved paths and settings used for the run.

`max_steps` limits optimizer updates, but still runs full validation and saves a
checkpoint. `model.student_weights` initializes only the student from a v2 checkpoint;
it starts a new run with fresh optimizer/adapters. Full-state resume and legacy
checkpoint compatibility are intentionally not provided.

## Reserved scheduling interface

`tool/training_config.py:TrainingSchedule.values(epoch=..., step=...)` is called
before every optimizer update. It currently returns fixed `lr`, `grad_clip` and
all loss coefficients from YAML. The trainer applies these values and logs them.
`step` is the number of updates already completed; `epoch` starts at 1.

Only `schedule.kind: constant` is supported. Add future strategies and their
configuration validation at that entry point; no automatic ramp/decay is enabled.

## Teacher-only inference

```shell
uv run --locked python infer.py --input input.png --weights /path/to/dlssnr-logical.safetensors --output outputs/teacher.png
```

Inference saves RGB at original image dimensions. Existing outputs require
`--overwrite`; `--report` saves metadata. See [Teacher](docs/TEACHER_INFERENCE.md).
Student-only image CLI inference remains outside the implemented training workflow.

## Verification

```shell
uv run --locked python -m unittest tool.test_training -v
```

Tests cover loss formulas/gradients, spatial padding alignment, Teacher capture,
RGB composition parity and a small synthetic training/validation/checkpoint loop.
Synthetic fixtures do not validate real Teacher checkpoint quality or dataset
training. Actual runs require local Teacher weights and Sintel files.
