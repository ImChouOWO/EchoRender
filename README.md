# EchoRender

[繁體中文](CHT-README.md)

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-orange) ![Architecture](https://img.shields.io/badge/Student-SSM%20%2F%20Swin%20%2F%20ViT-purple) ![License](https://img.shields.io/badge/License-MIT-green)

Single-image knowledge distillation from a frozen recovered DLSS-NR Teacher into a trainable PyTorch Student. Images are independent, with no optical flow or cross-frame state.

```text
Clean image -> frozen Teacher -> cached head / features / RGB targets
Same crop -> optional augmentation -> Student -> distillation + RGB losses
```

<details>
<summary><em>Current limitations</em></summary>

The Student is a structural research model, not a numerically equivalent NVIDIA implementation. Its custom SSM uses a sequential PyTorch scan rather than an official fused Mamba/VMamba kernel. Metrics measure agreement with Teacher RGB, not quality against independent ground truth. Full-state training resume is not implemented; `val.py` provides checkpoint validation and comparison-image export.

</details>

---

## Overview

The Teacher uses decoded logical safetensors with `fully_logical=true` and the 71-block HT topology, executed through the pinned MLX-DLSS PyTorch backend. Packed and small-topology checkpoints are unsupported by this training workflow.

### Student architecture

Defined in [config/model.yaml](config/model.yaml). Channels and depths below describe the current configuration, not Teacher block indices.

| Stage | Module | Channels | Blocks |
|---|---|---:|---:|
| E0 | Swin | 32 | 2 |
| E1 | SSM | 32 | 2 |
| E2 | SSM | 64 | 4 |
| E3 | SSM | 128 | 6 |
| E4 | Swin | 512 | 8 |
| Bottleneck | ViT | 1024 | 8 |
| Decoder E4 / E3 / E2 / E1 / E0 | Swin | 256 / 128 / 64 / 32 / 32 | 8 / 8 / 6 / 4 / 5 |

E4 projects 256 input channels to 512. Its first decoder upsample also outputs 512 channels for skip addition; the next Swin stage projects to 256. All `Add` inputs must have equal shapes.

E1/E2/E3 and the bottleneck supply distillation features. Trainable 1x1 projections map Student widths to Teacher widths 64/128/256/1024. Shared channels initialize as identity; extra output channels initialize to zero. Unequal widths require `loss.projection: true`.

---

## Project structure

```text
EchoRender/
├── README.md / CHT-README.md
├── config/model.yaml             # Student architecture
├── config/train.yaml             # Training strategy and dataset
├── model/card/                   # SSM, attention modules and losses
├── model/student.py / teacher.py
├── tool/                        # Data, cache, training and monitoring
├── docs/TEACHER_INFERENCE.md
├── weights/                     # Local Teacher weights; ignored by Git
├── output/feats/                 # Persistent Teacher targets; ignored by Git
├── train.py / val.py / infer.py
└── pyproject.toml / uv.lock
```

---

## Installation

### 1. Create the environment

From the project root:

```bash
uv sync --locked
```

Python 3.12 is pinned. Linux and Windows use matching CUDA 12.8 PyTorch/torchvision wheels; CPU execution is also supported. The MLX-DLSS revision is pinned in `pyproject.toml` and `uv.lock`; backend changes require feature-capture verification.

### 2. Prepare local assets

Place decoded Teacher weights at `weights/dlssnr-logical.safetensors` and extract MPI-Sintel-complete. Set `dataset.root` in the existing training YAML to your dataset path. LPIPS defaults to official pretrained AlexNet and downloads its backbone on first initialization. Training defaults `TORCH_HOME` to `.cache/torch` unless already set.

---

## Dataset format

```text
MPI-Sintel-complete/
├── training/clean/<scene>/frame_0001.png
├── training/final/<scene>/frame_0001.png
└── test/clean/<scene>/frame_0001.png
```

The current configuration uses the `clean` pass: 19 training scenes and four validation scenes (`alley_1`, `alley_2`, `temple_2`, `temple_3`), with 864/200 images in the current local dataset. Training and validation scenes cannot overlap; official test data remains separate. `clean` and `final` are render passes, not an automatic input/ground-truth pair.

The shared center crop is **192x192**. Teacher-compatible preprocessing pads to a minimum extent of 320 and multiples of 64, giving a **320x320 network input** for this crop. Head/RGB losses exclude padding. The configured five-scale MS-SSIM requires real dimensions above 160; LPIPS requires at least 64x64.

---

## Teacher feature cache

| `training.releas` | Behavior |
|---|---|
| `true` | Build or reuse targets, save fixed `blend_scale`, fully unload Teacher, then train Student from cache. |
| `false` | Keep a frozen/eval Teacher and Student on the selected GPU for online distillation; no cache access or deletion prompt. |

The existing `releas` key spelling is intentional. Cached targets live at project-root `output/feats/`: clean E1/E2/E3/ViT features, raw four-channel head and composed RGB. Targets are matched by split and sample ID; they are loss targets, not Student inputs.

Complete caches are reused across epochs and runs. Compatible partial caches resume per file; missing or truncated targets are rebuilt. Atomic writes update the index after each completed file, and a process lock protects the cache, including the cleanup prompt. Teacher weights, dataset identity, crop, preprocessing, capture/backend code, device and benchmark settings participate in validity. Student architecture, batch sizes, augmentation and optimizer changes do not invalidate clean targets.

Cache generation uses `torch.inference_mode()`, overlapping CPU reading/preparation, a single Teacher GPU forward and background writes. Reader/writer counts are estimated from the first sample and bounded by CPU/cgroup/RAM limits and queue capacity; they are separate from `loader.num_workers`.

Supported CUDA cache extraction uses a batch-1 CUDA Graph with precomputed attention biases. The first capture checks the head and all captured features for exact equality to the reference path. Unsupported captures report the reason and fall back; OOM is not silently ignored. Graph resources are cleared when Teacher is unloaded. Head-only latency benchmarks describe reference inference, not graph cache throughput.

After successful training:

```text
[Done]: Delete cached features? [Y/N]:
```

`Y` clears the cache; `N`, EOF or interruption retains it. Failed training preserves extraction progress.

---

## Model and training configuration

Edit the existing [config/train.yaml](config/train.yaml). Selected settings:

```yaml
model:
  student_config: model.yaml
  student_weights: null
  teacher_weights: weights/dlssnr-logical.safetensors
runtime:
  device: cuda:1
  cpu_threads: 4
  output_dir: output/test
training:
  epochs: 10
  gradient_checkpointing: false
  micro_batch_size: 4
  releas: true
  max_steps: null
  crop_size: 192
  optimizer:
    lr: 0.001
    momentum: 0.0
    weight_decay: 0.0
  grad_clip: 1.0
loader:
  batch_size: 16
  num_workers: 0
  seed: 42
  train_shuffle: true
schedule:
  kind: cosine
  eta_min: 0.000001
```

Teacher weights and `runtime.output_dir` resolve from the project root. Teacher weights must remain inside its `weights/` directory; `output/test` resolves to `EchoRender/output/test`. Other paths resolve relative to the training YAML. Existing run directories are preserved; new runs use `_2`, `_3`, etc. The selected absolute output path is printed and recorded.

`runtime.device` accepts `auto`, `cpu`, `cuda` and `cuda:N`. CUDA indices refer to visible devices, so `CUDA_VISIBLE_DEVICES` can remap them.

### Training strategy

`loader.batch_size` is the effective number of images per optimizer update. `training.micro_batch_size` is the maximum number forwarded together. The default runs **four groups of four images**, averages gradients over all 16 images, clips gradients and updates once. Short final batches use their actual image count. Different network extents are processed in separate groups without changing image geometry.

`training.gradient_checkpointing: true` recomputes each SSM block during backward to reduce saved activations. `false` retains activations and avoids that recomputation. Validation/inference do not use checkpointing.

In [config/model.yaml](config/model.yaml), `ssm.rotation` accepts `clockwise` or `counterclockwise`; `ssm.directions` accepts 1-4. Current two-direction clockwise blocks rotate **right+down → down+left → left+up → up+right**. Each stage restarts the cycle and each row/column resets its state. See [SSM details](model/card/MAMBA.md).

SGD uses no momentum or weight decay by default. Cosine LR is applied before each optimizer update over `epochs * len(train_loader)`, capped by `max_steps` when set. The first update uses 0.001 and the last uses 0.000001; a one-update run uses the initial LR. There is no warmup or restart. Loss weights remain fixed; cache generation and validation do not advance the schedule. `kind: constant` disables annealing.

### Student-only augmentation

Each selected training image receives Gaussian noise plus one uniformly selected
adjustment: brightness, contrast or sharpness. The scheduled ratio controls both
selection probability and strength: `factor = 1 + ratio`, followed by Gaussian
noise with `sigma = noise_std * ratio`. The defaults produce factors 1.10-1.25
and sigma 0.02-0.05. The ratio reaches 25% halfway through configured epochs and
switches off from 75% onward. For 10 epochs: epoch 1 uses 10%, epochs 5-7 use 25%,
and epochs 8-10 use 0%. Actual selection counts fluctuate around the probability.
`max_steps` does not rescale this epoch schedule.

```yaml
augmentation:
  enabled: true
  start_ratio: 0.10
  max_ratio: 0.25
  peak_fraction: 0.50
  stop_fraction: 0.75
  noise_std: 0.20
```

Teacher targets stay clean and frozen. Student preparation and RGB composition
use the augmented RGB. Clean validation scores, feature-cache reuse and benchmarks
retain their existing behavior. Augmentation has its own RNG seeded by
`loader.seed`; `augmentation.enabled: false` disables training augmentation.
Validation image reports use one mode at a time and the configured report strengths.

---

## Losses

Defined in [model/card/loss.py](model/card/loss.py):

```text
E1, E2, E3 = channel-L2-normalized feature MSE
ViT        = 0.5 * feature MSE + off-diagonal cosine token-affinity MSE
Lmid       = (E1 + E2 + E3 + ViT) / 4
Lout       = raw four-channel head MSE
Ld         = 0.5 * Lmid + 0.5 * Lout
Lrgb       = 0.5 * RGB MSE + 0.2 * (1 - SSIM) + 0.3 * (1 - MS-SSIM)
Ltotal     = 3 * Ld + Lrgb
```

Feature alignment excludes padding-only cells and preserves physical grid correspondence; see [feature capture and alignment](tool/TEACHER_FEATURES.md). Token affinity uses exact row chunks without token sampling. Teacher targets are detached; only Student and feature adapters enter the optimizer.

Both RGB branches use the Teacher's fixed learned blend cap. RGB composition preserves FP16 rounding in forward and uses a straight-through gradient for rounding only; clamp/sigmoid retain their actual gradients. This is a training surrogate, not NVIDIA numerical equivalence. Reconstruction and PSNR/SSIM/LPIPS all reference Teacher RGB.

---

## Run, monitor and verify

```bash
uv run --locked python train.py --config config/train.yaml
uv run --locked python -m unittest tool.test_training tool.test_reporting tool.test_val -v
```

Progress shows epoch, loss, running image-weighted loss, iteration rate and GPU utilization. NVML matches the configured CUDA device by UUID. Utilization covers the whole device; unavailable readings display `N/A`.

Every epoch evaluates the full validation split, recording image means and equal-weight scene means for PSNR/SSIM/LPIPS. Infinite PSNR is serialized as `"Infinity"`. The optional batch-1 FP32 latency benchmark uses warmup, synchronization and eval mode, excluding preprocessing, feature capture, composition and backward. Teacher reference timing is retained from cache creation in cached mode; Student timing is measured each epoch.

| Output | Contents |
|---|---|
| `metrics.jsonl` | Per-update losses, controls, augmentation counts, GPU utilization and epoch evaluation. |
| `best.pt` | Selected validation-best checkpoint; same payload as `last.pt`. |
| `last.pt` | v2 Student, feature adapters, optimizer, progress, model/train configurations and blend scale. |
| `summary.json` | Epoch results, parameter counts and peak CUDA allocation. |
| `resolved_train.yaml` | Resolved paths and run settings. |

`max_steps` caps optimizer updates while still running validation and checkpoint saving. `model.student_weights` loads only Student weights into a new run with fresh optimizer/adapters. Tests cover gradients, projection widths, scan directions, batching, cache lifecycle and scheduling; synthetic fixtures do not establish real-data model quality.

With the current E4-512 model, crop 192, micro-batch 4, effective batch 16 and checkpointing off, two short updates on GPU 1 (RTX 5000 Ada) reached about **29.14 GiB allocated / 29.81 GiB reserved** and took **8-9 seconds per update**. These observations are not a full-epoch benchmark or a memory guarantee.

Weights, checkpoints, Teacher caches and output directories are excluded by `.gitignore`. Keep `uv.lock` versioned for reproducibility.

---

## Teacher inference

```bash
uv run --locked python infer.py --input input.png --weights weights/dlssnr-logical.safetensors --output output/teacher.png --device cuda:1
```

Output PNGs keep the original image dimensions. Use `--overwrite` to replace an existing output and `--report path.json` for timings/metadata. See [Teacher inference](docs/TEACHER_INFERENCE.md).

---

## Best checkpoint and final reports

Training keeps only `best.pt` and `last.pt`. The default selection criterion is the
highest validation image-mean PSNR. `reporting.best_metric` also accepts `loss`,
`ssim` and `lpips`; loss/LPIPS minimize and PSNR/SSIM maximize. Ties retain the
previous checkpoint. A new best prints `[Best]: save best weight | ...` with signed
percentage improvements relative to the previous best checkpoint; positive means
improvement, and undefined percentages display `N/A`.

At completion, the terminal prints max/min/mean/best for train loss, validation
loss, PSNR, SSIM and LPIPS over epoch means. Each metric's independent best and its
value at the selected checkpoint are recorded separately. Reports live under
`<runtime.output_dir parent>/val/<project_name>/` (project-root `output/val/` with the default configuration); null uses the reserved run name and
existing report folders receive a numeric suffix.

```yaml
reporting:
  enabled: true
  project_name: null
  best_metric: psnr
  crop_size: 0
  inference_size: 320
  samples_per_mode: 2
  strengths: [0.25, 0.50, 0.75, 1.00]
```

`statistics.json`, `statistics.csv`, `epochs.csv` and `training_metrics.png` contain
statistics and the four training curves. Training uses fixed-column progress for
epoch, GPU memory, GPU utilization, losses, LR and image counts; validation adds
PSNR, SSIM and LPIPS.

After statistics, the best Student and frozen Teacher run sequentially on validation
and test images. Each split randomly selects two sources for each of five modes:
original, noise, brightness, contrast and sharpness. This gives 10 sources per split
when enough data is available. Selection is reproducible with `loader.seed`; both
models use the same sources and inputs. Smaller splits reuse sources across modes,
while each mode contains at most `samples_per_mode` distinct sources. Clean scoring
still processes the complete split.

Validation variants use one operation at a time. The four noise sigmas are
0.05/0.10/0.15/0.20; brightness, contrast and sharpness use factors
1.25/1.50/1.75/2.00. Noise uses a fixed pattern scaled across strengths.
Each non-original grid includes an original column followed by the four strengths.
The original mode has one column. Each filename identifies one source image.

```text
output/val/<project_name>/
├── statistics.json / statistics.csv / epochs.csv / training_metrics.png
├── predictions.json / predictions.jsonl / predictions.csv
├── preview/<mode>/<split>__<sample_id>.png
└── pred/
    ├── student/<mode>/<split>__<sample_id>.png
    └── teacher/<mode>/<split>__<sample_id>.png
```

Each preview contains one input row. Each prediction grid contains the same inputs
in row one and their corresponding predictions in row two. Bold English titles
include `| total infer 1.234 s | avg infer 0.247 s |`. Times are synchronized model
forward seconds, excluding one warmup per input shape, preprocessing, RGB
composition and file I/O. Total and average cover the displayed columns.
Panels preserve aspect ratio and use up to 384 pixels of display width.

`reporting.crop_size: 0` keeps the full source image. The inference longest edge is
bounded by `reporting.inference_size: 320`, with no upscaling of smaller images.
Augmentation runs after resize and before feature preparation, so its noise is not
attenuated by downsampling. The same input change is mapped to original size for
display. RGB composition finishes at inference resolution before its correction
is mapped onto the original-size augmented input. Set inference_size to 0 for
native inference; a positive crop_size center-crops before this process.

Manifests record source selection, augmentation parameters, layout geometry,
input/feature/prediction hashes and timings. `predictions.csv` includes each
variant's inference time and prediction RMSE relative to its original-input
prediction. Missing test data is recorded as `completed_with_missing_splits`;
available images still export. Failed exports retain the feature cache, and the
cleanup prompt runs only after reports finish. `reporting.enabled: false` skips
final image/plot export while retaining best/last and printed statistics.

These features apply to new training processes. Existing runs that only saved
`last.pt` cannot reconstruct an earlier best checkpoint.

---

## Standalone validation and prediction export

Run `val.py` beside `train.py` to validate an existing checkpoint and regenerate
comparison images without training:

```bash
uv run --locked python val.py --weights output/test_12/last.pt --project my_validation --imgsz 320
# Quick validation-only check, limited to one image:
uv run --locked python val.py --weights output/test_12/last.pt --project preview --split val --max-images 1
```

Use `best.pt` for the selected best model or `last.pt` for the latest model. Model
structure, feature adapters, validation crop and loss settings come from the
checkpoint. Dataset paths, device, augmentation and comparison settings come from
`--config config/train.yaml`; `--device cuda:1` overrides the configured device.

`--project` names the report directory. With `runtime.output_dir: output/test`,
reports go to `output/val/<project>/`. Repeated names receive `_2`, `_3`, etc.,
without overwriting previous reports. `--split` accepts `val`, `test` or `both`
(default). `--max-images N` limits each selected split, including both fresh scoring
and image export. Explicitly requested missing splits fail; `both` reports missing
splits and processes the available data.

`validation.json` contains freshly recomputed clean-image loss/PSNR/SSIM/LPIPS
against Teacher RGB and the selected checkpoint epoch. Historical curves/statistics
are reconstructed from `summary.json` beside the checkpoint when available; their
best epoch is independent of the chosen `last.pt`. If no history exists, the CLI
still scores and exports images without fabricating a training curve. Validation
uses the checkpoint's training crop; comparison images use `reporting.crop_size`
and resized inference controlled by `reporting.inference_size` or `--imgsz 320`.
The existing Teacher feature cache is not modified or deleted.

---

## License

Project code is licensed under [MIT](LICENSE). External dependencies, datasets and Teacher weights retain their own terms.
