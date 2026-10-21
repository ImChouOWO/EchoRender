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

The Student is a structural research model, not a numerically equivalent NVIDIA implementation. Its custom SSM uses a sequential PyTorch scan rather than an official fused Mamba/VMamba kernel. Metrics measure agreement with Teacher RGB, not quality against independent ground truth. Full-state training resume and a Student image-inference CLI are not implemented.

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
├── train.py / infer.py
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

Teacher weights resolve from the project root and must remain inside its `weights/` directory. Other paths resolve relative to the training YAML: `output/test` therefore becomes `config/output/test`. Existing run directories are preserved; new runs use `_2`, `_3`, etc. The selected absolute output path is printed and recorded.

`runtime.device` accepts `auto`, `cpu`, `cuda` and `cuda:N`. CUDA indices refer to visible devices, so `CUDA_VISIBLE_DEVICES` can remap them.

### Training strategy

`loader.batch_size` is the effective number of images per optimizer update. `training.micro_batch_size` is the maximum number forwarded together. The default runs **four groups of four images**, averages gradients over all 16 images, clips gradients and updates once. Short final batches use their actual image count. Different network extents are processed in separate groups without changing image geometry.

`training.gradient_checkpointing: true` recomputes each SSM block during backward to reduce saved activations. `false` retains activations and avoids that recomputation. Validation/inference do not use checkpointing.

In [config/model.yaml](config/model.yaml), `ssm.rotation` accepts `clockwise` or `counterclockwise`; `ssm.directions` accepts 1-4. Current two-direction clockwise blocks rotate **right+down → down+left → left+up → up+right**. Each stage restarts the cycle and each row/column resets its state. See [SSM details](model/card/MAMBA.md).

SGD uses no momentum or weight decay by default. Cosine LR is applied before each optimizer update over `epochs * len(train_loader)`, capped by `max_steps` when set. The first update uses 0.001 and the last uses 0.000001; a one-update run uses the initial LR. There is no warmup or restart. Loss weights remain fixed; cache generation and validation do not advance the schedule. `kind: constant` disables annealing.

### Student-only augmentation

Each selected image receives Gaussian noise plus one uniformly selected adjustment: brightness, contrast or sharpness. Probability starts at 10%, reaches 25% at 50% of configured epochs and switches off from 75% onward. For 10 epochs: epoch 1 is 10%, epochs 5-7 are 25%, and epochs 8-10 are 0%. The cap is a sampling probability, not a guaranteed per-batch count. Short runs may not complete the ramp; `max_steps` does not rescale it.

Noise sigma is sampled from 0-0.02; brightness/contrast factors from 0.8-1.2, sharpness from 0.5-1.5. Teacher targets stay clean. Student preparation and RGB composition use the augmented image. Validation, benchmarks and inference are clean. Augmentation uses its own RNG seeded by `loader.seed`; set `augmentation.enabled: false` to disable it.

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
uv run --locked python -m unittest tool.test_training -v
```

Progress shows epoch, loss, running image-weighted loss, iteration rate and GPU utilization. NVML matches the configured CUDA device by UUID. Utilization covers the whole device; unavailable readings display `N/A`.

Every epoch evaluates the full validation split, recording image means and equal-weight scene means for PSNR/SSIM/LPIPS. Infinite PSNR is serialized as `"Infinity"`. The optional batch-1 FP32 latency benchmark uses warmup, synchronization and eval mode, excluding preprocessing, feature capture, composition and backward. Teacher reference timing is retained from cache creation in cached mode; Student timing is measured each epoch.

| Output | Contents |
|---|---|
| `metrics.jsonl` | Per-update losses, controls, augmentation counts, GPU utilization and epoch evaluation. |
| `last.pt` | v2 Student, feature adapters, optimizer, progress, model/train configurations and blend scale. |
| `summary.json` | Epoch results, parameter counts and peak CUDA allocation. |
| `resolved_train.yaml` | Resolved paths and run settings. |

`max_steps` caps optimizer updates while still running validation and checkpoint saving. `model.student_weights` loads only Student weights into a new run with fresh optimizer/adapters. Tests cover gradients, projection widths, scan directions, batching, cache lifecycle and scheduling; synthetic fixtures do not establish real-data model quality.

With the current E4-512 model, crop 192, micro-batch 4, effective batch 16 and checkpointing off, two short updates on GPU 1 (RTX 5000 Ada) reached about **29.14 GiB allocated / 29.81 GiB reserved** and took **8-9 seconds per update**. These observations are not a full-epoch benchmark or a memory guarantee.

Weights, checkpoints, Teacher caches and output directories are excluded by `.gitignore`. Keep `uv.lock` versioned for reproducibility.

---

## Teacher inference

```bash
uv run --locked python infer.py --input input.png --weights weights/dlssnr-logical.safetensors --output outputs/teacher.png --device cuda:1
```

Output PNGs keep the original image dimensions. Use `--overwrite` to replace an existing output and `--report path.json` for timings/metadata. See [Teacher inference](docs/TEACHER_INFERENCE.md).

---

## License

Project code is licensed under [MIT](LICENSE). External dependencies, datasets and Teacher weights retain their own terms.
