# EchoRender

[繁體中文](CHT-README.md)

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-orange) ![Architecture](https://img.shields.io/badge/Student-SSM%20%2F%20Swin%20%2F%20ViT-purple) ![License](https://img.shields.io/badge/License-MIT-green)

Controlled RGB sequence distillation from a frozen recovered DLSS-NR Teacher into a PyTorch Student. Clean Sintel clips contain 1–10 consecutive frames; four rendering profiles share one conditioned Student.

```text
Clean RGB clip + Sintel flow -> frozen Teacher per profile -> cached targets + history
Same RGB clip -> optional corruption -> raw/RAFT-Small history alignment -> BHWC8 Student -> losses
```

<details>
<summary><em>Current limitations</em></summary>

The Student is a structural research model, not a numerically equivalent NVIDIA implementation. CUDA SSM uses the official fused Mamba selective-scan kernel with FP32 recurrence. Student linear, convolution and attention operations use BF16 when training.precision is bf16; RGB composition and losses remain FP32. Metrics measure agreement with Teacher RGB, not quality against independent ground truth. `--resume` restores epoch-boundary training state; `val.py` provides checkpoint validation and comparison-image export.

</details>

---

## Overview

The Teacher uses decoded logical safetensors with `fully_logical=true` and the 71-block HT topology, executed through the pinned MLX-DLSS PyTorch backend. Packed and small-topology checkpoints are unsupported by this training workflow.

### Student architecture

Defined in [config/model.yaml](config/model.yaml). Callers supply RGB `[0,1]`; preprocessing directly builds `input_channels: 8`: current RGB in channels 0-2, confidence-mixed aligned Student history RGB in 3-5, normalized style in 6 and resolved tone in 7. The first frame uses current RGB for both image slots. Fixed noise, constant and structure-control channels are removed from the Student. Structure/skin/auto settings remain fixed Teacher-target settings; changing them requires new targets and retraining. `tone: null` resolves to 1 for standard/natural/cinematic and 0 for neutral, so tone remains necessary to distinguish the configured cinematic and neutral modes, which both use style 2. The independent raw/RAFT-Small branch predicts displacement/confidence before the input adapter, without stage skip connections. Intensity, colour/detail strength and radius remain output-composition settings. Teacher inputs remain BHWC16. The root InputAdapter resizes the assembled full-field Student input to `training.adapter_size` (320x320 by default). Distillation taps default to E0/E1/E2/E3/E4/VIT. Checkpoints use `echorender-bhwc8-raft-small-encoder-bf16-v8`; previous checkpoints and crop-based targets require fresh training/extraction. Channels/depths below describe the Student, not Teacher block indices.

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

Named taps are defined by `stages` in the model YAML. `training.distillation_layers: [RGB composition, E0, E1, E2, E3, E4, VIT]` selects final RGB and encoder/global supervision. Decoder stages remain present. Trainable 1x1 projections map selected Student widths to Teacher widths 32/64/128/256/512/1024. Shared channels initialize as identity; extra output channels initialize to zero. Unequal widths require `loss.projection: true`.

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

Python 3.12 is pinned. CUDA training requires Linux, matching PyTorch/torchvision wheels and a CUDA toolkit to build mamba-ssm 2.2.6.post3. Set CUDA_HOME to the toolkit directory when needed. The CPU recurrence remains available for numerical tests. Transformers 4.44.2 is pinned for the Mamba package import contract. The MLX-DLSS revision is pinned in `pyproject.toml` and `uv.lock`; backend changes require feature-capture verification.

### 2. Prepare local assets

Place decoded Teacher weights at `weights/dlssnr-logical.safetensors` and extract MPI-Sintel-complete. Set `dataset.root` in the existing training YAML to your dataset path. LPIPS defaults to official pretrained AlexNet and downloads its backbone on first initialization. Training defaults `TORCH_HOME` to `.cache/torch` unless already set.

---

## Dataset format

```text
MPI-Sintel-complete/
├── training/clean/<scene>/frame_0001.png
├── training/final/<scene>/frame_0001.png
├── training/flow/<scene>/frame_0001.flo
├── training/occlusions/<scene>/frame_0001.png
├── training/invalid/<scene>/frame_0001.png
└── test/clean/<scene>/frame_0001.png
```

The current configuration uses the `clean` pass: 19 training scenes and four validation scenes (`alley_1`, `alley_2`, `temple_2`, `temple_3`), with 864/200 images in the current local dataset. Training and validation scenes cannot overlap; official test data remains separate. `dataset.render_pass` is fixed to `clean`; `final` is rejected. Flow/occlusion/invalid guides are required for consecutive frames. Test images without matching temporal guides cannot be scored as a Sintel temporal split. Clips never cross a scene, frame gap, or ten-frame boundary; only whole clips are shuffled.

Training, validation and prediction preserve the entire image. `training.adapter_size: 320` replaces crop settings: the root InputAdapter resizes the assembled eight-channel input to 320x320 before projection. Stage channel adapters preserve their own grids. Teacher clean RGB and aligned history use the same full-image target extent, without changing its frozen graph. Motion/validity remain in normalized full-image coordinates; composed outputs map back to native size. Five-scale MS-SSIM requires real output dimensions above 160; LPIPS requires at least 64x64.

---

## Teacher feature cache

| `training.releas` | Behavior |
|---|---|
| `true` | Build or reuse targets, save fixed `blend_scale`, fully unload Teacher, then train Student from cache. |
| `false` | Keep a frozen/eval Teacher and Student on the selected GPU for online distillation; no cache access or deletion prompt. |

The existing `releas` key spelling is intentional. Cached targets live at project-root `output/feats/`: clean selected-stage features (default E0/E1/E2/E3/E4/VIT), raw four-channel head and composed RGB. Targets are matched by profile, split, clip context and frame ID. Each frame group stores its raw head, selected features, unaugmented source RGB, RGB target, pre-display neural history and identity metadata in `output/feats/<profile>/<split>/<source>/<scene>/<frame>/targets.pt`. They are targets, not Teacher-forced Student history.

Complete caches are reused across epochs and runs. Compatible partial caches resume per file; missing or truncated targets are rebuilt. Atomic writes update the index after each completed file, and a process lock protects the cache, including the cleanup prompt. Teacher weights, dataset identity, adapter size, selected distillation stages, preprocessing, capture/backend code, device and benchmark settings participate in validity. Rendering/profile controls and sequence boundaries also participate in validity. Student widths/depths, batch sizes, augmentation and optimizer changes do not invalidate clean targets when selected taps and target geometry are unchanged. Four profiles produce approximately four times the targets at unchanged geometry; old flat files are left untouched and are not indexed by new extraction.

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
  epochs: 100
  profiles: [standard, natural, cinematic, neutral]
  early_stopping:
    patience: 15
    min_delta: 0.001
  precision: bf16
  gradient_checkpointing: false
  micro_batch_size: 4
  releas: true
  max_steps: null
  adapter_size: 320
  distillation_layers: [RGB composition, E0, E1, E2, E3, E4, VIT]
  optimizer:
    lr: 0.001
    momentum: 0.0
    weight_decay: 0.0
  grad_clip: 1.0
loader:
  batch_size: 16
  num_workers: 2
  seed: 42
  train_shuffle: true
schedule:
  kind: cosine
  eta_min: 0.000001
```

Teacher weights and `runtime.output_dir` resolve from the project root. Teacher weights must remain inside its `weights/` directory; `output/test` resolves to `EchoRender/output/test`. Other paths resolve relative to the training YAML. Existing run directories are preserved; new runs use `_2`, `_3`, etc. The selected absolute output path is printed and recorded.

`runtime.device` accepts `auto`, `cpu`, `cuda` and `cuda:N`. CUDA indices refer to visible devices, so `CUDA_VISIBLE_DEVICES` can remap them.

### Training strategy

`loader.batch_size` counts clips with one profile each, not individual frames. `training.micro_batch_size` limits concurrent frames. Each clip contributes its frame-mean loss, then gradients are averaged across the effective batch. RGB history is detached after every frame (one-frame truncated backpropagation), keeping saved activation memory independent of clip length. Longer clips still require more forwards/backwards. Training samples prefix lengths uniformly between `train_min_frames` and the available clip length; evaluation processes the whole clip.

`training.profiles` selects any nonempty subset of the four profiles, for example `[standard, neutral]`. Training, validation, checkpoints and prediction exports use this selection. Keep all four control definitions under `rendering.profiles`; unselected profiles are not processed. Retained matching Teacher targets are reused by profile, so adding a mode only extracts its missing targets. Changing a profile setting invalidates that profile; changing Teacher/data/preprocessing invalidates affected targets.

Early stopping follows `reporting.best_metric` (PSNR by default): `training.early_stopping.patience: 15` stops after 15 epochs without an absolute improvement greater than `min_delta: 0.001`. Set patience to 0 to disable. Early stopping still saves best/last, statistics, predictions and curves. `training_metrics.png` contains train/validation loss, PSNR, SSIM and LPIPS; `profile_metrics.png` compares selected profiles. Both are written to the run report directory under `output/val/`.

Rendering controls live in `rendering.profiles.standard/natural/cinematic/neutral`. Each profile defines `style_index`, `tone`, `structure`, `skin_structure`, `auto_structure`, `intensity`, `detail_strength`, `colour_strength` and `detail_radius`. Defaults use structure 1.8, skin 1.5, auto 1.0, intensity 1.8, detail 1.5, colour 1.0 and radius 0.5. Neutral explicitly selects style 2; the other profiles inherit their own style. `tone: null` follows the named profile. Training geometry is defined only by `training.adapter_size`; noise index is 1. The standalone Teacher demo retains its independent `render_scale` and `min_extent` arguments. `rendering.temporal: false` creates single-frame samples; `true` supports up to `max_frames: 10`. Teacher uses Sintel ground-truth guides for training/validation targets. Student motion is selected by `motion: raw` or `motion: raft_small` in `config/model.yaml`. `raw` estimates backward optical flow with external Farneback from consecutive RGB frames; `raft_small` uses pretrained torchvision RAFT-Small C_T_V2 at adapter resolution (12 updates), estimates current-to-previous flow, and adds a trainable validity head. The branch has no skip connections to backbone/decoder stages; masked flow supervision and differentiable RGB warping train the branch with `loss.motion: 0.1`. The external source remains RGB; the assembled Student tensor has 8 channels. Neither Student mode requires dataset flow during inference; first frames and scene cuts reset history. Test targets without Sintel guides use externally estimated RGB flow, explicitly distinct from ground-truth motion. Artist control masks remain unsupported.

`training.gradient_checkpointing: true` recomputes each SSM block during backward to reduce saved activations. `false` retains activations and avoids that recomputation. Validation/inference do not use checkpointing.

In [config/model.yaml](config/model.yaml), `ssm.rotation` accepts `clockwise` or `counterclockwise`; `ssm.directions` accepts 1-4. Current two-direction clockwise blocks rotate **right+down → down+left → left+up → up+right**. Each stage restarts the cycle and each row/column resets its state. See [SSM details](model/card/MAMBA.md).

SGD uses no momentum or weight decay by default. Cosine LR is applied before each optimizer update over `epochs * len(train_loader)`, capped by `max_steps` when set. The first update uses 0.001 and the last uses 0.000001; a one-update run uses the initial LR. There is no warmup or restart. Loss weights remain fixed; cache generation and validation do not advance the schedule. `kind: constant` disables annealing.

### Student-only augmentation

Each selected training clip receives Gaussian noise plus one uniformly selected
adjustment: brightness, contrast or sharpness. The scheduled ratio controls both
selection cap and strength: `factor = 1 + ratio`, followed by Gaussian
noise with `sigma = noise_std * ratio`. The defaults produce factors 1.10-1.25
and sigma 0.02-0.05. The ratio reaches 25% halfway through configured epochs and
switches off from 75% onward. For 10 epochs: epoch 1 uses 10%, epochs 5-7 use 25%,
and epochs 8-10 use 0%. Selection is rounded down and bounded by both clip count and frame count, so at least 75% remain original. Frames within a selected clip share the adjustment and strength; noise realizations may differ.
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
use the augmented RGB. Motion estimation pairs the actual preceding Student input with the current input, including augmentation, and warps the Student's own detached pre-display prediction. Clean validation scores, feature-cache reuse and benchmarks
retain their existing behavior. Augmentation has its own RNG seeded by
`loader.seed`; `augmentation.enabled: false` disables training augmentation.
Validation image reports use one mode at a time and the configured report strengths.

---

## Losses

Defined in [model/card/loss.py](model/card/loss.py):

```text
E0-E4      = channel-L2-normalized feature MSE
VIT        = 0.5 * feature MSE + off-diagonal cosine token-affinity MSE
Lmid       = mean(feature_weights[stage] * loss[stage] for selected stages)
Lout       = 0.75 * raw RGB-head MSE + 0.05 * raw gate-logit MSE
Ld         = 0.5 * Lmid + 0.5 * Lout
Lrgb       = 1.0 * weighted RGB MSE + 0.5 * (1 - weighted SSIM) + 0.8 * (1 - weighted MS-SSIM) + 0.05 * LPIPS
Ltotal     = 3 * Ld + 2 * Lrgb + motion_weight * Lmotion  # raft_small only; motion_weight = 0.1
```


Raw-head weights are separate direct coefficients; they are not normalized. LPIPS compares final composed Student/Teacher RGB using a frozen pretrained network in eval mode, with RGB mapped from [0,1] to [-1,1]. Gradients pass through LPIPS to Student RGB; Teacher targets and LPIPS parameters remain frozen. `rgb_lpips` belongs to the reconstruction group, so the current effective coefficient is `2.0 * 0.05 = 0.1`. Its standard image-level reduction is independent of the spatial importance used by MSE/SSIM/MS-SSIM. Loss terms and weighted contributions are logged separately, including `output_rgb`, `output_gate` and `rgb_lpips`.

```yaml
loss:
  reconstruction: 2.0
  output_rgb: 0.75
  output_gate: 0.05
  rgb_lpips: 0.05
  lpips_net: alex
```

Set `rgb_lpips: 0` to disable perceptual training and avoid loading its network. `validation.lpips_net` independently selects the evaluation metric. LPIPS training needs native RGB dimensions of at least 64x64. New loss settings apply to a new run; resume requires the original loss settings and checkpoint format. Teacher targets remain reusable because these loss changes do not alter their generation.

Teacher-change importance uses the unaugmented source: `delta = mean(abs(Teacher - clean_RGB), channels)` and `w = 1 + alpha * clamp(delta / tau, 0, 1)`. The current `loss.teacher_change` settings are `alpha: 2.0`, `tau: 0.1`, giving weights from 1 to 3. Importance is detached and normalized per image by its weight sum. SSIM/MS-SSIM apply importance to local score maps at each scale; RGB inputs are never masked. Set alpha to 0 for uniform weighting. Quality metrics remain ordinary unweighted PSNR/SSIM/LPIPS.

Feature alignment excludes padding-only cells and preserves physical grid correspondence; see [feature capture and alignment](tool/TEACHER_FEATURES.md). Token affinity uses exact row chunks without token sampling. Teacher targets are detached; only Student and feature adapters enter the optimizer.

Both temporal RGB branches use the Teacher's fixed learned blend cap; the first frame has no history gate blend. Raw heads are mapped before native RGB composition, followed by the configured grade, intensity and detail/colour split. RGB composition preserves FP16 rounding in forward and uses a straight-through gradient for rounding only; clamp/sigmoid retain their actual gradients. This is a training surrogate, not NVIDIA numerical equivalence. Reconstruction and PSNR/SSIM/LPIPS all reference Teacher RGB.

---

## Run, monitor and verify

```bash
uv run --locked python train.py --config config/train.yaml
uv run --locked python -m unittest tool.test_training tool.test_reporting tool.test_val tool.test_teacher_infer tool.test_temporal tool.test_pipeline -v
```

Resume an unfinished run without reserving a new output directory:

```bash
uv run --locked python train.py --config config/train.yaml --resume output/test/last.pt
```

Resume starts at the next epoch and restores Student, feature adapters, SGD, global step, cosine horizon, augmentation/loader RNG, metric history, best selection and early stopping. Keep the run's `best.pt` and `metrics.jsonl` beside `last.pt`; core dataset/model/loss/schedule settings must match. `epochs` remains the original total. Resume requires complete RNG and progress state; incomplete and earlier-format checkpoints are rejected. An output lock prevents two trainers from writing the same run. `model.student_weights` remains a separate weights-only start and cannot be combined with `--resume`.

Student neural history and RAFT RGB pairs remain on the selected device. RAFT pairs with matching dimensions are batched without changing the network, warp or per-image loss. Training groups sampled clips by length after drawing their augmentation noise, retaining the same sampled prefixes, images and clip-mean loss weights. Cached targets use two CPU reader threads with a bounded lookahead (up to twice the micro batch, at least two files, 256 MiB; one oversized file may exceed the byte cap), then pinned asynchronous copies to the selected GPU. Target loading retains the configured selected taps and clean source reference.

After `best.pt` is successfully saved, the `[Best]` message and metric changes appear in bold bright yellow in a terminal; redirected logs stay plain, and `NO_COLOR` disables styling.

Progress shows epoch, loss, running clip-weighted loss, iteration rate and GPU utilization. NVML matches the configured CUDA device by UUID. Utilization covers the whole device; unavailable readings display `N/A`.

Every epoch evaluates the full validation split, recording image means and equal-weight scene means for PSNR/SSIM/LPIPS. Infinite PSNR is serialized as `"Infinity"`. The optional batch-1 latency benchmark (FP32 Teacher, configured Student precision) uses warmup, synchronization and eval mode, excluding preprocessing, feature capture, composition and backward. Teacher reference timing is retained from cache creation in cached mode; Student timing is measured each epoch.

| Output | Contents |
|---|---|
| `metrics.jsonl` | Per-update losses, controls, augmentation counts, GPU utilization and epoch evaluation. |
| `best.pt` | Selected validation-best checkpoint; same payload as `last.pt`. |
| `last.pt` | v3 conditioned RGB Student, feature adapters, optimizer, progress, model/train configurations and blend scale. |
| `summary.json` | Epoch results, parameter counts and peak CUDA allocation. |
| `resolved_train.yaml` | Resolved paths and run settings. |

`max_steps` caps optimizer updates while still running validation and checkpoint saving. `model.student_weights` loads only Student weights into a new run with fresh optimizer/adapters. Tests cover gradients, projection widths, scan directions, batching, cache lifecycle and scheduling; synthetic fixtures do not establish real-data model quality.

A short full-field GPU 1 check (RTX 5000 Ada, adapter 320, effective batch 16, micro batch 4, two frames per clip, 25% augmentation, checkpointing off) reached **13.91 GiB allocated / 14.21 GiB reserved**. This verifies one optimizer update, not full-epoch throughput or a memory guarantee for every clip length/image size.

Weights, checkpoints, Teacher caches and output directories are excluded by `.gitignore`. Keep `uv.lock` versioned for reproducibility.

---

## Teacher inference

Edit `demo()` in `teacher_infer.py`, then run:

```bash
uv run --locked python teacher_infer.py
```

Use `input_image` for a single image or `scene` for clean Sintel frames. `max_frames` is 1–10, motion is fixed to `sintel`, and artist control masks are not accepted by this API. `temporal: true` also accepts a single image as a history-free first frame. Output PNGs keep native dimensions. Teacher-only demo settings do not change training settings in `rendering`.
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
  samples_per_mode: 2
  strengths: [0.25, 0.50, 0.75, 1.00]
```

`statistics.json`, `statistics.csv`, `epochs.csv` and `training_metrics.png` contain
statistics and the four training curves. Training uses fixed-column progress for
epoch, GPU memory, GPU utilization, losses, LR and image counts; validation adds
PSNR, SSIM and LPIPS.

After statistics, the best Student and frozen Teacher run sequentially on validation
and test images. Each split randomly selects two sources for each of five modes:
original, noise, brightness, contrast and sharpness. One endpoint image per selected clip is exported for all profiles. This gives 10 source images per split
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
    ├── student/<profile>/<mode>/<split>__<sample_id>.png
    └── teacher/<profile>/<mode>/<split>__<sample_id>.png
```

Each preview contains one input row. Each prediction grid contains the same inputs
in row one and their corresponding predictions in row two. Bold English titles
include `| total infer 1.234 s | avg infer 0.247 s |`. Times synchronize GPU execution. Student timing includes motion, input preparation and backbone; Teacher timing measures its neural forward. RGB composition and file I/O are excluded. Total includes the context frames for every displayed variant; average divides by all actual frame forwards, including context.
Panels preserve aspect ratio and use up to 384 pixels of display width.

Comparison inputs use native full-image RGB with one deterministic augmentation per mode/strength. The Student root InputAdapter and Teacher target preparation share the checkpoint's adapter size. Raw heads are mapped back to native RGB for composition and display; the displayed input is exactly the augmented source supplied to that path. Spatial downsampling can reduce high-frequency input noise. There are no separate report crop/resize settings.

Manifests record source selection, profiles, frame positions, RGB/Teacher input shapes, input hashes and synchronized forward timings. Missing images or temporal guides are reported explicitly. Failed exports retain completed cache files; the cleanup prompt runs after reports finish. Best/last saving, printed metric statistics and end-of-training curves are retained.

`temporal_latest.json/.csv/.png` record each profile at frame positions 1–10. Solid curves use Student history; dashed curves reset history on the same current image. Both are scored against the same matching-profile temporal Teacher RGB target. Rows include loss/PSNR/SSIM/LPIPS, signed temporal-minus-reset deltas, actual history depth/use, valid-pixel counts and flow-warped RGB MAE. Lower loss/LPIPS/warp MAE is better; higher PSNR/SSIM is better. Warp MAE measures consistency, not recovered detail or independent ground-truth quality.
These features apply to new training processes. Existing runs that only saved
`last.pt` cannot reconstruct an earlier best checkpoint.

---

## Standalone validation and prediction export

Run `val.py` beside `train.py` to validate an existing checkpoint and regenerate
comparison images without training:

```bash
uv run --locked python val.py --weights output/test_12/last.pt --project my_validation
# Quick scoring/export check, one physical clip and all four profiles:
uv run --locked python val.py --weights output/test_12/last.pt --project preview --split val --max-clips 1
```

Use `best.pt` for the selected best model or `last.pt` for the latest model. Model
structure, selected feature adapters, adapter size and loss settings come from the
checkpoint. Dataset paths, device, augmentation and comparison settings come from
`--config config/train.yaml`; `--device cuda:1` overrides the configured device.

`--project` names the report directory. With `runtime.output_dir: output/test`,
reports go to `output/val/<project>/`. Repeated names receive `_2`, `_3`, etc.,
without overwriting previous reports. `--split` accepts `val`, `test` or `both`
(default). `--max-clips N` limits physical clips in each selected split, retaining every profile for both scoring and export. Explicitly requested missing splits fail; `both` reports missing
splits and processes the available data.

`validation.json` contains freshly recomputed clean-image loss/PSNR/SSIM/LPIPS
against Teacher RGB and the selected checkpoint epoch. Historical curves/statistics
are reconstructed from `summary.json` beside the checkpoint when available; their
best epoch is independent of the chosen `last.pt`. If no history exists, the CLI
still scores and exports images without fabricating a training curve. Validation
and comparison exports both use the checkpoint's full-field adapter geometry.
Standalone validation extracts/reuses sequential Teacher targets under the same profile-grouped cache using an isolated `evaluate` split and `evaluation.json` index; it does not overwrite training targets. Teacher is unloaded before Student scoring. Per-split `temporal_val`/`temporal_test` JSON/CSV/plots accompany `validation.json`.

---

## License

Project code is licensed under [MIT](LICENSE). External dependencies, datasets and Teacher weights retain their own terms.
