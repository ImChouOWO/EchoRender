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

Teacher weights resolve from the project root and must stay inside its `weights/`
directory, regardless of YAML location or shell directory. Other paths resolve
relative to the YAML file. Set:

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

E1/E2/E3 use the project's custom pure-PyTorch **SSMStage**, with widths
32/64/128 and depths 2/4/6. Top-level `ssm.rotation: clockwise` and
`ssm.directions: 2` in `config/model.yaml` configure all three stages. This uses two shared axial
mixers per block: right+down, down+left, left+up, up+right, repeating every four
blocks and restarting at each stage. State resets per row/column. SSM checkpointing is controlled by `training.gradient_checkpointing`:
`true` enables non-reentrant recomputation during training; the current default
`false` retains activations and avoids recomputation. E0/E4 remain Swin, the bottleneck
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
1x1 student projections accommodate differing feature bases and channel widths.
Their input widths are read from the Student; output widths match Teacher
64/128/256/1024. Unequal widths require `loss.projection: true`; rectangular
projections initialize the shared channels to identity and the rest to zero. They are optimized
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

## Learning-rate scheduling

`tool/training_config.py:TrainingSchedule.values(epoch=..., step=...)` is called
before every optimizer update. It returns the scheduled `lr`, fixed `grad_clip`
and fixed loss coefficients from YAML. The trainer applies and logs these values.
`step` is the number of updates already completed; `epoch` starts at 1.

The optimizer remains **SGD**, with momentum and weight decay configured under
`training.optimizer` (both default to zero). The default schedule is:

```yaml
schedule:
  kind: cosine
  eta_min: 0.000001
```

Cosine annealing runs once across the planned optimizer updates, with no warmup
or restarts. The horizon is `min(epochs * len(train_loader), max_steps)` when a
step cap is present. With T >= 2, the LR used by the first update is the configured
initial LR (default 0.001); the last update uses `eta_min`. A one-update run uses
the initial LR. Values remain at the minimum after the horizon. Validation and
latency measurement do not advance the schedule.

Use `kind: constant` to keep LR fixed (`eta_min` is then unused). Loss-weight
ramps and other training strategies remain extension points; none are enabled.

`loader.batch_size: 16` 為每次 optimizer 更新的有效批次；
`training.micro_batch_size: 4` 為 Student 同時前向的張數。每 4 張計算一次
loss／backward，累積 4 組後裁剪梯度並更新一次，LR 也只前進一次。
不足 16 張的最後一批以實際張數平均；不同網路尺寸會分組，保留原圖幾何。
每張影像依 sample ID 讀取乾淨 Teacher 目標，增強只套用 Student。
改動 Student 方向或 micro-batch 不需要重建 Teacher 快取。

## Teacher feature cache

`training.releas: true` (default) enables this mode. Training first runs the frozen Teacher once on clean train and validation images,
writing E1/E2/E3/ViT features, raw head and RGB targets to project `output/feats/`.
Models initialize on CPU. During target generation only Teacher is on the selected
GPU; after caching Teacher is completely unloaded from GPU and CPU, and cached targets
are loaded one image at a time. LPIPS moves to GPU only for validation. Teacher latency is measured once during cache creation; student latency is
measured each epoch, using the same input extent. SGD, cosine LR, augmentation and loss definitions remain unchanged.

All epochs reuse the same cache, loading targets by sample ID for each shuffled
batch. Complete caches are also reused across runs when Teacher weights, image
paths/sizes/modification times, crop, preprocessing and backend code match.
Incomplete or mismatched caches are rebuilt; old caches without a manifest are
rebuilt once. Augmentation, loss weights and optimizer settings do not invalidate
clean Teacher targets. Benchmark settings and device also participate in validity.

After successful training the terminal asks `[Done]: Delete cached features? [Y/N]:`: Y clears
`output/feats/`; N retains it for the next run. EOF or Ctrl+C retains the cache.
Failed/interrupted training also retains it, with complete caches reusable.
A process lock protects the directory, including while waiting for the answer.
`training.releas: true` keeps cache distillation enabled and saves the fixed `blend_scale` in the cache manifest/checkpoint/summary and unloads Teacher;
`false` keeps the frozen Teacher and Student together on the selected GPU,
using online distillation without cache reads, writes or cleanup prompt.

## Student-only augmentation

Training uses a per-image augmentation probability: 10% in epoch 1, rising
linearly to 25% at 50% of the configured epochs, then held until augmentation
is disabled from 75% onward (rounded up to the next integer epoch). For the
default 10 epochs: epoch 1 = 10%, epoch 5 = 25%, epochs 5–7 = 25%, epochs 8–10 = 0%.
Short runs may reach the cutoff before a full ramp is possible. `max_steps`
does not rescale this epoch schedule. The 25% cap is a sampling probability;
actual sample counts fluctuate, and `augmented_samples` records the count.

Every selected image receives Gaussian noise followed by exactly one uniformly
selected adjustment: brightness, contrast, or sharpness. In `augmentation`,
`noise_std: 0.02` sets the maximum sigma (uniformly sampled from 0 to 0.02 on
RGB in [0,1]); brightness/contrast factors range from 0.8 to 1.2, and sharpness
from 0.5 to 1.5. These are configurable strengths, separate from probability.

Teacher inputs, raw-head/feature/RGB targets remain clean. The same center crop
is used by both branches; student features and RGB composition use the corrupted
image. Validation, benchmarks, and inference remain clean. Loss weights, SGD,
and cosine LR settings are unchanged. Augmentation has its own RNG seeded by
`loader.seed`; training logs include `augmentation_probability` and
`augmented_samples`. Set `augmentation.enabled: false` to disable it.

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

Training reserves a fresh `runtime.output_dir` automatically: `output/test`,
then `output/test_2`, `output/test_3`, etc. Existing directories are preserved,
including empty ones. The selected absolute path is printed and recorded in
`resolved_train.yaml` and checkpoints.

Teacher cache creation overlaps CPU image loading/preparation, single-Teacher GPU
inference, and background atomic writes. Worker counts are automatically estimated
from the first real sample's preparation/inference/write times and capped by CPU
affinity/cgroup limits and available RAM. Reader/writer queues have bounded capacity.
The selected counts are printed and recorded in the cache manifest; this setting
is separate from `loader.num_workers`, which still controls the training loader.

Cache extraction checkpoints its index after each atomic target write. If a run is
interrupted after train extraction, the next compatible run reuses completed train
targets and extracts only missing validation targets. Missing/truncated target files
are rebuilt individually. The progress log reports reused and newly extracted counts.
Pipeline/display changes no longer invalidate targets; Teacher capture code, weights,
images and preprocessing still participate in the fingerprint. Unindexed old files
cannot be safely resumed and require one complete extraction.

CUDA Teacher cache extraction automatically uses a single-shape CUDA Graph for
batch-1 inference. Fixed attention biases are reordered once before capture;
the graph preserves the pinned reference arithmetic and checks head plus all
captured features for exact equality on its first input. GPU warmup/capture adds
one-time startup work. CPU execution uses the reference path, and unsupported
captures report the reason and fall back. Moving Teacher to CPU resets the graph
and releases its device buffers; normal Teacher inference outside cache creation
continues to use the reference path. Latency benchmarks of head-only inference
therefore describe that reference path, not graph-accelerated cache throughput.
