# EchoRender

[繁體中文](CHT-README.md)

![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange) ![Task](https://img.shields.io/badge/Task-RGB_SR-blue) ![Precision](https://img.shields.io/badge/Precision-BF16-green)

RGB super-resolution with historical RGB alignment, an encoder/decoder backbone, routed selective state-space scans and three supervised reconstruction stages. Teacher models, distillation and feature caches are no longer part of the application.

```text
Current LR RGB + aligned previous LR RGB
  -> 6-channel bottleneck/Swin adapter -> encoder / SSM / ViT / decoder
  -> Masked residual cross-attention SR -> NAFNet deblurring -> Restormer denoising -> HR RGB
```

<details><summary><em>Current limitations</em></summary>

The masked cross-attention SR head requires new training. Earlier PixelShuffle SR and distillation checkpoints are unsupported. Gaussian degradations are synthetic and do not establish real CCTV performance. Native-size evaluation uses overlapping LR tiles; tile/context settings must match when comparing scores. The configured model sizes are adaptations, not reproductions of upstream benchmark results.

</details>

---

## Overview

The backbone retains its encoder/decoder skip connections and the InputAdapter-to-output feature bypass. Input geometry is preserved; training uses aligned random patches covering the full image rather than globally resizing a frame to a fixed square.

| Module | Behavior |
|---|---|
| Input | Current RGB plus flow-aligned previous input RGB; no profile/tone/control channels |
| Motion | Default frozen pretrained `raft_small` RGB alignment; `raw`: external Farneback; `feat`: RGB-pair CNN trained through reconstruction |
| SSM | Official `mamba-ssm` CUDA scan; CPU official reference for tests; FP32 recurrence |
| Routing | Semantic grouping inside overlapping windows, stable directional ordering, inverse permutation, weighted overlap reconstruction |
| Parameter generation | Shared projection split into `[dt_rank, state_dim, state_dim]`; delta projection, B and C |
| Prompt | Routed learned dictionary plus 1x1 projection of ranked features and the three parameter chunks |
| Learned prior | Zero-initialized learnable feature map; no claim of externally supplied semantic knowledge |
| Reconstruction | Two 2× masked residual cross-attention stages for 4× SR, followed by multiscale NAFNet and Restormer |

Each 2× SR stage preserves feature anchors and queries only the three inserted positions per anchor. It uses four heads, independent Q/K/V projections and 8×8 windows with shifts 0 and 4, following the [SwinIR windowing idea](https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py). A shared MLP adds RGB residuals only at inserted positions of the bicubic base. The existing 4× supervision is applied after the second stage.

MambaIRv2-Small provides state dimension 16, expansion 2, 128 routing groups and prompt rank 64. Overlapping directional windows and the parameter-conditioned prompt are EchoRender-specific changes.

---

## Project structure

```text
config/model.yaml           Backbone, motion and reconstruction sizes
config/train.yaml           Data, training, synthesis and evaluation
model/card/                  Backbone, scan, reconstruction and task losses
model/student.py             EchoRenderSR
model/motion.py              RGB-only motion alignment
train.py / val.py / infer.py Training, evaluation and image/frame inference
preview.py                   Browser comparison and training dashboard
output/                      All generated artifacts
```

---

## Installation

### 1. Environment

On the existing Linux CUDA server:

```bash
cd /home/soic/Desktop/EchoRender
uv sync --locked
source .venv/bin/activate
```

The pinned official Mamba CUDA extension must match PyTorch/CUDA. Linear, convolution and attention operations use BF16 when configured; scan recurrence and RGB losses use FP32. No legacy Teacher packages are required.

### 2. Verification

```bash
python -m unittest tool.test_sr -v
```

---

## Dataset format

```text
REDS/
  train/train_sharp/000/00000000.png
  train/train_sharp_bicubic/X4/000/00000000.png
  val/val_sharp/000/00000000.png
  val/val_sharp_bicubic/X4/000/00000000.png
```

LR/HR filenames and frame order must match, with exact spatial scale. The loader also accepts official extraction without the intermediate `X4` folder. The example uses ten training scenes and two validation scenes; set `scenes: null` to select all scenes in a split. A clip contains 1–10 consecutive frames. Only current/previous RGB is used for motion; REDS does not supply flow supervision here.

---

## Degradation supervision

Starting with clean HR `G`, synthesize Gaussian blur `B(G)` and LR Gaussian noise `N`. Map the same noise to HR with bilinear interpolation `U(N)`:

```text
SR intermediate target      B(G) + U(N)
Deblur intermediate target  G + U(N)
Final target                G
Degraded LR input           downsample(B(G)) + N
```

Blur sigma is measured in LR pixels and scaled for HR synthesis. Noise sigma is measured on LR RGB [0,1]. Intermediate targets are not clipped, preserving the algebra of shared noise. Independent random HR noise is not used because downsampling would erase information that the SR stage cannot reconstruct. Observable degraded LR is clipped to [0,1]. For unaugmented clips, use the supplied REDS LR and clean HR as all three targets. Brightness, contrast and sharpness augmentation have been removed.

The epoch schedule starts at 10%, reaches 25% at 50% progress and disables synthesis at 75%. The ratio controls strength and caps the number of augmented clips: `floor(batch_size * ratio)`. Small batches may contain no augmented clip; unaugmented examples are always retained. Validation/export strengths are separate fixed 25/50/75/100% of configured maximum sigmas. Preview optionally applies degradation directly to LR for deployment-style inspection; it labels this separately from HR-based training synthesis.

---

## Configuration

| Setting | Meaning |
|---|---|
| `runtime.output_dir` | Must resolve inside the project `output/`; repeated names get `_2`, `_3`, ... |
| `dataset.lr_patch_size` | LR training patch, default 80; x4 gives a 320-pixel HR patch |
| `loader.batch_size` | Effective clips per optimizer update |
| `training.micro_batch_size` | Simultaneous clips; gradients accumulate across clip frames/microbatches |
| `model.yaml: motion` | `raft_small` (preserved default), `raw` or `feat`; REDS provides no flow GT |
| `model.yaml: ssm` | Direction rotation/count, window, overlap stride and routing sizes |
| `model.yaml: reconstruction` | Scale, head width and per-level NAFNet/Restormer block counts |
| `validation.tile_size/tile_overlap` | LR inference tile/context; output keeps native HR geometry |
| `dataset.splits.test` | Optional paired test split; null because official REDS test GT is not public |

The model YAML keeps the existing DAG and all global skips. Changing reconstruction scale also requires matching LR data and `dataset.scale`. No fixed-size global InputAdapter resize is applied.

---

## Optimization and losses

SGD defaults remain LR=0.001, momentum=0, weight_decay=0. Cosine is applied immediately before each optimizer update. The first update uses initial LR and the final scheduled update uses `eta_min=0.000001`; a single-update run uses initial LR. There is no warmup or restart.

```text
L = w_sr * L1(Y1, B(G)+U(N))
  + w_deblur * [Charbonnier(Y2, G+U(N)) + 0.05 * Edge(Y2, G+U(N))]
  + w_denoise * L1(Y3, G)
```

All three task weights initially equal 1 and are configurable. Edge compares Laplacian-pyramid residuals following MPRNet; it is a loss, not an invented evaluation score. Frame loss is normalized per clip so long clips do not dominate short clips. Historical RGB is the previous degraded LR input; no recurrent output graph is retained across frames.

---

## Training and resume

```bash
python train.py --config config/train.yaml
python train.py --config config/train.yaml --resume output/sr/last.pt
```

`last.pt` and `best.pt` are the only persistent checkpoint names. A complete epoch-boundary resume restores optimizer, cosine step, data-loader RNG, Python/NumPy/PyTorch RNG, metric history and early-stop state. It requires the same training/data/model configuration and original `best.pt`. Stop/restart from the last completed epoch; partially completed epochs are replayed. The terminal retains fixed-width YOLO-style progress and GPU telemetry; best checkpoints receive a highlighted improvement line.

---

## Validation and inference

PSNR (higher), SSIM (higher) and pretrained LPIPS (lower) compare final RGB against clean REDS HR. Default protocol is RGB, crop border 0, image-wise averaging, overlapping tiled inference. Y-channel conversion and border crop are explicit configuration options, not silently mixed with the REDS protocol. PSNR/SSIM use the chosen RGB/Y protocol; LPIPS always uses RGB. Error heatmaps are visualization diagnostics, not additional quality metrics.

```bash
python val.py --weights output/sr/best.pt --project sr_validation --split val
python infer.py --weights output/sr/best.pt --input image.png --device cuda:1
python infer.py --weights output/sr/best.pt --input frames/ --max-frames 10
python preview.py --ip 127.0.0.1 --port 8092
```

The Preview tab includes checkpoint/image selection, uploads, device/frame/tile controls, separate blur/noise settings, intermediate-stage selection and an interactive input/prediction or GT/prediction divider. The Training tab reads current-format histories and includes the standard TensorBoard interface. Empty, loading, error and completed states are explicit. Old checkpoints remain on disk but are excluded from the current run picker.

---

## Outputs

| Location | Content |
|---|---|
| `output/<run>/best.pt`, `last.pt` | Model and complete training state |
| `output/<run>/history.json` | Completed epoch history |
| `output/<run>/tensorboard/` | Standard TensorBoard event files |
| `output/val/<run>/` | Metric summary, CSV, four metric curves and best-weight examples |
| `output/val/<run>/val/{preview,pred,heatmaps}/<mode>/<image>/` | Separate full-resolution file for each strength |
| `output/preview/` | Browser jobs and uploaded images |
| `output/infer/` | Standalone predictions |

Reports print min/max/mean/best for loss, PSNR, SSIM and LPIPS. Summary JSON also records values at the selected best checkpoint. Best-weight exports include original, noise, blur and combined degradation; each condition gets its own image instead of a compressed multi-strength grid.

---

## References

- [REDS](https://seungjunnah.github.io/Datasets/reds.html)
- [MambaIR](https://github.com/csguoh/MambaIR), [Mamba](https://github.com/state-spaces/mamba)
- [NAFNet](https://github.com/megvii-research/NAFNet), [Restormer](https://github.com/swz30/Restormer)
- [MPRNet losses](https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py)
- [PyTorch TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard.html)

## License

See [LICENSE](LICENSE) and upstream notices in [THIRD_PARTY.md](THIRD_PARTY.md).
