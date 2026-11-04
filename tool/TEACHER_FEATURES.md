# EchoRender feature distillation

## Technical tags

`PyTorch` `Teacher` `Feature Distillation`

## Introduction

The frozen HT Teacher retains its recovered execution graph and rounding. Captures observe the existing window, split-window and global operations. Only the stages selected by `training.distillation_layers` are retained; default: `[RGB composition, E0, E1, E2, E3, E4, VIT]`. All Student architecture stages remain present.

| Name | Teacher capture | Channels | Teacher stride | Student YAML layer | Student stride |
| --- | --- | ---: | ---: | --- | ---: |
| E0 | Block 4, before pooling/projection | 32 | 2 | 1, SwinStage | 1 |
| E1 | Block 8, before pooling/projection | 64 | 4 | 3, SSMStage | 2 |
| E2 | Block 14, before pooling/projection | 128 | 8 | 5, SSMStage | 4 |
| E3 | Block 22, before pooling/projection | 256 | 16 | 7, SSMStage | 8 |
| E4 | Block 30 split-window output, before downsample | 512 | 32 | 9, SwinStage | 16 |
| VIT | Block 38 global output | 1024 | 64 | 11, ViTStage | 32 |
| D1 | Block 47 split-window output | 512 | 32 | 14, SwinStage | 16 |
| D2 | Block 55 window output | 256 | 16 | 17, SwinStage | 8 |
| D3 | Block 61 window output | 128 | 8 | 20, SwinStage | 4 |

Model YAML `stages` maps names to DAG layers. Unknown/duplicate selections fail validation. `RGB composition` is a loss selection, not an extra captured backbone tensor. Selected feature names control captures, identity-initialized 1x1 feature projections, loss averaging, cache signatures and resume checks.

## Environment setup

Use the locked EchoRender environment. CUDA Student training uses fused mamba-ssm 2.2.6.post3 selective scan and optional BF16; the recurrence and RGB loss stay FP32. RAFT-Small is pretrained and fine-tuned with Sintel flow/validity. The output head receives an additional direct InputAdapter skip. `training.adapter_size: 320` preserves the full image and resizes the assembled Student input in the root InputAdapter. Teacher preparation uses the same target size. Motion guides and temporal histories retain full-image coordinates; composed RGB maps back to native size.

Projected Student features are resized to the Teacher's physical grid, then cropped to cells intersecting the target image. Teacher-only split/global end padding is excluded; Student features are never stretched to fill that padding. This gives approximate spatial correspondence, not equal receptive fields. Captures are detached BCHW and cleared after errors.

Selected E/D features use channel-L2-normalized MSE. VIT uses `0.5 * feature_MSE + off_diagonal_cosine_affinity_MSE`, with exact row chunks. The intermediate term averages weighted stage losses over the selected stage count. Raw RGB residual and gate-logit MSE use separate direct weights 0.75 and 0.05; `RGB composition` in the selected list enables native composed RGB supervision. Projections belong to the optimizer and checkpoint.

RGB loss coefficients are MSE 1.0, SSIM loss 0.5, MS-SSIM loss 0.8 and LPIPS 0.05, multiplied by reconstruction 2.0. LPIPS uses a frozen/eval pretrained AlexNet and [-1,1] RGB; its input gradient trains Student, with detached Teacher targets. The network state is saved with the criterion and excluded from the optimizer. It uses standard image-level reduction; the following spatial importance applies to MSE/SSIM/MS-SSIM. Importance is detached: `w = 1 + alpha * clamp(mean(abs(Teacher_RGB - clean_RGB), channels) / tau, 0, 1)`, with current alpha 2.0 and tau 0.1. Each image normalizes by its weight sum. Structural losses weight local score maps at matching scales; quality metrics remain unweighted.

Current formats are `echorender-bhwc8-raft-small-encoder-bf16-v8` for checkpoints and `sequence-targets-v5` for caches. Earlier formats are rejected; retain old results separately and start fresh training. No crop or legacy resize aliases remain.

## Experimental data

GPU 1 at 320x320: real Teacher captures were E2 `[1,128,40,40]`, VIT `[1,1024,8,8]`, D1 `[1,512,12,12]`, D3 `[1,128,40,40]`. D1's valid physical grid is 10x10; VIT's is 5x5. Capture and CUDA Graph heads matched the ordinary frozen forward exactly.

## References

- [Training settings](../config/train.yaml)
- [Model stage mapping](../config/model.yaml)
- [Loss definitions](../model/card/loss.py)

## License

See the project LICENSE.
