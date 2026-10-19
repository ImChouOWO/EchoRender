# Online stage distillation

Training captures Teacher E1/E2/E3 and the final global block in the same forward
as the raw head. `FeatureRecordingBackend` subclasses the pinned MLX-DLSS backend
and observes its functional `_window`/`_global` results; the backend graph and
rounding are unchanged. Captures are detached BCHW, cleared even after failure.
The old generic hook/cache interface has been removed.

For the required HT checkpoint:

| Name | Teacher observation | Teacher stride | Student YAML layer | Student stride |
| --- | --- | --- | --- | --- |
| E1 | Block 8 window output, before pooling/projection | 4 | 3, SSMStage | 2 |
| E2 | Block 14 window output, before pooling/projection | 8 | 5, SSMStage | 4 |
| E3 | Block 22 window output, before pooling/projection | 16 | 7, SSMStage | 8 |
| ViT | Block 38 global output | 64 | 11, ViTStage | 32 |

The structural student and recovered Teacher do not have identical sampling
boundaries. Student features are projected (optional, identity-initialized 1x1),
resized onto the Teacher's physical spatial grid, and cropped to cells intersecting
the real image. Teacher-only end padding in the split/global path is excluded;
we do not stretch the student to the padded capture extent. This is approximate
spatial correspondence, not proof of equivalent receptive fields. Raw-head and
RGB losses exclude all reflected padding; MS-SSIM is never computed on synthetic
padding to make a small real image pass.

E1/E2/E3 use per-location channel-L2 normalization and mean squared error. ViT uses
`0.5 * feature_MSE + off_diagonal_cosine_affinity_MSE`. Affinity is computed exactly
in row chunks, without sampling tokens. Output loss compares all four raw-head
channels; RGB-head and gate MSE are also logged separately. Teacher targets never
receive gradients. Feature projections belong to the optimizer and checkpoint.

All formulas and loss composition live in `model/card/loss.py`; all coefficients,
SSIM scales and normalization epsilon live in `config/train.yaml`.
