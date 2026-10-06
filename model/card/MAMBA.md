# Pure PyTorch Mamba / SSM

`mamba.py` provides a Mamba-1 style selective SSM without mamba-ssm, Triton or
compiled CUDA extensions. It supports CPU and CUDA using ordinary PyTorch.
It is a research implementation, not an official package or performance equivalent.
The scan is sequential; long images can be slow and consume large training memory.

Classes: Mamba (BLC mixer), MambaBlock (BCHW residual block), MambaStage (BCHW stage).
SSMStage is an alias registered in card.py. Existing stage arguments are preserved.

YAML example (replace a stage row; keep graph indices unchanged):

```yaml
# [channels, depth, state_dim, d_conv, expand, scan]
- [-1, 1, MambaStage, [32, 2, 16, 4, 2, raster]]
# or four-direction, weight-shared axial scans:
- [-1, 1, SSMStage, [32, 2, 16, 4, 2, cross]]
```

raster flattens the entire image in row-major order. cross scans each row and
column in both directions, resetting state for each sequence. It is our custom
image wrapper, not the exact VMamba SS2D algorithm. All sequences start at zero
state; there is no temporal state sharing between frames or streaming API.

The core has input-dependent delta/B/C, negative diagonal A, causal depthwise
convolution, D skip and SiLU gate. Float16/bfloat16 recurrent arithmetic uses
float32; float64 remains available for gradient checks. Avoid blanket
reinitialization of dt_proj.bias. The _no_weight_decay flags on A_log/D are hints:
your optimizer must explicitly group these parameters to honor them.

No original DLSS teacher weights or official fused-kernel numeric parity is
claimed. Keep the original teacher inference backend for distillation.

Reference for architecture:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

The existing mambatest.py still imports the external mamba_ssm package. To test
this implementation use `from model.card.mamba import Mamba` from project root,
or run `python -m unittest discover -s tests -v`.
