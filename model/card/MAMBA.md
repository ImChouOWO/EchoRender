# EchoRender routed selective scan

## Technical tags

`SSM` `MambaIRv2` `CUDA` `RGB SR`

## Overview

The SSM uses official `mamba-ssm` selective scan with FP32 recurrence. CUDA runs the fused kernel; CPU tests use the official reference. This is a custom window/direction/routing adaptation, not the complete MambaIRv2 architecture.

## Environment

Use `uv sync --locked` on the existing Linux CUDA environment and run `python -m unittest tool.test_sr -v`.

## References

- https://github.com/csguoh/MambaIR/blob/main/basicsr/archs/mambairv2_arch.py
- https://github.com/state-spaces/mamba

## License

Upstream notices are retained in ../../THIRD_PARTY.md.
