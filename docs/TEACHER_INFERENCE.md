# Single-image recovered Teacher

The Teacher executes the real decoded checkpoint through MLX-DLSS's PyTorch
reference. It does not reuse the randomly initialized structural student.

## Backend and reproducibility

- Repository: https://github.com/iamwavecut/MLX-DLSS
- Pinned revision: `7870371e8bf6c03967213f1a5d40db6c5a3bee38`
- Package subdirectory: `python`; upstream license: Apache-2.0.
- The dependency is pinned in `pyproject.toml` and `uv.lock`. `uv sync --locked`
  installs it; Git/network access is needed on the first installation.
- Architecture and precision operations come from `mlxdlss.model`, with complete
  checkpoint shape/format validation from `mlxdlss.pipeline.load_weights`.
- Geometry, noise and first-frame channels come from `mlxdlss.features`.
- Four-channel composition follows `mlxdlss.temporal.compose_temporal` with
  current-image values in channels 7:10. The gate is still evaluated, but there
  is no cross-image state, sequence input, optical flow or temporal training.

This fixes the particular recovered reference as the future distillation
Teacher. Independent reverse-engineered implementations differ in graph and
rounding. Successful image inference does not establish bit-exact agreement
with NVIDIA, nor prove an improvement in image quality. Do not silently update
the backend while keeping old Teacher feature caches.

## Interfaces

`EchoRenderTeacher(weights, device="auto")` returns a frozen PyTorch module.
Its forward accepts `[B,H,W,16]` prepared floating-point features, with H/W at
least 320 and multiples of 64, and returns `[B,H,W,4]` raw heads. It validates
finite inputs/outputs and disables autocast around the recovered float32 graph.
It accepts logical v8-v18 checkpoints supported by the pinned backend; packed
byte buffers are rejected. Checkpoint buffers are not student-compatible weights.

`TeacherPipeline(teacher).enhance(rgb)` accepts HWC float RGB in [0,1]. The result
contains the original-size RGB image, detached CPU raw head on the padded field,
network extent and timings. `enhance_file(path)` reads one image directly.
Each invocation restarts the image condition and uses noise index zero unless
explicitly changed. Standard style is the only exposed profile. Intensity zero
is an identity output; the normal default is one. No image resizing is performed.

The geometry matches the pinned MLX-DLSS implementation, including its minimum
320/multiple-of-64 field; it does not claim the later Intel port's geometry fixes.

## Run

```powershell
uv run python infer.py --input input.png --weights "C:/path/to/dlssnr-logical.safetensors" --output outputs/teacher.png --report outputs/teacher.json
```

The default device is CUDA when available, otherwise CPU. For an explicit
selection use `--device cuda`, `--device cuda:0`, or `--device cpu`. CUDA uses
the reference operations, not the vendor's CUDA kernels. Existing output files
are protected unless `--overwrite` is supplied. Outputs are local artifacts
ignored by Git; weights are not copied or committed.

## Tests

The ordinary test suite checks feature conditions, deterministic noise, padding
and cropping, gate composition against the pinned reference, RGB IO, malformed
inputs, packed/missing weights and CLI output protection:

```powershell
uv run python -m unittest discover -s tests -v
```

To additionally test a real external checkpoint:

```powershell
$env:ECHORENDER_TEACHER_WEIGHTS = "C:/path/to/dlssnr-logical.safetensors"
uv run python -m unittest discover -s tests -v
```

The real-weight test runs three independent images (A, B, A), checks identical
A results, finite non-identity RGB output and frozen/eval behavior, then verifies
the detached raw head can supervise a small differentiable head with MSE.
It does not train or update the Teacher and does not save model weights.
Without that environment variable the real-checkpoint test is explicitly skipped.

During implementation, the CLI also processed a full Sintel clean image
`training/clean/ambush_2/frame_0001.png` (1024x436) on RTX 5060 Ti with the user's
logical-v18 checkpoint (649 tensors). The padded field was 1024x448, output
values were finite, and the RGB output differed from the input. This is an
execution smoke test, not a fidelity benchmark or steady-state latency result.
