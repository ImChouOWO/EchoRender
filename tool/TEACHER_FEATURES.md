# Teacher feature modes

`teacher_features.py` receives your existing teacher; it does not define one.
There are two training modes: `online` runs the teacher on each request; `cached`
only reads precomputed features and fails on missing or changed inputs.
Call `precompute()` explicitly before cached training.

```python
from tool.teacher_features import TorchFeatureExtractor, TeacherFeatures, file_sha256

# teacher is the actual nn.Module used by your existing inference pipeline.
print(TorchFeatureExtractor.list_layers(teacher))
layers = ['actual.module.name']  # replace with names printed above
extractor = TorchFeatureExtractor(teacher, layers)
provider = TeacherFeatures(
    extractor, layers, mode='online', cache_dir='data/teacher_features',
    teacher_id=file_sha256('weight/mlxw/dlssnr-logical.safetensors')
        + ':reference:preprocess-v1',
    output_device='cuda',
)

# Option A: precompute, then train without teacher inference.
# prepared_samples yields (stable_id, prepared input tensor).
provider.precompute(prepared_samples)
provider.set_mode('cached')
features = provider.get(prepared_input, sample_id='frame-00001')

# Option B: generate features during training.
provider.set_mode('online')
features = provider.get(prepared_input)
```

Move input tensors to the teacher's device yourself. Default runner calls
`teacher(inputs)`; pass `runner=lambda inputs: ...` to TorchFeatureExtractor for
keyword arguments or a custom pipeline entry point which executes the hooked
teacher. Tuple/dict module outputs require `selectors={name: lambda output: ...}`.
Hook captures are detached clones; no teacher gradients are produced. Train/eval
states are restored and hooks removed after each call. This tool is sequential;
do not use the same hooked teacher concurrently from multiple threads.

For a NumPy/native backend, provide your own callable extractor returning
`{selected_name: torch.from_numpy(array.copy())}`. The native/reference backend
must expose those intermediate arrays first; a progress callback containing only
step names cannot provide features. Passing a pipeline object instead of its
actual nn.Module to TorchFeatureExtractor will be rejected.

Cache identity includes teacher_id and ordered selected layers. Include checkpoint
hash, precision, backend/code version and preprocessing policy in teacher_id.
The input digest covers actual input tensors, including noise/history/control
channels, so random augmentations cannot silently reuse mismatched features.
Use deterministic inputs for precomputation, or online mode for random inputs.
Recompute caches whenever teacher weights or preprocessing changes.

A cache entry represents one sample or one fixed batch, exactly as provided.
A different batch ordering/size requires separate entries. For flexible shuffled
training, precompute one sample at a time and stack corresponding features during
collation. No automatic BCHW/BHWC conversion or student projection is performed;
align feature layouts, spatial sizes and channels explicitly before loss.
Large feature maps consume substantial disk space; select only needed layers.
Cached mode can be constructed with extractor=None, so the teacher need not load.

Tests: `python -m unittest discover -s tests -v` from the project root.
