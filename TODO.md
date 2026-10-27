# EchoRender TODO

## Technical tags

`PyTorch` `Knowledge Distillation` `Temporal Rendering`

## Introduction

- [x] Change Student to the same 16-channel input layout as Teacher. Set unused control channels to zero. Assemble the tensor from RGB, aligned Student history and supported controls in preprocessing; remove the separate control-vector and history-projection interfaces. Update training, validation, prediction exports, model configuration and checkpoint validation together. Preserve Teacher outputs, distillation stages, four profiles, augmentation scheduling and feature-cache behavior. Student channels 0-2 and 15 are zero, channel 3 is one, RGB occupies 4-6, aligned history occupies 7-9 and active controls occupy 10-14. Verified both motion modes, training, validation, cache reuse, checkpoints and prediction exports.

- [x] Add selectable raw/feat RGB motion. The independent feature encoder/decoder predicts backward displacement and confidence without backbone/decoder skip connections.

## Environment setup

Use the existing EchoRender environment and GPU 1 for verification.

## License

See [LICENSE](LICENSE).
