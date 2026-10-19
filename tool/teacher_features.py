"""Capture stages from the pinned functional Teacher, without copying its graph.

E1/E2/E3: last window output before fused pooling/projection; ViT: last global
block. The existing backend performs every operation. Captures are detached BCHW.
"""
from mlxdlss.model import NeuralRenderingModel


class FeatureRecordingBackend(NeuralRenderingModel):
    def __init__(self, weights):
        super().__init__(weights)
        self._captures = None
        self._window_names = {
            self.topology.encoder_stage(stage)[-1]: f'e{stage}'
            for stage in (1, 2, 3)
        }

    def _capture(self, name, value):
        if self._captures is not None:
            self._captures[name] = value.detach().permute(0, 3, 1, 2).contiguous()

    def _window(self, value, index, *, head_count, publish=True):
        result = super()._window(value, index, head_count=head_count, publish=publish)
        if index in self._window_names:
            self._capture(self._window_names[index], result)
        return result

    def _global(self, value, index):
        result = super()._global(value, index)
        if index == self.topology.global_stage[-1]:
            self._capture('vit', result)
        return result

    def forward_with_features(self, value):
        if self._captures is not None:
            raise RuntimeError('Teacher feature extraction cannot be nested')
        self._captures = {}
        try:
            head = self(value)
            if set(self._captures) != {'e1', 'e2', 'e3', 'vit'}:
                raise RuntimeError('Teacher did not execute all selected stages')
            return head, self._captures
        finally:
            self._captures = None
