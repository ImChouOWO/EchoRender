"""GPU utilization telemetry and isolated forward latency measurements."""
from time import monotonic, perf_counter
import statistics

import torch


class GPUMonitor:
    def __init__(self, device, interval=1.0):
        self.device, self.interval = torch.device(device), interval
        self._last, self._value = float('-inf'), None
        self._nvml, self._handle = None, None
        if self.device.type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = pynvml
                # UUID respects CUDA_VISIBLE_DEVICES remapping, unlike NVML index 0.
                properties = torch.cuda.get_device_properties(self.device)
                uuid = properties.uuid
                self._handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.decode() if isinstance(uuid, bytes) else str(uuid))
            except (ImportError, RuntimeError, AttributeError) as exc:
                self.error = str(exc)
            except Exception as exc:
                # NVML can be unavailable despite CUDA working (e.g. restricted hosts).
                self.error = str(exc)

    def sample(self):
        now = monotonic()
        if now - self._last >= self.interval:
            self._last = now
            if self._handle is not None:
                try:
                    self._value = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                except self._nvml.NVMLError:
                    self._value = None
        return self._value

    def display(self):
        value = self.sample()
        return f'{value:.0f}%' if value is not None else 'N/A'

    def close(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except self._nvml.NVMLError:
                pass
            finally:
                self._nvml = None


@torch.no_grad()
def measure_forward(model, inputs, *, warmup, repeats):
    device = inputs.device
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(warmup):
            model(inputs)
        sync()
        durations = []
        for _ in range(repeats):
            sync()
            started = perf_counter()
            model(inputs)
            sync()
            durations.append((perf_counter() - started) * 1000)
        return {'latency_ms': statistics.mean(durations),
                'latency_median_ms': statistics.median(durations)}
    finally:
        model.train(was_training)


@torch.no_grad()
def benchmark_forward(teacher, student, features, *, warmup, repeats):
    """Batch-1 float32 forward only, features disabled, eval mode, CUDA synced."""
    result = {'device': str(features.device), 'precision': 'float32', 'batch_size': 1,
              'network_bhwc': list(features.shape), 'warmup': warmup, 'repeats': repeats,
              'scope': 'forward_only_without_feature_capture'}
    for name, model, inputs in (
        ('teacher', teacher, features),
        ('student', student, features if student.layout == 'BHWC' else features.permute(0, 3, 1, 2).contiguous()),
    ):
        result.update({f'{name}_{key}': value for key, value in measure_forward(
            model, inputs, warmup=warmup, repeats=repeats).items()})
    return result
