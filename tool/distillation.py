"""Raw-head distillation and validation for train.py."""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from .teacher_pipeline import prepare_image


def module_digest(module, *, parameters_only=False):
    """Hash parameters and buffers without retaining a second GPU model."""
    digest = hashlib.sha256()
    values = module.named_parameters() if parameters_only else module.state_dict().items()
    for name, tensor in values:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_models(teacher, student):
    if student.input_channels != 16 or student.channels[-1] != 4:
        raise ValueError('Student must have 16 input and 4 output channels')
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError('Student has no trainable parameters')
    if any(parameter.device != teacher.device for parameter in parameters):
        raise ValueError('Teacher and student must use the same device')
    return parameters


def image_loss(teacher, student, image, *, crop_size=0):
    """Shared geometry/conditions, raw four-channel MSE on real image pixels."""
    if type(crop_size) is not int or crop_size < 0:
        raise ValueError('crop_size must be a nonnegative integer')
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected a CPU CHW RGB image')
    rgb = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    original_hw = list(rgb.shape[:2])
    if crop_size:
        height, width = rgb.shape[:2]
        top, left = max(0, (height - crop_size) // 2), max(0, (width - crop_size) // 2)
        rgb = rgb[top:top + crop_size, left:left + crop_size]
    prepared = prepare_image(rgb)
    features = prepared.features.to(teacher.device)
    with torch.no_grad():
        target = teacher(features)
    if target.requires_grad:
        raise RuntimeError('Teacher target unexpectedly requires gradients')
    student_input = features if student.layout == 'BHWC' else features.permute(0, 3, 1, 2).contiguous()
    predicted = student(student_input)
    if student.layout == 'BCHW':
        predicted = predicted.permute(0, 2, 3, 1)
    if predicted.shape != target.shape:
        raise RuntimeError(f'Head shape mismatch: student {predicted.shape}, teacher {target.shape}')
    height, width = rgb.shape[:2]
    loss = F.mse_loss(predicted[:, :height, :width].float(), target[:, :height, :width].float())
    if not torch.isfinite(loss).item():
        raise RuntimeError('Distillation loss is not finite')
    return loss, {'original_hw': original_hw, 'image_hw': [height, width],
                  'prepared_bhwc': list(features.shape), 'head_bhwc': list(target.shape)}


def train_batch(teacher, student, batch, optimizer, *, crop_size=0, grad_clip=None):
    """One optimizer update; ragged images accumulate equally weighted gradients."""
    parameters = _validate_models(teacher, student)
    if not batch['images']:
        raise ValueError('Training batch must contain at least one image')
    if grad_clip is not None and (not np.isfinite(grad_clip) or grad_clip <= 0):
        raise ValueError('grad_clip must be finite and positive')
    if {id(parameter) for group in optimizer.param_groups for parameter in group['params']} != {
            id(parameter) for parameter in parameters}:
        raise ValueError('Optimizer must own exactly the trainable student parameters')
    student.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    losses, shapes = [], []
    for image in batch['images']:
        loss, shape = image_loss(teacher, student, image, crop_size=crop_size)
        (loss / len(batch['images'])).backward()
        losses.append(loss.detach().item())
        shapes.append(shape)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if len(gradients) != len(parameters) or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError('Student gradients are missing or non-finite')
    if grad_clip is None:
        norm = torch.stack([gradient.detach().float().norm() for gradient in gradients]).norm().item()
    else:
        norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True).item()
    if not np.isfinite(norm):
        raise RuntimeError('Student gradient norm must be finite')
    optimizer.step()
    if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
        raise RuntimeError('Student parameters became non-finite after the update')
    return {'loss': sum(losses) / len(losses), 'per_image_loss': losses,
            'gradient_norm': norm, 'student_gradient_tensors': len(gradients),
            'batch_size': len(batch['images']), 'sample_ids': list(batch['sample_ids']), 'shapes': shapes}


@torch.no_grad()
def evaluate(teacher, student, loader, *, crop_size=0):
    """Evaluate without optimizer updates, keeping the model's previous mode."""
    _validate_models(teacher, student)
    was_training = student.training
    student.eval()
    teacher.eval()
    total, images, batches = 0.0, 0, 0
    try:
        for batch in loader:
            for image in batch['images']:
                loss, _ = image_loss(teacher, student, image, crop_size=crop_size)
                total += loss.item()
                images += 1
            batches += 1
    finally:
        student.train(was_training)
    if not images:
        raise ValueError('Validation loader contains no images')
    return {'loss': total / images, 'images': images, 'batches': batches}
