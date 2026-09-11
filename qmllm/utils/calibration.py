"""Tensor movement and batching for prepared multimodal calibration inputs."""
import torch


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def slice_batch(value, start, end, total, key=None):
    if isinstance(value, torch.Tensor):
        if key == 'position_ids' and value.ndim == 3 and value.shape[0] == 3:
            return value[:, start:end] if value.shape[1] == total else value
        if value.ndim and value.shape[0] == total:
            return value[start:end]
        return value
    if isinstance(value, dict):
        return {k: slice_batch(v, start, end, total, k) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(slice_batch(v, start, end, total, key) for v in value)
    if isinstance(value, list):
        return [slice_batch(v, start, end, total, key) for v in value]
    return value


def prepared_inputs(prompt_inputs, prompt_kwargs):
    inputs = {**(prompt_inputs or {}), **(prompt_kwargs or {})}
    for name in ('vision_mask', 'caption_mask'):
        inputs.pop(name, None)
    inputs['use_cache'] = False
    return inputs
