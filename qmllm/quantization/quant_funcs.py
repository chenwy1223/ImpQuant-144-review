import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    if inplace:
        (
            (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        tensor = (
            torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
        ) * scales

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    # return the quantized tonsor, the scaling factor and the zero point value
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, zero_point=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, zero_point=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)

@torch.no_grad()
def quantize_activation_per_token_importance(t, n_bits=8, eps=1e-5):
    t_shape = t.shape
    d = t_shape[-1]
    t_flat = t.view(-1, d)
    token_norm = torch.linalg.vector_norm(t_flat, ord=2, dim=1, keepdim=True)
    # token_norm = token_norm * (1.0 / math.sqrt(d))

    step = (2.0 * token_norm) / (2 ** n_bits - 1.0).clamp(min=eps)

    max_int = 2 ** (n_bits - 1) - 1
    min_int = -(2 ** (n_bits - 1))
    q = torch.clamp(torch.round(t_flat / step), min_int, max_int) * step
    return q.view(t_shape).to(dtype=t.dtype)


@torch.no_grad()
def quantize_activation_per_token_outlier(t, outlier_mask: torch.Tensor, n_bits=8):
    """
    Channel-importance-aware per-token activation quantization (method.tex):
    split channels into outlier/non-outlier sets and quantize each subset with
    its own per-token absmax scale.
    """
    t_shape = t.shape
    d = t_shape[-1]
    t_flat = t.view(-1, d)

    if outlier_mask is None:
        return quantize_activation_per_token_importance(t_flat).view(t_shape).to(dtype=t.dtype)

    if not isinstance(outlier_mask, torch.Tensor):
        raise TypeError("outlier_mask must be a torch.Tensor or None")

    outlier_mask = outlier_mask.to(device=t_flat.device)
    if outlier_mask.dtype != torch.bool:
        outlier_mask = outlier_mask.bool()
    if outlier_mask.numel() != d:
        return quantize_activation_per_token_importance(t_flat).view(t_shape).to(dtype=t.dtype)

    if outlier_mask.all() or (~outlier_mask).all():
        return quantize_activation_per_token_importance(t_flat).view(t_shape).to(dtype=t.dtype)

    hi = t_flat[:, outlier_mask]
    lo = t_flat[:, ~outlier_mask]
    q_hi = quantize_activation_per_token_importance(hi)
    q_lo = quantize_activation_per_token_importance(lo)

    out = t_flat.clone()
    out[:, outlier_mask] = q_hi
    out[:, ~outlier_mask] = q_lo
    return out.view(t_shape).to(dtype=t.dtype)
