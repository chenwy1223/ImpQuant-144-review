"""ImpQuant activation quantization with separate scales for two channel sets."""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def quantize_activation_per_token_absmax(x, n_bits=8):
    if n_bits is None or n_bits >= 16:
        return x
    if not 2 <= n_bits <= 15:
        raise ValueError('Activation bit width must be between 2 and 16')
    qmax = 2 ** (n_bits - 1) - 1
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / qmax
    return ((x.float() / scale).round().clamp(-2 ** (n_bits - 1), qmax) * scale).to(x.dtype)


@torch.no_grad()
def quantize_activation_per_token_outlier(x, outlier_mask, n_bits=8):
    if n_bits is None or n_bits >= 16:
        return x
    if outlier_mask is None or outlier_mask.numel() != x.shape[-1]:
        raise ValueError('A calibrated outlier mask is required for every input channel')
    mask = outlier_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
    if mask.all() or not mask.any():
        return quantize_activation_per_token_absmax(x, n_bits)
    result = torch.empty_like(x)
    result[..., mask] = quantize_activation_per_token_absmax(x[..., mask], n_bits)
    result[..., ~mask] = quantize_activation_per_token_absmax(x[..., ~mask], n_bits)
    return result


class ImpQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, a_bit=8, w_bit=4,
                 dev='cpu', outlier_mask=None, input_scale=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.register_buffer('weight', torch.empty(out_features, in_features, device=dev))
        self.register_buffer('bias', torch.empty(out_features, device=dev) if bias else None)
        self.register_buffer('outlier_mask', outlier_mask)
        self.register_buffer('input_scale', input_scale)

    @torch.no_grad()
    def forward(self, x):
        if self.input_scale is not None:
            x = (x.float() / self.input_scale.float()).to(x.dtype)
        return F.linear(quantize_activation_per_token_outlier(x, self.outlier_mask, self.a_bit),
                        self.weight, self.bias)
