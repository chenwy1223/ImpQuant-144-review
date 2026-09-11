import torch
from torch import nn
from functools import partial
from qmllm.quantization.quant_funcs import *

class WALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant='per_token',
        a_bit=8,
        w_bit=8,
        quantize_output=False,
        dev='cuda',
        input_scale=None,
        outlier_mask=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_bit = a_bit
        self.w_bit = w_bit

        self.register_buffer('weight', torch.zeros(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_buffer('bias', None)

        # scale-transfer reparameterization: x <- x / input_scale
        self.register_buffer("input_scale", input_scale)
        self.register_buffer("outlier_mask", outlier_mask)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=self.a_bit)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=self.a_bit)
        elif act_quant == 'per_token_importance':
            self.act_quant_name = 'per_token_importance'
            self.act_quant = partial(quantize_activation_per_token_importance, n_bits=self.a_bit)
        elif act_quant == 'per_token_outlier':
            self.act_quant_name = 'per_token_outlier'
            self.act_quant = None
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            if self.act_quant_name == "per_token_outlier":
                self.output_quant = lambda x: quantize_activation_per_token_outlier(x, self.outlier_mask, n_bits=self.a_bit)
            else:
                self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WALinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.input_scale is not None:
            x = x / self.input_scale.to(dtype=x.dtype)

        if self.a_bit is None or self.a_bit >= 16:
            q_x = x
        elif self.act_quant_name == "per_token_outlier":
            q_x = quantize_activation_per_token_outlier(x, self.outlier_mask, n_bits=self.a_bit)
        else:
            q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=4, a_bit=8, weight_group=128, quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = WALinear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, a_bit=a_bit, w_bit=w_bit, quantize_output=quantize_output, dev=module.weight.device)
        
        # Quantize the weight matrices
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=w_bit)
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(module.weight, n_bits=w_bit)
        elif weight_quant == 'per_group':
            new_module.weight = pseudo_quantize_tensor(module.weight, n_bits=w_bit, q_group_size=weight_group, inplace=True)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        del module
        return new_module

    def __repr__(self):
        return 'W{}A{}Linear'.format(self.w_bit, self.a_bit)

