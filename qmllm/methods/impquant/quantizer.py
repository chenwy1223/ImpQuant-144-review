import torch
from qmllm.methods.mbq.quantize.pre_quant import get_blocks, get_named_linears
from qmllm.methods.impquant.activation_quant import ImpQuantLinear


@torch.no_grad()
def apply_impquant(model, w_bit, a_bit, q_config, impquant_results):
    state = impquant_results
    if (state.get('method') != 'impquant' or state.get('format_version') != 2
            or state.get('scale_normalization') != 'sqrt_max_times_min'):
        raise ValueError('Run ImpQuant calibration to create a compatible cache')
    for key, value in [('w_bit', w_bit), ('a_bit', a_bit), ('q_group_size', q_config['q_group_size'])]:
        if state.get(key) != value:
            raise ValueError(f'ImpQuant cache {key} does not match the requested value')
    layers = get_blocks(model)
    if any(len(state[key]) != len(layers) for key in ('weights', 'act_outlier_masks', 'input_scales')):
        raise ValueError('ImpQuant cache has a different number of decoder layers')
    for i, layer in enumerate(layers):
        for name, module in get_named_linears(layer).items():
            weight = state['weights'][i][name]
            mask = state['act_outlier_masks'][i][name]
            scale = state['input_scales'][i][name]
            if (weight.shape != module.weight.shape or mask.numel() != module.in_features
                    or scale.shape != (module.in_features,)):
                raise ValueError(f'ImpQuant cache shape mismatch at layer {i}: {name}')
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f'Invalid ImpQuant input scale at layer {i}: {name}')
            quantized = ImpQuantLinear(module.in_features, module.out_features, module.bias is not None,
                a_bit=a_bit, w_bit=w_bit,
                dev=module.weight.device, outlier_mask=mask.to(module.weight.device),
                input_scale=scale.to(device=module.weight.device, dtype=torch.float32))
            quantized.weight = weight.to(module.weight.device, module.weight.dtype)
            if module.bias is not None:
                quantized.bias = module.bias.detach().clone()
            parent_name, _, leaf = name.rpartition('.')
            parent = layer.get_submodule(parent_name) if parent_name else layer
            setattr(parent, leaf, quantized)
