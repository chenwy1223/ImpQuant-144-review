import os
import torch
from qmllm.methods.impquant.pre_quant import run_impquant_weight_importance
from qmllm.methods.impquant.quantizer import apply_impquant


def impquant_entry(model, prompt_inputs, prompt_kwargs, run_impquant_process,
                   pseudo_quant, scale_path=None, q_group_size=128, w_bit=4, a_bit=8,
                   text_scale=1.0, vision_scale=1.0, alpha=0.1, outlier_ratio=0.01,
                   jsd_eps=1e-2):
    if scale_path is None:
        raise ValueError('ImpQuant requires scale_path')
    if run_impquant_process:
        state = run_impquant_weight_importance(model, prompt_inputs, prompt_kwargs,
            w_bit=w_bit, a_bit=a_bit, q_group_size=q_group_size,
            text_scale=text_scale, vision_scale=vision_scale, alpha=alpha,
            jsd_eps=jsd_eps, outlier_ratio=outlier_ratio)
        os.makedirs(os.path.dirname(scale_path) or '.', exist_ok=True)
        torch.save(state, scale_path)
    if pseudo_quant:
        state = torch.load(scale_path, map_location='cpu', weights_only=True)
        apply_impquant(model.model, w_bit, a_bit, {'q_group_size': q_group_size}, state)
    model.to_cuda()
    return model
