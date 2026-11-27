import os
import torch
from qmllm.methods.impquant.pre_quant import run_impquant_weight_importance
from qmllm.methods.impquant.quantizer import apply_impquant


def impquant_entry(model, prompt_inputs, prompt_kwargs, run_impquant_process,
                   pseudo_quant, scale_path=None, q_group_size=128, w_bit=4, a_bit=8,
                   text_scale=1.0, vision_scale=1.0, alpha=0.1, outlier_ratio=0.01,
                   jsd_eps=1e-2, percdamp=0.01, probe_cache_path=None,
                   probe_source_path=None, probe_only=False, skip_probe1=False,
                   probe2_mode='exact', probe_group_h=2, probe_group_w=2,
                   probe_refine_ratio=0.25, probe_refine_strategy='top_jsd',
                   probe_refine_seed=2026):
    if scale_path is None and not probe_only:
        raise ValueError('ImpQuant requires scale_path')
    if probe_only and not run_impquant_process:
        raise ValueError('ImpQuant probe-only mode is available only during calibration')
    if probe_only and probe_cache_path is None:
        raise ValueError('ImpQuant probe-only mode requires probe_cache_path')
    if run_impquant_process:
        state = run_impquant_weight_importance(model, prompt_inputs, prompt_kwargs,
            w_bit=w_bit, a_bit=a_bit, q_group_size=q_group_size,
            text_scale=text_scale, vision_scale=vision_scale, alpha=alpha,
            jsd_eps=jsd_eps, outlier_ratio=outlier_ratio, percdamp=percdamp,
            probe_cache_path=probe_cache_path, probe_source_path=probe_source_path,
            probe_only=probe_only,
            skip_probe1=skip_probe1,
            probe2_mode=probe2_mode,
            probe_group_h=probe_group_h,
            probe_group_w=probe_group_w,
            probe_refine_ratio=probe_refine_ratio,
            probe_refine_strategy=probe_refine_strategy,
            probe_refine_seed=probe_refine_seed)
        if not probe_only:
            os.makedirs(os.path.dirname(scale_path) or '.', exist_ok=True)
            torch.save(state, scale_path)
    if pseudo_quant:
        state = torch.load(scale_path, map_location='cpu', weights_only=True)
        apply_impquant(model.model, w_bit, a_bit, {'q_group_size': q_group_size}, state)
    model.to_cuda()
    return model
