import os
import torch

from qmllm.methods.smoothquant.quantize.smooth import smooth_lm, smooth_vit
from qmllm.methods.smoothquant.quantize.quantizer import quantize_model, pseudo_quantize_model_weight_act
from qmllm.methods.smoothquant.quantize.gen_act_scales import get_act_scales
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight as _pseudo_quantize_model_weight

def smoothquant_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_sq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    w_bit: int = 4,
    a_bit: int = 8,
    alpha: float = 0.5,
    q_group_size: int = 128,
    zero_point: bool = True,
):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    '''

    assert scale_path is not None

    if run_sq_process:
        act_scales = get_act_scales(model, prompt_inputs, prompt_kwargs)
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        
        torch.save(act_scales, scale_path)
        print("SmoothQuant results saved at", scale_path)
    else:
        act_scales = torch.load(scale_path)
    
    if pseudo_quant:
        smooth_lm(model.model, act_scales, alpha)
        if a_bit is None or int(a_bit) >= 16:
            q_config = {"zero_point": bool(zero_point), "q_group_size": int(q_group_size)}
            _pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    if hasattr(model, "to_cuda"):
        model.to_cuda()
    
    return model
