import os
import torch

from qmllm.methods.awq.quantize.pre_quant import run_awq, apply_awq
from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight


def awq_entry(model, prompt_inputs, prompt_kwargs, run_awq_process: bool, scale_path: str=None, zero_point: str=True, q_group_size: int=128, w_bit: int=4):
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    # unwrap to HF model
    base_model = model.fetch_llm() if hasattr(model, "fetch_llm") else model.model if hasattr(model, "model") else model

    def extract_tokens_and_mask(src):
        tok = None
        mask = None
        if isinstance(src, dict):
            mask = src.get("attention_mask")
            tok = src.get("input_ids") or src.get("inputs_embeds")
        elif isinstance(src, torch.Tensor):
            tok = src
        elif isinstance(src, (list, tuple)) and len(src) > 0 and isinstance(src[0], torch.Tensor):
            tok = src[0]
            if len(src) > 1 and isinstance(src[1], torch.Tensor):
                mask = src[1]
        return tok, mask

    prompt_tokens, prompt_mask = extract_tokens_and_mask(prompt_inputs)
    tok2, mask2 = extract_tokens_and_mask(prompt_kwargs)
    if prompt_tokens is None and tok2 is not None:
        prompt_tokens = tok2
        if prompt_mask is None:
            prompt_mask = mask2

    if run_awq_process:
        awq_results = run_awq(
            base_model,
            prompt_tokens,
            prompt_mask,
            w_bit=w_bit,
            q_config=q_config,
            auto_scale=True,
        )
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(awq_results, scale_path)
        print("AWQ results saved at", scale_path)
    else:
        awq_results = torch.load(scale_path, map_location="cpu")
        apply_awq(base_model, awq_results)

    # weight quantization
    pseudo_quantize_model_weight(base_model, w_bit=w_bit, q_config=q_config)

    # move back to cuda for inference
    if hasattr(model, "to_cuda"):
        model.to_cuda()
    else:
        base_model.cuda()

    return model
