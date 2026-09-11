import os
import torch

from qmllm.methods.gptq.quantizer import apply_gptq_state, run_gptq


def gptq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_gptq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int = -1,
    actorder: bool = False,
    w_bit: int = 4,
):
    # GPTQ itself does not require a "scale" file. In this repo we reuse `scale_path` as an optional
    # cache location for quantized weights/state so that `main.py --pseudo_quant` can work.
    if pseudo_quant and (not run_gptq_process) and (not scale_path):
        raise ValueError("[gptq] pseudo_quant in main.py requires scale_path (no calibration inputs are provided).")

    base_model = model.model if hasattr(model, "model") else model
    if base_model.__class__.__name__ == "InternVLChatModel":
        base_model = base_model.language_model

    if run_gptq_process:
        quant_state = run_gptq(
            base_model,
            prompt_inputs,
            prompt_kwargs,
            blocksize=blocksize,
            percdamp=percdamp,
            groupsize=groupsize,
            actorder=actorder,
            w_bit=int(w_bit),
        )
        if scale_path:
            dirpath = os.path.dirname(scale_path)
            if dirpath:
                os.makedirs(dirpath or ".", exist_ok=True)
            torch.save(quant_state, scale_path)
            print(f"[gptq] quant state saved at {scale_path}")
        else:
            print("[gptq] run_process completed (no scale_path provided; skipped saving quant state).")

    if pseudo_quant and not run_gptq_process:
        if not os.path.exists(scale_path):
            raise FileNotFoundError(f"[gptq] scale_path not found: {scale_path}")
        quant_state = torch.load(scale_path, map_location="cpu", weights_only=True)
        cfg = quant_state.get("config", {})
        if cfg.get("w_bit") != w_bit or cfg.get("groupsize") != groupsize:
            raise ValueError("GPTQ cache precision or group size does not match the request")
        apply_gptq_state(base_model, quant_state)
        print(f"[gptq] loaded and applied quant state from {scale_path}")

    if hasattr(model, "to_cuda"):
        model.to_cuda()
    else:
        base_model.cuda()

    return model
