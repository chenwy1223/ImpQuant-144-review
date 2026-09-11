"""Token-weighted channel scale search for ImpQuant."""
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.utils.calibration import to_device


def normalize_scales(scales):
    """Normalize by the geometric mean of the largest and smallest channel scales."""
    scales = scales.float()
    return scales / (scales.max() * scales.min()).sqrt()


@torch.no_grad()
def auto_scale_block_token_weighted(module, module_kwargs, *, w_bit, q_config,
                                    input_feat, token_weights, n_grid=20):
    """Search the original channel groups with token-weighted reconstruction MSE.

    module_kwargs contains one captured attention-argument dictionary per sample,
    preserving position embeddings and masks for multimodal decoder layers.
    """
    def _search_module_scale(block, linears2scale, x, kwargs):
        param = next(block.parameters())
        device, dtype = param.device, param.dtype
        weights = token_weights.float().cpu()
        if x.ndim != 3 or weights.shape != x.shape[:2]:
            raise ValueError('Channel search requires aligned sample/token weights')
        denom = weights.sum()
        if denom <= 0:
            raise ValueError('Channel search requires positive token importance')
        act_scale = (x.float().abs() * weights.unsqueeze(-1)).sum((0, 1)) / denom
        def forward_sample(i):
            kw = to_device(kwargs[i], device) if kwargs is not None else {}
            kw.pop('use_cache', None)
            kw.pop('output_attentions', None)
            out = block(x[i:i+1].to(device=device, dtype=dtype), **kw)
            return out[0] if isinstance(out, tuple) else out
        reference = [forward_sample(i).detach().float().cpu() for i in range(len(x))]
        # Clone even on CPU: trial quantization must not overwrite the originals.
        originals = [linear.weight.detach().cpu().clone() for linear in linears2scale]
        best_loss, best_scales = float('inf'), None
        try:
            for r in range(n_grid):
                scales = normalize_scales(act_scale.pow(r / n_grid).clamp(min=1e-4))
                for linear, original in zip(linears2scale, originals):
                    s = scales.to(device).view(1, -1)
                    trial = original.to(device).float() * s
                    trial = pseudo_quantize_tensor(trial, n_bits=w_bit, **q_config) / s
                    linear.weight.copy_(trial.to(dtype))
                loss = 0.0
                for i in range(len(x)):
                    diff = forward_sample(i).float().cpu() - reference[i]
                    loss += (diff.square().mean(-1) * weights[i:i+1]).sum().item()
                loss /= denom.item()
                if loss < best_loss:
                    best_loss, best_scales = loss, scales.clone()
        finally:
            for linear, original in zip(linears2scale, originals):
                linear.weight.copy_(original.to(device))
        if best_scales is None or not torch.isfinite(best_scales).all():
            raise RuntimeError('ImpQuant channel scale search failed')
        return best_scales

    def get_op_name(op):
        return next(name for name, child in module.named_modules() if child is op)

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs=None):
        block = module2inspect if module2inspect is not None else layers[0]
        scales = _search_module_scale(block, layers, inp, kwargs)
        return get_op_name(prev_op), tuple(get_op_name(layer) for layer in layers), scales

    scales_list = []

    if isinstance(module, LlamaDecoderLayer) or module.__class__.__name__ in ("Qwen2DecoderLayer", "Qwen2VLDecoderLayer", "Qwen2_5_VLDecoderLayer"):
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

    elif module.__class__.__name__ == "InternLM2DecoderLayer":
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attention_norm,
                layers=[module.attention.wqkv],
                inp=input_feat["attention.wqkv"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        if module.attention.wqkv.weight.shape == module.attention.wo.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.attention.wqkv,
                    layers=[module.attention.wo],
                    inp=input_feat["attention.wo"],
                )
            )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn_norm,
                layers=[module.feed_forward.w1, module.feed_forward.w3],
                inp=input_feat["feed_forward.w1"],
                module2inspect=module.feed_forward,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.feed_forward.w3,
                layers=[module.feed_forward.w2],
                inp=input_feat["feed_forward.w2"],
            )
        )

    else:
        raise NotImplementedError(f'{type(module)} not supported for ImpQuant channel scaling')
    return scales_list
