import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.awq.quantize.auto_scale import auto_scale_block, apply_scale

__all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenModel":
        # LlavaQwenModel inherits HF Qwen2Model
        layers = model.layers
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "Qwen2VLModel":
        layers = model.layers
    elif model.__class__.__name__ == "Qwen2_5_VLModel":
        layers = model.layers
    elif model.__class__.__name__ == "Qwen2Model":
        layers = model.layers
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.to(device)
            if hasattr(model.model, "rotary_emb"):
                try:
                    model.model.rotary_emb = model.model.rotary_emb.to(device)
                except Exception:
                    pass
        elif hasattr(model, "embed_tokens"):
            model.embed_tokens = model.embed_tokens.to(device)
            if hasattr(model, "rotary_emb"):
                try:
                    model.rotary_emb = model.rotary_emb.to(device)
                except Exception:
                    pass
    elif model.__class__.__name__ == "LlavaQwenModel":
        model.embed_tokens = model.embed_tokens.to(device)
        if hasattr(model, "rotary_emb"):
            try:
                model.rotary_emb = model.rotary_emb.to(device)
            except Exception:
                pass
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2VLModel":
        model.embed_tokens = model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2_5_VLModel":
        model.embed_tokens = model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2Model":
        model.embed_tokens = model.embed_tokens.to(device)
        if hasattr(model, "rotary_emb"):
            try:
                model.rotary_emb = model.rotary_emb.to(device)
            except Exception:
                pass
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        # InternLM2 wraps the actual module under .model
        if hasattr(model, "model") and hasattr(model.model, "tok_embeddings"):
            model.model.tok_embeddings = model.model.tok_embeddings.to(device)
        elif hasattr(model, "tok_embeddings"):
            model.tok_embeddings = model.tok_embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)  
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq(
    model,
    prompt_tokens,
    prompt_mask,
    w_bit,
    q_config,
    auto_scale=True,
):
    class _CatcherExit(Exception):
        pass

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise _CatcherExit  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    device = next(model.parameters()).device

    def _call_model_with_inputs(tokens, mask):
        if isinstance(tokens, torch.Tensor):
            if tokens.dtype in (torch.int32, torch.int64):
                return model(tokens.to(device), attention_mask=mask, use_cache=False)
            else:
                return model(inputs_embeds=tokens.to(device), attention_mask=mask, use_cache=False)
        elif isinstance(tokens, (list, tuple)) and len(tokens) > 0 and isinstance(tokens[0], torch.Tensor):
            t0 = tokens[0]
            if t0.dtype in (torch.int32, torch.int64):
                return model(t0.to(device), attention_mask=mask, use_cache=False)
            else:
                return model(inputs_embeds=t0.to(device), attention_mask=mask, use_cache=False)
        else:
            raise ValueError("AWQ calibration requires input_ids or inputs_embeds as torch.Tensor.")

    try:
        _call_model_with_inputs(prompt_tokens, prompt_mask)
    except _CatcherExit:
        pass
    
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                ans_mask=prompt_mask,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])