import gc
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from qmllm.methods.mbq.quantize.pre_quant import get_blocks, move_embed, get_named_linears
from qmllm.methods.impquant.weight_quant import visual_importance, weighted_hessian
from qmllm.methods.impquant.auto_scale import auto_scale_block_token_weighted
from qmllm.methods.impquant.modality_weights import fixed_modality_weights
from qmllm.methods.gptq.quantizer import GPTQ
from qmllm.utils.calibration import slice_batch, to_device


def process_input(
    prompt_inputs: dict, prompt_kwargs: dict
) -> Tuple[
    Dict[str, Any],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    return inputs, inputs.get("attention_mask", None), vision_mask, caption_mask






def _expand_to_batch(value: Any, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        if value.shape[0] == 1:
            return value.expand(batch_size, *value.shape[1:])
        if value.dim() > 1 and value.shape[1] == 1 and value.shape[0] <= 4:
            return value.expand(value.shape[0], batch_size, *value.shape[2:])
    return value


def _tree_expand_to_batch(value: Any, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        return _expand_to_batch(value, batch_size)
    if isinstance(value, tuple):
        return tuple(_tree_expand_to_batch(v, batch_size) for v in value)
    if isinstance(value, list):
        return [_tree_expand_to_batch(v, batch_size) for v in value]
    if isinstance(value, dict):
        return {k: (v.expand(3, batch_size, -1) if k == "position_ids" and isinstance(v, torch.Tensor) and v.ndim == 3 else _tree_expand_to_batch(v, batch_size)) for k, v in value.items()}
    return value


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_tree_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_tree_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device) for k, v in value.items()}
    return value


def _find_non_flash_attention_base(cls: type) -> Optional[type]:
    for base in getattr(cls, "__mro__", ())[1:]:
        name = getattr(base, "__name__", "").lower()
        if ("attn" in name or "attention" in name) and "flash" not in name:
            return base
    return None


def _replace_flash_attention_modules(layers) -> list:
    """
    Best-effort: swap FlashAttention modules to their non-flash base class so we can
    run with output_attentions and avoid flash-attn padding/shape edge cases.
    Returns a list of (layer, attr_name, old_module) for restoring.
    """
    replaced = []
    for layer in layers:
        for attr in ("self_attn", "attn", "attention"):
            if not hasattr(layer, attr):
                continue
            attn = getattr(layer, attr, None)
            if attn is None:
                continue

            cname = attn.__class__.__name__.lower()
            if "flash" not in cname or ("attn" not in cname and "attention" not in cname):
                continue

            base_cls = _find_non_flash_attention_base(attn.__class__)
            if base_cls is None:
                continue

            cfg = getattr(attn, "config", None)
            layer_idx = getattr(attn, "layer_idx", None)
            try:
                new_attn = base_cls(cfg, layer_idx=layer_idx)
            except TypeError:
                try:
                    new_attn = base_cls(cfg)
                except Exception:
                    continue

            try:
                new_attn.load_state_dict(attn.state_dict(), strict=False)
            except Exception:
                continue

            try:
                p = next(attn.parameters())
                new_attn.to(device=p.device, dtype=p.dtype)
            except StopIteration:
                pass

            setattr(layer, attr, new_attn)
            replaced.append((layer, attr, attn))
    return replaced




def _extract_attn_list(outputs: Any) -> Optional[List[torch.Tensor]]:
    attn = getattr(outputs, "attentions", None)
    if isinstance(attn, (tuple, list)) and attn:
        return list(attn)

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if isinstance(item, (tuple, list)) and item and isinstance(item[0], torch.Tensor):
                if item[0].dim() in (3, 4):
                    return list(item)
    return None








def _first_token_logit_index(
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
) -> int:
    if isinstance(labels, torch.Tensor):
        labels = labels.view(-1)
        non_ignored = torch.nonzero(labels != -100, as_tuple=False)
        if non_ignored.numel() > 0:
            first = int(non_ignored[0].item())
            return max(first - 1, 0)
    if isinstance(attention_mask, torch.Tensor):
        attn = attention_mask.view(-1)
        valid = torch.nonzero(attn, as_tuple=False)
        if valid.numel() > 0:
            return int(valid[-1].item())
    return max(int(seq_len) - 1, 0)






@torch.no_grad()
def _collect_text_importance_last_layer_attn(
    base_model: nn.Module,
    inputs: Dict[str, Any],
    valid_mask_all: torch.Tensor,
    batch_size: int,
) -> Optional[torch.Tensor]:
    """
    Text token importance from the last layer attention:
        rho_i = (1/H) sum_h sum_k A^{(h)}[k,i]
    where k iterates over valid query positions.
    Returns a (total, seq_len) tensor on CPU, or None if attentions are unavailable.
    """
    base_model.train(False)
    device = next(base_model.parameters()).device
    total, seq_len = valid_mask_all.shape
    out = []

    for start in tqdm(range(0, total, batch_size), desc="impquant: last-layer attn..."):
        end = min(start + batch_size, total)
        chunk_inputs = slice_batch(inputs, start, end, total)
        chunk_inputs = _tree_to_device(chunk_inputs, device)
        chunk_inputs["use_cache"] = False
        chunk_inputs["output_attentions"] = True
        chunk_inputs["return_dict"] = True

        try:
            outputs = base_model(**chunk_inputs)
        except TypeError:
            chunk_inputs.pop("output_attentions", None)
            chunk_inputs.pop("return_dict", None)
            try:
                outputs = base_model(**chunk_inputs)
            except Exception:
                return None

        attn_list = _extract_attn_list(outputs)
        if not attn_list:
            return None

        last = attn_list[-1]
        if not isinstance(last, torch.Tensor):
            return None

        if last.dim() == 4:
            a = last.float()  # (B, H, N, N) or (H, B, N, N)
            if a.shape[0] != (end - start) and a.shape[1] == (end - start):
                a = a.transpose(0, 1)
            if a.shape[0] != (end - start) or a.shape[-1] != seq_len or a.shape[-2] != seq_len:
                return None
            attn_mean = a.mean(dim=1)  # (B, N, N)
        elif last.dim() == 3:
            a = last.float()
            if a.shape[0] == (end - start) and a.shape[-1] == seq_len and a.shape[-2] == seq_len:
                attn_mean = a  # (B, N, N)
            elif (end - start) == 1 and a.shape[1] == seq_len and a.shape[2] == seq_len:
                attn_mean = a.mean(dim=0, keepdim=True)
            else:
                return None
        else:
            return None

        qmask = valid_mask_all[start:end].to(device=attn_mean.device, dtype=attn_mean.dtype)  # (B, N)
        rho = (attn_mean * qmask.unsqueeze(-1)).sum(dim=1)  # sum over queries -> (B, N)
        rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        out.append(rho.detach().cpu())

        del outputs, attn_list, last, attn_mean, rho
        torch.cuda.empty_cache()

    if not out:
        return None
    return torch.cat(out, dim=0)


@torch.no_grad()
def _calculate_jsd_from_logits(base_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence computed from (unnormalized) logits over the same support.
    Expects shape (B, K) and returns shape (B,).
    """
    if base_logits.shape != ref_logits.shape:
        raise ValueError('JSD distributions must have the same support')
    log_p = F.log_softmax(base_logits.float(), dim=-1)
    log_q = F.log_softmax(ref_logits.float(), dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - torch.log(torch.tensor(2., device=log_p.device))
    return 0.5 * ((log_p.exp() * (log_p - log_m)).sum(-1)
                  + (log_q.exp() * (log_q - log_m)).sum(-1))


@torch.no_grad()
def _collect_leave_one_out_jsd(
    base_model: nn.Module,
    inputs: Dict[str, Any],
    valid_mask_all: torch.Tensor,
    vision_mask_all: Optional[torch.Tensor],
    tokens_per_step: int,
) -> Optional[torch.Tensor]:
    """
    Context-centric leave-one-out: ablate one visual token at a time by replacing its embedding with a zero vector,
    then compute JSD between baseline and ablated next-token distributions over the full vocabulary.
    Returns a (total, seq_len) tensor with values at visual token positions, and -1 elsewhere.
    """
    if vision_mask_all is None:
        return None

    inputs_embeds = inputs.get("inputs_embeds", None)
    if not isinstance(inputs_embeds, torch.Tensor):
        return None

    total, seq_len, dim = inputs_embeds.shape
    device = next(base_model.parameters()).device
    labels = inputs.get("labels", None)
    attention_mask = inputs.get("attention_mask", None)

    logit_indices = []
    for b in range(total):
        lab = labels[b] if isinstance(labels, torch.Tensor) else None
        attn = None
        if isinstance(attention_mask, torch.Tensor):
            attn = attention_mask[b]
        elif isinstance(valid_mask_all, torch.Tensor):
            attn = valid_mask_all[b]
        logit_indices.append(_first_token_logit_index(lab, attn, seq_len))

    chunk_size = max(1, min(int(tokens_per_step), 16))
    jsd_weights = torch.full((total, seq_len), -1.0, dtype=torch.float32)

    base_model.train(False)

    for b in tqdm(range(total), desc="impquant: leave-one-out JSD..."):
        vis_mask = vision_mask_all[b] if isinstance(vision_mask_all, torch.Tensor) else None
        if not isinstance(vis_mask, torch.Tensor):
            continue
        if vis_mask.dim() != 1:
            vis_mask = vis_mask.view(-1)
        if not vis_mask.any():
            continue

        vis_idx = torch.nonzero(vis_mask, as_tuple=False).view(-1)
        if vis_idx.numel() == 0:
            continue

        logit_idx = logit_indices[b]
        if logit_idx < 0 or logit_idx >= seq_len:
            continue

        sample_inputs = slice_batch(inputs, b, b + 1, total)
        sample_inputs = _tree_to_device(sample_inputs, device)

        sample_embeds = sample_inputs.get("inputs_embeds", None)
        if not isinstance(sample_embeds, torch.Tensor):
            continue
        base_embeds = sample_embeds.squeeze(0).detach()
        if base_embeds.dim() != 2 or base_embeds.shape[0] != seq_len:
            continue

        # Prepare forward kwargs (we'll provide inputs_embeds explicitly).
        sample_inputs.pop("inputs_embeds", None)
        sample_inputs.pop("input_ids", None)
        sample_inputs.pop("pixel_values", None)
        sample_inputs.pop("output_attentions", None)
        sample_inputs["use_cache"] = False
        sample_inputs["return_dict"] = True

        # Baseline once per sample.
        base_inputs = dict(sample_inputs)
        base_inputs["inputs_embeds"] = base_embeds.unsqueeze(0)
        try:
            base_out = base_model(**base_inputs)
        except TypeError:
            base_inputs.pop("return_dict", None)
            base_out = base_model(**base_inputs)

        base_logits = getattr(base_out, "logits", None)
        if base_logits is None and isinstance(base_out, (tuple, list)) and base_out:
            base_logits = base_out[0]
        if not isinstance(base_logits, torch.Tensor) or base_logits.shape[1] <= logit_idx:
            continue

        base_step = base_logits[0, logit_idx, :].float()

        vis_idx_device = vis_idx.to(device=device)
        zero_vec = torch.zeros((dim,), device=device, dtype=base_embeds.dtype)

        for start in range(0, vis_idx.numel(), chunk_size):
            end = min(start + chunk_size, vis_idx.numel())
            chunk_ids = vis_idx[start:end]
            if chunk_ids.numel() == 0:
                continue

            chunk_ids_device = chunk_ids.to(device=device)
            chunk_embeds = base_embeds.unsqueeze(0).repeat(chunk_ids.numel(), 1, 1)
            chunk_embeds[torch.arange(chunk_ids.numel(), device=device), chunk_ids_device, :] = zero_vec

            chunk_inputs = _tree_expand_to_batch(sample_inputs, int(chunk_ids.numel()))
            chunk_inputs["inputs_embeds"] = chunk_embeds

            try:
                out = base_model(**chunk_inputs)
            except TypeError:
                chunk_inputs.pop("return_dict", None)
                out = base_model(**chunk_inputs)

            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, (tuple, list)) and out:
                logits = out[0]
            if not isinstance(logits, torch.Tensor) or logits.shape[1] <= logit_idx:
                continue

            ab_step = logits[:, logit_idx, :].float()
            jsd = _calculate_jsd_from_logits(base_step.unsqueeze(0).expand_as(ab_step), ab_step)
            jsd = torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            jsd_weights[b, chunk_ids] = jsd.detach().cpu()

            del out, logits, ab_step, jsd, chunk_inputs, chunk_embeds
            torch.cuda.empty_cache()

        del base_out, base_logits, base_step, base_inputs
        torch.cuda.empty_cache()

    return jsd_weights


@torch.no_grad()
def _collect_one_token_redundancy(
    base_model: nn.Module,
    inputs: Dict[str, Any],
    valid_mask_all: torch.Tensor,
    vision_mask_all: Optional[torch.Tensor],
    tokens_per_step: int,
) -> Optional[torch.Tensor]:
    """
    Single-token repetition: repeat a single visual token across
    all visual slots, then measure top-1 probability of the first generated token.
    """
    if vision_mask_all is None:
        return None

    inputs_embeds = inputs.get("inputs_embeds", None)
    if not isinstance(inputs_embeds, torch.Tensor):
        return None

    total, seq_len, _ = inputs_embeds.shape
    device = next(base_model.parameters()).device
    labels = inputs.get("labels", None)
    attention_mask = inputs.get("attention_mask", None)

    logit_indices = []
    for b in range(total):
        lab = labels[b] if isinstance(labels, torch.Tensor) else None
        attn = None
        if isinstance(attention_mask, torch.Tensor):
            attn = attention_mask[b]
        elif isinstance(valid_mask_all, torch.Tensor):
            attn = valid_mask_all[b]
        logit_indices.append(_first_token_logit_index(lab, attn, seq_len))

    chunk_size = max(1, min(int(tokens_per_step), 16))
    redundancy_weights = torch.full((total, seq_len), -1.0, dtype=torch.float32)

    base_model.train(False)

    for b in tqdm(range(total), desc="impquant: 1-token redundancy..."):
        vis_mask = vision_mask_all[b] if isinstance(vision_mask_all, torch.Tensor) else None
        if not isinstance(vis_mask, torch.Tensor):
            continue
        if vis_mask.dim() != 1:
            vis_mask = vis_mask.view(-1)
        if not vis_mask.any():
            continue

        vis_idx = torch.nonzero(vis_mask, as_tuple=False).view(-1)
        if vis_idx.numel() == 0:
            continue

        logit_idx = logit_indices[b]
        if logit_idx < 0 or logit_idx >= seq_len:
            continue

        sample_inputs = slice_batch(inputs, b, b + 1, total)
        sample_inputs = _tree_to_device(sample_inputs, device)

        sample_embeds = sample_inputs.get("inputs_embeds", None)
        if not isinstance(sample_embeds, torch.Tensor):
            continue
        base_embeds = sample_embeds.squeeze(0).detach()
        if base_embeds.dim() != 2 or base_embeds.shape[0] != seq_len:
            continue

        sample_inputs.pop("inputs_embeds", None)
        sample_inputs.pop("input_ids", None)
        sample_inputs.pop("pixel_values", None)
        sample_inputs.pop("output_attentions", None)
        sample_inputs["use_cache"] = False
        sample_inputs["return_dict"] = True

        vis_idx_device = vis_idx.to(device=device)

        for start in range(0, vis_idx.numel(), chunk_size):
            end = min(start + chunk_size, vis_idx.numel())
            chunk_ids = vis_idx[start:end]
            if chunk_ids.numel() == 0:
                continue

            chunk_ids_device = chunk_ids.to(device=device)
            chunk_embeds = base_embeds.unsqueeze(0).repeat(chunk_ids.numel(), 1, 1)
            selected = base_embeds.index_select(0, chunk_ids_device)
            chunk_embeds[:, vis_idx_device, :] = selected.unsqueeze(1).expand(-1, vis_idx_device.numel(), -1)

            chunk_inputs = _tree_expand_to_batch(sample_inputs, int(chunk_ids.numel()))
            chunk_inputs["inputs_embeds"] = chunk_embeds

            try:
                outputs = base_model(**chunk_inputs)
            except TypeError:
                chunk_inputs.pop("return_dict", None)
                outputs = base_model(**chunk_inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                logits = outputs[0]
            if not isinstance(logits, torch.Tensor) or logits.shape[1] <= logit_idx:
                continue

            logits_sel = logits[:, logit_idx, :].float()
            probs = torch.softmax(logits_sel, dim=-1)
            max_prob = probs.max(dim=-1).values
            redundancy = torch.nan_to_num(1.0 - max_prob, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=0.0, max=1.0)
            redundancy_weights[b, chunk_ids] = redundancy.detach().cpu()

            del outputs, logits, probs, max_prob, chunk_inputs, chunk_embeds
            torch.cuda.empty_cache()

    return redundancy_weights









@torch.no_grad()
def _collect_global_token_importance(base_model, inputs, valid_mask_all,
                                     vision_mask_all, batch_size, tokens_per_step,
                                     text_scale=1.0, vision_scale=1.0, alpha=0.1, jsd_eps=1e-2):
    text = _collect_text_importance_last_layer_attn(base_model, inputs, valid_mask_all, batch_size)
    if text is None:
        raise RuntimeError('ImpQuant calibration requires eager attention with attention weights')
    weights = text * text_scale
    if vision_mask_all is not None and vision_mask_all.any():
        redundancy = _collect_one_token_redundancy(base_model, inputs, valid_mask_all,
                                                  vision_mask_all, tokens_per_step)
        jsd = _collect_leave_one_out_jsd(base_model, inputs, valid_mask_all,
                                         vision_mask_all, tokens_per_step)
        if redundancy is None or jsd is None:
            raise RuntimeError('ImpQuant visual probes require prepared inputs_embeds')
        mask = vision_mask_all & valid_mask_all
        if (redundancy[mask] < 0).any() or (jsd[mask] < 0).any():
            raise RuntimeError('A visual importance probe failed')
        weights[mask] = visual_importance(1 - redundancy[mask], jsd[mask], alpha, vision_scale, jsd_eps)
    weights *= valid_mask_all.float()
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError('Invalid ImpQuant token importance')
    return weights


@torch.no_grad()
def run_impquant_weight_importance(model, prompt_inputs, prompt_kwargs, *, w_bit,
                                    a_bit, q_group_size, batch_size=1,
                                    outlier_ratio=0.01, tokens_per_step=1,
                                    text_scale=1.0, vision_scale=1.0, alpha=0.1, jsd_eps=1e-2,
                                    percdamp=0.01):
    base = model.model
    # InternVL's language model accepts prepared multimodal embeddings.
    probe_model = base.language_model if base.__class__.__name__ == 'InternVLChatModel' else base
    layers = get_blocks(base)
    inputs, attention_mask, vision_mask, _ = process_input(prompt_inputs, prompt_kwargs)
    total, seq_len = inputs['inputs_embeds'].shape[:2]
    valid = attention_mask.detach().cpu().bool() if attention_mask is not None else torch.ones(total, seq_len, dtype=torch.bool)
    vision = vision_mask.detach().cpu().bool() if vision_mask is not None else None
    model.to_cuda()
    config = probe_model.config
    old_impl = getattr(config, '_attn_implementation', None)
    config._attn_implementation = 'eager'
    replaced = _replace_flash_attention_modules(layers)
    try:
        raw_weights = _collect_global_token_importance(probe_model, inputs, valid, vision,
                    batch_size, tokens_per_step, 1.0, 1.0, alpha, jsd_eps)
    finally:
        for layer, attr, old_module in replaced:
            setattr(layer, attr, old_module)
        config._attn_implementation = old_impl

    weights, modality_scales = fixed_modality_weights(raw_weights, valid, vision, text_scale, vision_scale)

    captured, kwargs_list = [], []
    class CaptureComplete(Exception):
        pass
    class Catcher(nn.Module):
        def forward(self, inp, **kwargs):
            captured.append(inp.detach().cpu())
            kwargs_list.append(to_device(kwargs, 'cpu'))
            raise CaptureComplete
    first = layers[0]
    layers[0] = Catcher()
    try:
        for i in range(total):
            try:
                model(**slice_batch(inputs, i, i + 1, total))
            except CaptureComplete:
                pass
    finally:
        layers[0] = first
    if len(captured) != total:
        raise RuntimeError('ImpQuant failed to capture calibration samples')
    model.to_cpu()
    results = {'method': 'impquant', 'format_version': 2,
               'weight_solver': 'gptq_inverse_hessian_cholesky', 'w_bit': w_bit,
               'scale_normalization': 'sqrt_max_times_min',
               'a_bit': a_bit, 'q_group_size': q_group_size,
               'text_scale': text_scale, 'vision_scale': vision_scale, 'alpha': alpha,
               'jsd_eps': jsd_eps, 'modality_weighting': 'fixed',
               'modality_scales': modality_scales,
               'outlier_ratio': outlier_ratio, 'percdamp': percdamp,
               'effective_percdamp': [],
               'weights': [], 'act_outlier_masks': [], 'input_scales': []}
    for layer in tqdm(layers, desc='ImpQuant calibration'):
        layer.cuda()
        device = next(layer.parameters()).device
        layer_weights, masks = {}, {}
        layer_damping = {}
        named_linears = get_named_linears(layer)
        layer_scales = {name: torch.ones(linear.in_features) for name, linear in named_linears.items()}
        if w_bit < 16:
            input_feat = {name: [] for name in named_linears}
            handles = [linear.register_forward_hook(
                lambda module, args, output, key=name: input_feat[key].append(args[0].detach().cpu()))
                for name, linear in named_linears.items()]
            try:
                for i in range(total):
                    layer(captured[i].to(device), **to_device(kwargs_list[i], device))
            finally:
                for handle in handles:
                    handle.remove()
            scale_groups = auto_scale_block_token_weighted(
                layer, kwargs_list, w_bit=w_bit,
                q_config={'zero_point': False, 'q_group_size': q_group_size},
                input_feat={name: torch.cat(values, dim=0) for name, values in input_feat.items()},
                token_weights=weights)
            for _, names, scales in scale_groups:
                for name in names:
                    layer_scales[name] = scales
            del input_feat, scale_groups
        for name, linear in get_named_linears(layer).items():
            solver = GPTQ(linear)
            solver.quantizer.configure(w_bit, perchannel=True, sym=False)
            sum_abs = torch.zeros(linear.in_features, device=device)
            count = 0
            sample = 0
            def collect(module, args, output):
                nonlocal count
                x = args[0].detach().reshape(-1, linear.in_features).float()
                mask = valid[sample].to(device)
                x = x[mask]
                p = weights[sample].to(device)[mask]
                solver.H.add_(weighted_hessian(x, p))
                sum_abs.add_(x.abs().sum(0))
                count += x.shape[0]
            handle = linear.register_forward_hook(collect)
            try:
                for sample in range(total):
                    layer(captured[sample].to(device), **to_device(kwargs_list[sample], device))
            finally:
                handle.remove()
            if count == 0:
                raise RuntimeError('No valid calibration tokens')
            k = max(1, min(linear.in_features - 1, round(outlier_ratio * linear.in_features)))
            mask = torch.zeros(linear.in_features, dtype=torch.bool)
            mask[(sum_abs / count).topk(k).indices.cpu()] = True
            masks[name] = mask
            scale = layer_scales[name].to(device).float()
            # Reparameterize W -> W S and X -> S^-1 X before GPTQ.
            # Thus H -> S^-1 H S^-1; token importance is unchanged.
            solver.H.div_(scale[:, None]).div_(scale[None, :])
            linear.weight.copy_((linear.weight.float() * scale[None, :]).to(linear.weight.dtype))
            if not torch.isfinite(linear.weight).all():
                raise RuntimeError(f'Channel scaling overflow at {name}')
            if w_bit < 16:
                # Standard GPTQ: inverse-Hessian Cholesky sequential compensation.
                solver.fasterquant(percdamp=percdamp, groupsize=q_group_size)
                layer_damping[name] = solver.effective_percdamp
            solver.free()
            layer_weights[name] = linear.weight.detach().cpu().clone()
            # Keep the calibration layer in its original input coordinates.
            linear.weight.copy_((linear.weight.float() / scale[None, :]).to(linear.weight.dtype))
        next_inputs = []
        for i in range(total):
            output = layer(captured[i].to(device), **to_device(kwargs_list[i], device))
            next_inputs.append(output[0].detach().cpu())
        captured = next_inputs
        results['weights'].append(layer_weights)
        results['effective_percdamp'].append(layer_damping)
        results['act_outlier_masks'].append(masks)
        results['input_scales'].append(layer_scales)
        layer.cpu()
        torch.cuda.empty_cache()
    return results
