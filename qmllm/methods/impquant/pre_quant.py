import gc
import os
import time
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
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Context-centric leave-one-out: ablate one visual token at a time by replacing its embedding with a zero vector,
    then compute JSD between baseline and ablated next-token distributions over the full vocabulary.
    Returns a (total, seq_len) tensor with values at visual token positions, and -1 elsewhere,
    plus one-forward baseline statistics for each sample.
    """
    if vision_mask_all is None:
        return None, {}

    inputs_embeds = inputs.get("inputs_embeds", None)
    if not isinstance(inputs_embeds, torch.Tensor):
        return None, {}

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
    baseline_entropy = torch.full((total,), float('nan'), dtype=torch.float32)
    baseline_confidence = torch.full((total,), float('nan'), dtype=torch.float32)
    baseline_margin = torch.full((total,), float('nan'), dtype=torch.float32)

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
        base_probs = torch.softmax(base_step, dim=-1)
        baseline_entropy[b] = (
            -(base_probs * base_probs.clamp_min(1e-12).log()).sum().detach().cpu()
        )
        top2 = base_probs.topk(k=min(2, base_probs.numel())).values
        baseline_confidence[b] = top2[0].detach().cpu()
        baseline_margin[b] = (
            (top2[0] - top2[1]).detach().cpu() if top2.numel() > 1 else top2[0].detach().cpu()
        )

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

        del base_out, base_logits, base_step, base_probs, top2, base_inputs
        torch.cuda.empty_cache()

    return jsd_weights, {
        'baseline_entropy': baseline_entropy,
        'baseline_confidence': baseline_confidence,
        'baseline_margin': baseline_margin,
        'logit_index': torch.tensor(logit_indices, dtype=torch.long),
    }


def _infer_visual_grid(num_tokens: int) -> Tuple[int, int]:
    """Infer a compact row-major grid from the merged visual-token count.

    Qwen2.5-VL calibration examples in this project contain one image per
    sequence, and the visual token count is the product of its post-merge grid
    dimensions.  Orientation does not affect 2-D local grouping, so the factor
    pair closest to square is sufficient when explicit grid metadata is absent.
    """
    if num_tokens <= 0:
        raise ValueError('Visual token count must be positive')
    height = int(num_tokens ** 0.5)
    while height > 1 and num_tokens % height:
        height -= 1
    return height, num_tokens // height


def _build_spatial_groups(vis_idx: torch.Tensor, group_h: int,
                          group_w: int) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
    """Partition row-major visual indices into non-overlapping local groups."""
    if group_h <= 0 or group_w <= 0:
        raise ValueError('Probe group dimensions must be positive')
    height, width = _infer_visual_grid(int(vis_idx.numel()))
    local = torch.arange(vis_idx.numel(), dtype=torch.long).reshape(height, width)
    groups = []
    for row in range(0, height, group_h):
        for col in range(0, width, group_w):
            members = local[row:row + group_h, col:col + group_w].reshape(-1)
            groups.append(vis_idx.index_select(0, members))
    return groups, (height, width)


@torch.no_grad()
def _collect_grouped_leave_one_out_jsd(
    base_model: nn.Module,
    inputs: Dict[str, Any],
    valid_mask_all: torch.Tensor,
    vision_mask_all: Optional[torch.Tensor],
    tokens_per_step: int,
    group_h: int = 2,
    group_w: int = 2,
    refine_ratio: float = 0.0,
    refine_strategy: str = 'top_jsd',
    refine_seed: int = 2026,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Estimate patch JSD with spatial group screening and exact refinement.

    Every group is ablated once.  Its JSD density (group JSD divided by group
    size) is assigned to unrefined patches.  A budgeted subset of groups is then
    refined with the original single-patch leave-one-out intervention.
    """
    if vision_mask_all is None:
        return None, {}
    if not 0 <= refine_ratio <= 1:
        raise ValueError('Probe refinement ratio must be in [0, 1]')
    if refine_strategy not in ('top_jsd', 'random'):
        raise ValueError(f'Unknown refinement strategy: {refine_strategy}')

    inputs_embeds = inputs.get('inputs_embeds', None)
    if not isinstance(inputs_embeds, torch.Tensor):
        return None, {}

    total, seq_len, _ = inputs_embeds.shape
    device = next(base_model.parameters()).device
    labels = inputs.get('labels', None)
    attention_mask = inputs.get('attention_mask', None)
    logit_indices = []
    for batch_idx in range(total):
        lab = labels[batch_idx] if isinstance(labels, torch.Tensor) else None
        attn = attention_mask[batch_idx] if isinstance(attention_mask, torch.Tensor) else valid_mask_all[batch_idx]
        logit_indices.append(_first_token_logit_index(lab, attn, seq_len))

    chunk_size = max(1, min(int(tokens_per_step), 16))
    jsd_weights = torch.full((total, seq_len), -1.0, dtype=torch.float32)
    baseline_entropy = torch.full((total,), float('nan'), dtype=torch.float32)
    baseline_confidence = torch.full((total,), float('nan'), dtype=torch.float32)
    baseline_margin = torch.full((total,), float('nan'), dtype=torch.float32)
    group_count = torch.zeros(total, dtype=torch.long)
    refined_group_count = torch.zeros(total, dtype=torch.long)
    group_probe_calls = torch.zeros(total, dtype=torch.long)
    exact_probe_calls = torch.zeros(total, dtype=torch.long)
    inferred_grid_hw = torch.zeros((total, 2), dtype=torch.long)

    base_model.train(False)
    for batch_idx in tqdm(range(total), desc='impquant: grouped leave-one-out JSD...'):
        vis_mask = vision_mask_all[batch_idx]
        if vis_mask.dim() != 1:
            vis_mask = vis_mask.view(-1)
        vis_idx = torch.nonzero(vis_mask, as_tuple=False).view(-1)
        if vis_idx.numel() == 0:
            continue
        logit_idx = logit_indices[batch_idx]
        if logit_idx < 0 or logit_idx >= seq_len:
            continue

        sample_inputs = slice_batch(inputs, batch_idx, batch_idx + 1, total)
        sample_inputs = _tree_to_device(sample_inputs, device)
        sample_embeds = sample_inputs.get('inputs_embeds', None)
        if not isinstance(sample_embeds, torch.Tensor):
            continue
        base_embeds = sample_embeds.squeeze(0).detach()
        sample_inputs.pop('inputs_embeds', None)
        sample_inputs.pop('input_ids', None)
        sample_inputs.pop('pixel_values', None)
        sample_inputs.pop('output_attentions', None)
        sample_inputs['use_cache'] = False
        sample_inputs['return_dict'] = True

        base_inputs = dict(sample_inputs)
        base_inputs['inputs_embeds'] = base_embeds.unsqueeze(0)
        try:
            base_out = base_model(**base_inputs)
        except TypeError:
            base_inputs.pop('return_dict', None)
            base_out = base_model(**base_inputs)
        base_logits = getattr(base_out, 'logits', None)
        if base_logits is None and isinstance(base_out, (tuple, list)) and base_out:
            base_logits = base_out[0]
        if not isinstance(base_logits, torch.Tensor) or base_logits.shape[1] <= logit_idx:
            continue
        base_step = base_logits[0, logit_idx, :].float()
        base_probs = torch.softmax(base_step, dim=-1)
        baseline_entropy[batch_idx] = (
            -(base_probs * base_probs.clamp_min(1e-12).log()).sum().detach().cpu()
        )
        top2 = base_probs.topk(k=min(2, base_probs.numel())).values
        baseline_confidence[batch_idx] = top2[0].detach().cpu()
        baseline_margin[batch_idx] = (
            (top2[0] - top2[1]).detach().cpu() if top2.numel() > 1 else top2[0].detach().cpu()
        )

        groups, grid_hw = _build_spatial_groups(vis_idx.cpu(), group_h, group_w)
        inferred_grid_hw[batch_idx] = torch.tensor(grid_hw, dtype=torch.long)
        group_count[batch_idx] = len(groups)
        group_probe_calls[batch_idx] = len(groups)
        group_jsd = torch.zeros(len(groups), dtype=torch.float32)

        for start in range(0, len(groups), chunk_size):
            batch_groups = groups[start:start + chunk_size]
            variants = base_embeds.unsqueeze(0).repeat(len(batch_groups), 1, 1)
            for local_idx, members in enumerate(batch_groups):
                variants[local_idx, members.to(device=device), :] = 0
            chunk_inputs = _tree_expand_to_batch(sample_inputs, len(batch_groups))
            chunk_inputs['inputs_embeds'] = variants
            try:
                out = base_model(**chunk_inputs)
            except TypeError:
                chunk_inputs.pop('return_dict', None)
                out = base_model(**chunk_inputs)
            logits = getattr(out, 'logits', None)
            if logits is None and isinstance(out, (tuple, list)) and out:
                logits = out[0]
            if not isinstance(logits, torch.Tensor) or logits.shape[1] <= logit_idx:
                continue
            ab_step = logits[:, logit_idx, :].float()
            values = _calculate_jsd_from_logits(base_step.unsqueeze(0).expand_as(ab_step), ab_step)
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            group_jsd[start:start + len(batch_groups)] = values.detach().cpu()
            del out, logits, ab_step, values, chunk_inputs, variants
            torch.cuda.empty_cache()

        for group_idx, members in enumerate(groups):
            density = group_jsd[group_idx] / max(int(members.numel()), 1)
            jsd_weights[batch_idx, members] = density

        requested = int(round(refine_ratio * len(groups)))
        requested = min(max(requested, 0), len(groups))
        if requested:
            if refine_strategy == 'top_jsd':
                selected = torch.topk(group_jsd, k=requested, largest=True).indices.tolist()
            else:
                generator = torch.Generator().manual_seed(int(refine_seed) + batch_idx)
                selected = torch.randperm(len(groups), generator=generator)[:requested].tolist()
            refined_group_count[batch_idx] = len(selected)
            refine_tokens = torch.cat([groups[group_idx] for group_idx in selected]).unique(sorted=True)
            exact_probe_calls[batch_idx] = refine_tokens.numel()
            for start in range(0, refine_tokens.numel(), chunk_size):
                chunk_ids = refine_tokens[start:start + chunk_size]
                chunk_ids_device = chunk_ids.to(device=device)
                variants = base_embeds.unsqueeze(0).repeat(chunk_ids.numel(), 1, 1)
                variants[torch.arange(chunk_ids.numel(), device=device), chunk_ids_device, :] = 0
                chunk_inputs = _tree_expand_to_batch(sample_inputs, int(chunk_ids.numel()))
                chunk_inputs['inputs_embeds'] = variants
                try:
                    out = base_model(**chunk_inputs)
                except TypeError:
                    chunk_inputs.pop('return_dict', None)
                    out = base_model(**chunk_inputs)
                logits = getattr(out, 'logits', None)
                if logits is None and isinstance(out, (tuple, list)) and out:
                    logits = out[0]
                if not isinstance(logits, torch.Tensor) or logits.shape[1] <= logit_idx:
                    continue
                ab_step = logits[:, logit_idx, :].float()
                values = _calculate_jsd_from_logits(base_step.unsqueeze(0).expand_as(ab_step), ab_step)
                values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
                jsd_weights[batch_idx, chunk_ids] = values.detach().cpu()
                del out, logits, ab_step, values, chunk_inputs, variants
                torch.cuda.empty_cache()

        del base_out, base_logits, base_step, base_probs, top2, base_inputs, sample_inputs, base_embeds
        torch.cuda.empty_cache()

    return jsd_weights, {
        'baseline_entropy': baseline_entropy,
        'baseline_confidence': baseline_confidence,
        'baseline_margin': baseline_margin,
        'logit_index': torch.tensor(logit_indices, dtype=torch.long),
        'probe2_group_count': group_count,
        'probe2_refined_group_count': refined_group_count,
        'probe2_group_calls': group_probe_calls,
        'probe2_exact_calls': exact_probe_calls,
        'probe2_total_calls': group_probe_calls + exact_probe_calls,
        'probe2_inferred_grid_hw': inferred_grid_hw,
    }


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


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _local_neighbor_cosine(
    inputs_embeds: torch.Tensor,
    vision_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean cosine similarity to the previous/next visual token in sequence order."""
    embeds = inputs_embeds.detach().float().cpu()
    total, seq_len, _ = embeds.shape
    output = torch.zeros((total, seq_len), dtype=torch.float32)
    if vision_mask is None:
        return output
    vision = vision_mask.detach().cpu().bool()
    for sample in range(total):
        indices = torch.nonzero(vision[sample], as_tuple=False).view(-1)
        if indices.numel() < 2:
            continue
        values = F.normalize(embeds[sample].index_select(0, indices), dim=-1)
        adjacent = (values[:-1] * values[1:]).sum(dim=-1)
        sums = torch.zeros(indices.numel(), dtype=torch.float32)
        counts = torch.zeros(indices.numel(), dtype=torch.float32)
        sums[:-1] += adjacent
        counts[:-1] += 1
        sums[1:] += adjacent
        counts[1:] += 1
        output[sample, indices] = sums / counts.clamp_min(1)
    return output


def _save_probe_cache(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    temporary = f'{path}.tmp.{os.getpid()}'
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _importance_from_probe_cache(path: str, valid: torch.Tensor,
                                 vision: Optional[torch.Tensor], alpha: float,
                                 jsd_eps: float, skip_probe1: bool):
    """Rebuild token importance from cached exact probes.

    This is used for paired ablations so the full and Probe-2-only variants share
    exactly the same Probe 2 measurements instead of repeating expensive model
    interventions. It does not change the cost reported for either method.
    """
    started = time.perf_counter()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('method') != 'impquant_probe_cache':
        raise ValueError(f'Invalid ImpQuant probe cache: {path}')
    if payload.get('sample_count') != valid.shape[0] or payload.get('sequence_length') != valid.shape[1]:
        raise ValueError('Probe cache shape does not match current calibration inputs')
    token_features = payload.get('token_features', {})
    targets = payload.get('targets', {})
    cached_valid = token_features.get('valid_mask')
    cached_vision = token_features.get('vision_mask')
    if not isinstance(cached_valid, torch.Tensor) or not torch.equal(cached_valid.bool(), valid.bool()):
        raise ValueError('Probe cache valid-token mask does not match current inputs')
    if vision is None:
        if cached_vision is not None:
            raise ValueError('Probe cache contains visual tokens but current inputs do not')
    elif not isinstance(cached_vision, torch.Tensor) or not torch.equal(cached_vision.bool(), vision.bool()):
        raise ValueError('Probe cache visual-token mask does not match current inputs')

    attention = token_features.get('last_layer_attention')
    jsd = targets.get('probe2_jsd')
    confidence = targets.get('probe1_confidence')
    if not isinstance(attention, torch.Tensor) or attention.shape != valid.shape:
        raise ValueError('Probe cache is missing last-layer attention')
    raw_weights = attention.float().clone()
    redundancy = None
    if vision is not None:
        if not isinstance(jsd, torch.Tensor) or jsd.shape != valid.shape:
            raise ValueError('Probe cache is missing Probe 2 JSD values')
        if skip_probe1:
            confidence = torch.ones_like(jsd)
            redundancy = torch.zeros_like(jsd)
        else:
            if not isinstance(confidence, torch.Tensor) or confidence.shape != valid.shape:
                raise ValueError('Probe cache is missing Probe 1 confidence values')
            confidence = confidence.float()
            redundancy = 1 - confidence
        mask = vision & valid
        raw_weights[mask] = visual_importance(
            confidence[mask], jsd.float()[mask], alpha, 1.0, jsd_eps
        )
    raw_weights *= valid.float()
    if not torch.isfinite(raw_weights).all() or raw_weights.sum() <= 0:
        raise RuntimeError('Invalid token importance reconstructed from probe cache')

    sample_features = payload.get('sample_features', {})
    baseline_stats = {
        key: value for key, value in sample_features.items()
        if key not in {'valid_token_count', 'visual_token_count'}
    }
    source_timings = payload.get('timings_seconds', {})
    timings = {
        'last_layer_attention': 0.0,
        'probe1_one_token_repetition': 0.0,
        'probe2_leave_one_out_jsd': 0.0,
        'cache_load': time.perf_counter() - started,
        'source_total_probes': float(source_timings.get('total_probes', 0.0)),
    }
    return raw_weights, {
        'last_layer_attention': attention.detach().cpu(),
        'probe1_redundancy': redundancy.detach().cpu() if redundancy is not None else None,
        'probe1_confidence': confidence.detach().cpu() if vision is not None else None,
        'probe2_jsd': jsd.detach().cpu() if isinstance(jsd, torch.Tensor) else None,
        'baseline_stats': baseline_stats,
        'timings_seconds': timings,
    }


@torch.no_grad()
def _collect_global_token_importance(base_model, inputs, valid_mask_all,
                                     vision_mask_all, batch_size, tokens_per_step,
                                     text_scale=1.0, vision_scale=1.0, alpha=0.1,
                                     jsd_eps=1e-2, skip_probe1=False,
                                     probe2_mode='exact', probe_group_h=2,
                                     probe_group_w=2, probe_refine_ratio=0.25,
                                     probe_refine_strategy='top_jsd',
                                     probe_refine_seed=2026):
    timings = {}
    _synchronize_cuda()
    started = time.perf_counter()
    text = _collect_text_importance_last_layer_attn(base_model, inputs, valid_mask_all, batch_size)
    _synchronize_cuda()
    timings['last_layer_attention'] = time.perf_counter() - started
    if text is None:
        raise RuntimeError('ImpQuant calibration requires eager attention with attention weights')
    weights = text * text_scale
    redundancy = None
    jsd = None
    baseline_stats = {}
    if probe2_mode == 'uniform':
        weights[~valid_mask_all] = 0
        if vision_mask_all is not None:
            weights[vision_mask_all & valid_mask_all] = 1
    elif vision_mask_all is not None and vision_mask_all.any():
        if skip_probe1:
            # JSD-only ablation. confidence = 1 makes the Probe-1 term
            # -alpha*log(confidence) exactly zero in visual_importance().
            redundancy = torch.zeros_like(weights)
            timings['probe1_repetition'] = 0.0
        else:
            _synchronize_cuda()
            started = time.perf_counter()
            redundancy = _collect_one_token_redundancy(base_model, inputs, valid_mask_all,
                                                      vision_mask_all, tokens_per_step)
            _synchronize_cuda()
            timings['probe1_repetition'] = time.perf_counter() - started
        started = time.perf_counter()
        if probe2_mode == 'exact':
            jsd, baseline_stats = _collect_leave_one_out_jsd(
                base_model, inputs, valid_mask_all, vision_mask_all, tokens_per_step
            )
        elif probe2_mode in ('grouped', 'selective'):
            effective_ratio = 0.0 if probe2_mode == 'grouped' else probe_refine_ratio
            jsd, baseline_stats = _collect_grouped_leave_one_out_jsd(
                base_model, inputs, valid_mask_all, vision_mask_all, tokens_per_step,
                group_h=probe_group_h, group_w=probe_group_w,
                refine_ratio=effective_ratio,
                refine_strategy=probe_refine_strategy,
                refine_seed=probe_refine_seed,
            )
        else:
            raise ValueError(f'Unknown Probe-2 mode: {probe2_mode}')
        _synchronize_cuda()
        timings['probe2_leave_one_out_jsd'] = time.perf_counter() - started
        if redundancy is None or jsd is None:
            raise RuntimeError('ImpQuant visual probes require prepared inputs_embeds')
        mask = vision_mask_all & valid_mask_all
        if (redundancy[mask] < 0).any() or (jsd[mask] < 0).any():
            raise RuntimeError('A visual importance probe failed')
        weights[mask] = visual_importance(1 - redundancy[mask], jsd[mask], alpha, vision_scale, jsd_eps)
    weights *= valid_mask_all.float()
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError('Invalid ImpQuant token importance')
    timings['total_probes'] = sum(timings.values())
    return weights, {
        'last_layer_attention': text.detach().cpu(),
        'probe1_redundancy': redundancy.detach().cpu() if redundancy is not None else None,
        'probe1_confidence': (1 - redundancy).detach().cpu() if redundancy is not None else None,
        'probe2_jsd': jsd.detach().cpu() if jsd is not None else None,
        'baseline_stats': baseline_stats,
        'timings_seconds': timings,
    }


@torch.no_grad()
def run_impquant_weight_importance(model, prompt_inputs, prompt_kwargs, *, w_bit,
                                    a_bit, q_group_size, batch_size=1,
                                    outlier_ratio=0.01, tokens_per_step=1,
                                    text_scale=1.0, vision_scale=1.0, alpha=0.1, jsd_eps=1e-2,
                                    percdamp=0.01, probe_cache_path=None,
                                    probe_source_path=None, probe_only=False, skip_probe1=False,
                                    probe2_mode='exact', probe_group_h=2, probe_group_w=2,
                                    probe_refine_ratio=0.25,
                                    probe_refine_strategy='top_jsd',
                                    probe_refine_seed=2026):
    if probe2_mode == 'uniform' and (not skip_probe1 or probe_only or probe_source_path or probe_cache_path):
        raise ValueError('Uniform mode requires both visual probes disabled and no probe cache')
    base = model.model
    # InternVL's language model accepts prepared multimodal embeddings.
    probe_model = base.language_model if base.__class__.__name__ == 'InternVLChatModel' else base
    layers = get_blocks(base)
    inputs, attention_mask, vision_mask, _ = process_input(prompt_inputs, prompt_kwargs)
    total, seq_len = inputs['inputs_embeds'].shape[:2]
    valid = attention_mask.detach().cpu().bool() if attention_mask is not None else torch.ones(total, seq_len, dtype=torch.bool)
    vision = vision_mask.detach().cpu().bool() if vision_mask is not None else None
    model.to_cuda()
    if probe_source_path is not None:
        raw_weights, probe_components = _importance_from_probe_cache(
            probe_source_path, valid, vision, alpha, jsd_eps, skip_probe1
        )
    else:
        config = probe_model.config
        old_impl = getattr(config, '_attn_implementation', None)
        config._attn_implementation = 'eager'
        replaced = _replace_flash_attention_modules(layers)
        try:
            raw_weights, probe_components = _collect_global_token_importance(
                probe_model, inputs, valid, vision, batch_size, tokens_per_step,
                1.0, 1.0, alpha, jsd_eps, skip_probe1,
                probe2_mode, probe_group_h, probe_group_w,
                probe_refine_ratio, probe_refine_strategy, probe_refine_seed
            )
        finally:
            for layer, attr, old_module in replaced:
                setattr(layer, attr, old_module)
            config._attn_implementation = old_impl

    weights, modality_scales = fixed_modality_weights(raw_weights, valid, vision, text_scale, vision_scale)

    if probe_cache_path is not None:
        input_embeddings = inputs['inputs_embeds'].detach().float().cpu()
        packed_visual_embeddings = (
            [input_embeddings[i][vision[i]].to(torch.float16).contiguous() for i in range(total)]
            if vision is not None else []
        )
        normalized_position = torch.arange(seq_len, dtype=torch.float32)
        normalized_position = normalized_position / max(seq_len - 1, 1)
        normalized_position = normalized_position.unsqueeze(0).expand(total, -1).clone()
        probe_cache = {
            'method': 'impquant_probe_cache',
            'format_version': 1,
            'sample_count': total,
            'sequence_length': seq_len,
            'parameters': {
                'text_scale': text_scale,
                'vision_scale': vision_scale,
                'alpha': alpha,
                'jsd_eps': jsd_eps,
                'tokens_per_step': tokens_per_step,
                'skip_probe1': skip_probe1,
                'probe2_mode': probe2_mode,
                'probe_group_h': probe_group_h,
                'probe_group_w': probe_group_w,
                'probe_refine_ratio': probe_refine_ratio,
                'probe_refine_strategy': probe_refine_strategy,
                'probe_refine_seed': probe_refine_seed,
            },
            'timings_seconds': probe_components['timings_seconds'],
            'sample_features': {
                **probe_components['baseline_stats'],
                'valid_token_count': valid.sum(dim=-1),
                'visual_token_count': vision.sum(dim=-1) if vision is not None else torch.zeros(total, dtype=torch.long),
            },
            'token_features': {
                'valid_mask': valid,
                'vision_mask': vision,
                'last_layer_attention': probe_components['last_layer_attention'],
                'input_activation_norm': input_embeddings.norm(dim=-1),
                'local_neighbor_cosine': _local_neighbor_cosine(input_embeddings, vision),
                'normalized_position': normalized_position,
            },
            'packed_visual_features': {
                'input_embedding_fp16': packed_visual_embeddings,
                'hidden_size': int(input_embeddings.shape[-1]),
            },
            'targets': {
                'probe1_redundancy': probe_components['probe1_redundancy'],
                'probe1_confidence': probe_components['probe1_confidence'],
                'probe2_jsd': probe_components['probe2_jsd'],
                'raw_importance': raw_weights.detach().cpu(),
                'final_importance': weights.detach().cpu(),
            },
            'modality_scales': modality_scales,
        }
        _save_probe_cache(probe_cache_path, probe_cache)
        del input_embeddings, packed_visual_embeddings, normalized_position, probe_cache

    if probe_only:
        return None

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
               'skip_probe1': skip_probe1,
               'probe2_mode': probe2_mode,
               'probe_group_h': probe_group_h, 'probe_group_w': probe_group_w,
               'probe_refine_ratio': probe_refine_ratio,
               'probe_refine_strategy': probe_refine_strategy,
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
            bounded = (probe2_mode == 'uniform' and layer.__class__.__name__ == 'Qwen2_5_VLDecoderLayer'
                       and layer.self_attn.v_proj.weight.shape != layer.self_attn.o_proj.weight.shape)
            scale_inputs = ('self_attn.q_proj', 'mlp.gate_proj', 'mlp.down_proj') if bounded else named_linears
            input_feat = {} if bounded else {name: [] for name in scale_inputs}

            def record_scale_input(tensor, name):
                x = tensor.detach().cpu()
                if not bounded:
                    input_feat[name].append(x)
                else:
                    if name not in input_feat:
                        input_feat[name] = x.new_empty((total, *x.shape[1:]))
                    target = input_feat[name][i:i + 1]
                    assert target.shape == x.shape and target.dtype == x.dtype
                    target.copy_(x)

            handles = [named_linears[name].register_forward_hook(
                lambda module, args, output, key=name: record_scale_input(args[0], key))
                for name in scale_inputs]
            try:
                for i in range(total):
                    layer(captured[i].to(device), **to_device(kwargs_list[i], device))
            finally:
                for handle in handles:
                    handle.remove()
            scale_groups = auto_scale_block_token_weighted(
                layer, kwargs_list, w_bit=w_bit,
                q_config={'zero_point': False, 'q_group_size': q_group_size},
                input_feat=input_feat if bounded else {name: torch.cat(values, dim=0) for name, values in input_feat.items()},
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
