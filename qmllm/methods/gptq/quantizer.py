import math
import warnings
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from qmllm.utils.calibration import slice_batch, to_device


def get_blocks(model: nn.Module):
    name = model.__class__.__name__
    if name == "LlamaForCausalLM":
        return model.model.layers
    if name == "LlavaLlamaForCausalLM":
        return model.model.layers
    if name == "LlavaQwenForCausalLM":
        return model.model.layers
    if name in ["Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"]:
        return model.model.layers
    if name in ["Qwen2VLModel", "Qwen2_5_VLModel"]:
        return model.layers
    if name == "InternLM2ForCausalLM":
        return model.model.layers
    if name == "InternVLChatModel":
        return model.language_model.model.layers
    raise NotImplementedError(f"GPTQ get_blocks unsupported for {name}")


def find_layers(module: nn.Module, layers=(nn.Linear,), name: str = "") -> Dict[str, nn.Module]:
    if isinstance(module, layers):
        return {name: module}
    res: Dict[str, nn.Module] = {}
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name else name1
        res.update(find_layers(child, layers=layers, name=child_name))
    return res


def quantize(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    if int(maxq.item()) < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq.item())
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape: int = 1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits: int, perchannel: bool = True, sym: bool = True):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym

    def find_params(self, x: torch.Tensor, weight: bool = False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            neg = xmin < 0
            if torch.any(neg):
                xmin[neg] = -xmax[neg]

        both_zero = (xmin == 0) & (xmax == 0)
        xmin[both_zero] = -1
        xmax[both_zero] = +1

        if int(self.maxq.item()) < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq.item() + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            repeat = shape[0] if weight else (shape[1] if len(shape) != 3 else shape[2])
            self.scale = self.scale.repeat(repeat)
            self.zero = self.zero.repeat(repeat)

        if weight:
            reshape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(reshape)
            self.zero = self.zero.reshape(reshape)
            return

        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def ready(self) -> bool:
        return bool(torch.all(self.scale != 0))


def _factor_hessian(hessian: torch.Tensor, percdamp: float):
    """Factor a damped Hessian, with bounded precision/damping retries."""
    if not math.isfinite(percdamp) or percdamp < 0:
        raise ValueError('GPTQ damping must be finite and nonnegative')
    if not torch.isfinite(hessian).all():
        raise ValueError('GPTQ Hessian contains NaN or Inf')
    if (hessian.diagonal() < 0).any():
        raise ValueError('GPTQ Hessian has a negative diagonal')
    damped = hessian.clone()
    damped.diagonal().add_(percdamp * hessian.diagonal().mean())
    try:
        return torch.linalg.cholesky(damped), percdamp
    except torch.linalg.LinAlgError:
        pass
    del damped
    warnings.warn('GPTQ Hessian Cholesky failed; retrying in float64.',
                  RuntimeWarning, stacklevel=2)
    symmetric = hessian.double()
    symmetric = (symmetric + symmetric.T) * 0.5
    mean_diagonal = symmetric.diagonal().mean()
    # Keep the configured damping on the first float64 attempt. Only increase
    # it if needed, up to 10x the original value and at most 0.1.
    candidates = [percdamp]
    if 0 < percdamp < 0.1:
        for multiplier in (2, 5, 10):
            value = min(percdamp * multiplier, 0.1)
            if value > candidates[-1]:
                candidates.append(value)
    for value in candidates:
        damped = symmetric.clone()
        damped.diagonal().add_(value * mean_diagonal)
        try:
            factor = torch.linalg.cholesky(damped)
        except torch.linalg.LinAlgError:
            del damped
            continue
        if value != percdamp:
            warnings.warn(f'GPTQ Hessian required percdamp={value:g} '
                          f'(requested {percdamp:g}).', RuntimeWarning, stacklevel=2)
        return factor, value
    raise torch.linalg.LinAlgError(
        f'GPTQ Hessian is not positive-definite after float64 retries '
        f'up to percdamp={candidates[-1]:g}; check calibration activations and scales.')


def _cholesky_of_inverse(factor: torch.Tensor) -> torch.Tensor:
    """Return the upper Cholesky factor of (factor @ factor.T)^-1."""
    inverse = torch.cholesky_inverse(factor)
    try:
        return torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError:
        if factor.dtype == torch.float64:
            raise
    # Recompute the inverse from the successful Hessian factorization. Casting
    # the already rounded inverse to float64 would preserve its SPD error.
    del inverse
    warnings.warn(
        "GPTQ inverse-Hessian Cholesky failed in float32; "
        "retrying inversion and Cholesky in float64.",
        RuntimeWarning,
        stacklevel=2,
    )
    inverse = torch.cholesky_inverse(factor.double())
    return torch.linalg.cholesky(inverse, upper=True).to(factor.dtype)


class GPTQ:
    def __init__(self, layer: nn.Linear, dev: Optional[torch.device] = None):
        self.layer = layer
        self.dev = dev if dev is not None else self.layer.weight.device
        W = layer.weight.data.clone().to(self.dev)
        self.rows, self.columns = W.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def add_batch(self, inp: torch.Tensor):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.to(self.dev).t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize: int = 128, percdamp: float = 0.01, groupsize: int = -1, actorder: bool = False):
        orig_device = self.layer.weight.data.device
        W = self.layer.weight.data.clone().to(self.dev).float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        H, self.effective_percdamp = _factor_hessian(H, percdamp)
        Hinv = _cholesky_of_inverse(H).to(W.dtype)
        del H

        Q = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)], weight=True)

                q = quantize(w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(orig_device).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _unwrap_hidden(output: Any) -> torch.Tensor:
    out = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Unexpected layer output type: {type(out)}")
    if out.dim() == 3 and out.shape[0] == 1:
        return out.squeeze(0)
    return out


def _iter_calib_batches(
    prompt_inputs: Optional[Dict[str, Any]],
    prompt_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    if isinstance(prompt_inputs, dict):
        merged.update(prompt_inputs)
    if isinstance(prompt_kwargs, dict):
        merged.update(prompt_kwargs)

    for k in ["labels", "vision_mask", "caption_mask"]:
        merged.pop(k, None)

    num_samples = None
    for v in merged.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            num_samples = int(v.shape[0])
            break
    if num_samples is None:
        num_samples = 1
    take = num_samples if max_samples is None else min(num_samples, int(max_samples))

    for i in range(take):
        batch: Dict[str, Any] = {"use_cache": False}
        for k, v in merged.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                batch[k] = to_device(slice_batch(v, i, i + 1, num_samples, k), device)
            else:
                batch[k] = v
        yield batch


def _layer_uses_flash_attention(layer: nn.Module) -> bool:
    """
    Heuristic: many custom models (e.g., InternLM2/InternVL2) implement a FlashAttention path
    via an internal `_flash_attention_forward` helper.
    """
    for m in layer.modules():
        if hasattr(m, "_flash_attention_forward"):
            return True
    return False


def _sanitize_attention_mask_for_flash(
    attention_mask: Any, batch_size: int, seq_len: int, device: torch.device
) -> Optional[torch.Tensor]:
    """
    FlashAttention varlen implementations expect a *padding mask* of shape (B, S),
    where 0 indicates padding and 1 indicates non-padding. Some models provide a
    causal/additive mask with shape (B, 1, S, S) or even (S, S).
    """
    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        return attention_mask

    m = attention_mask

    # Already a padding mask (B, S).
    if m.dim() == 2 and m.shape[0] == batch_size and m.shape[1] == seq_len:
        return m.to(device=device)

    def _as_padding_mask_from_row(row: torch.Tensor) -> torch.Tensor:
        """
        Convert an attention row (shape [..., S]) to a padding mask (same shape) where
        1/True indicates non-padding.
        """
        if row.dtype == torch.bool:
            # Heuristic: for a causal mask, key 0 is always valid. If row[..., 0] is False,
            # the mask likely follows the "True = masked" convention, so we invert.
            flag = row[..., 0].unsqueeze(-1)
            return torch.where(flag, row, ~row)
        if row.dtype.is_floating_point:
            # Additive mask convention: 0 for valid, -inf for masked.
            return torch.isfinite(row)
        # Integer mask convention: 1 for valid, 0 for padding/masked.
        return row != 0

    # Common bad case: (S, S) causal/additive mask without batch dim.
    if m.dim() == 2 and m.shape[0] == m.shape[1] == seq_len and batch_size == 1:
        key_valid = _as_padding_mask_from_row(m[-1])
        return key_valid.unsqueeze(0).to(device=device)

    # Convert masks with explicit query/key dims into a key padding mask using the last query row.
    # This avoids O(S^2) reductions on large (S, S) masks.
    if m.dim() in (3, 4) and m.shape[-1] == seq_len:
        if m.shape[-2] == seq_len:
            row = m[..., -1, :]
        elif m.shape[-2] == 1:
            # e.g. (B, 1, 1, S) or (B, 1, S)
            row = m[..., 0, :]
        else:
            return None

        key_valid = _as_padding_mask_from_row(row)

        # Remove singleton dims like the attention-head dim.
        while key_valid.dim() > 2:
            key_valid = key_valid.squeeze(1)
        if key_valid.dim() == 1:
            key_valid = key_valid.unsqueeze(0)

        if key_valid.shape[0] != batch_size:
            return None
        if key_valid.shape[1] != seq_len:
            key_valid = key_valid[:, :seq_len]
        return key_valid.to(device=device)

    # Unknown format: drop it (assumes no padding).
    return None


@torch.no_grad()
def run_gptq(
    model: nn.Module,
    prompt_inputs: Optional[Dict[str, Any]],
    prompt_kwargs: Optional[Dict[str, Any]],
    blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int = -1,
    actorder: bool = False,
    w_bit: int = 4,
    max_samples: Optional[int] = None,
    stat_device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Layer-wise GPTQ (original repo style): capture inputs to the first decoder layer,
    then quantize Linear modules one decoder layer at a time to avoid OOM.

    Returns a serializable dict containing quantized linear weights/biases (on CPU)
    so `main.py --pseudo_quant` can reload without calibration inputs.
    """
    if prompt_inputs is None and prompt_kwargs is None:
        raise ValueError("GPTQ calibration requires prompt_inputs/prompt_kwargs (run in main_quant.py).")

    model.eval()
    layers = get_blocks(model)
    if len(layers) == 0:
        raise ValueError("GPTQ: no decoder layers found.")

    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    batches = list(_iter_calib_batches(prompt_inputs, prompt_kwargs, device=device, max_samples=max_samples))
    if not batches:
        raise ValueError("GPTQ: empty calibration batches.")

    # Capture hidden-states input to the first decoder layer and its kwargs.
    use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        # fallback: infer from first linear in the first layer
        first_linear = next((m for m in layers[0].modules() if isinstance(m, nn.Linear)), None)
        if first_linear is None:
            raise ValueError("GPTQ: cannot infer hidden_size.")
        hidden_size = int(first_linear.in_features)

    # infer sequence length from inputs
    seq_len = None
    for k in ["inputs_embeds", "input_ids", "attention_mask"]:
        v = batches[0].get(k, None)
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            seq_len = int(v.shape[1])
            break
    if seq_len is None:
        raise ValueError("GPTQ: cannot infer sequence length from calibration inputs.")

    nsamples = len(batches)
    inps = torch.zeros((nsamples, seq_len, hidden_size), dtype=dtype, device=device)
    layer_kwargs: list[Dict[str, Any]] = [{} for _ in range(nsamples)]
    cache: Dict[str, Any] = {"i": 0}

    class CaptureComplete(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp: torch.Tensor, **kwargs):
            idx = cache["i"]
            inps[idx] = _unwrap_hidden(inp)
            layer_kwargs[idx] = {k: kwargs[k] for k in ["attention_mask", "position_ids", "cache_position", "position_embeddings"] if k in kwargs}
            cache["i"] = idx + 1
            raise CaptureComplete

    layers[0] = Catcher(layers[0])
    first = layers[0].module
    try:
        for batch in batches:
            try:
                model(**batch)
            except CaptureComplete:
                pass
    finally:
        layers[0] = first
    if cache["i"] != nsamples:
        raise RuntimeError("GPTQ did not capture every calibration sample")

    outs = torch.zeros_like(inps)
    module_to_name = {id(m): n for n, m in model.named_modules()}

    stat_dev = stat_device if stat_device is not None else device
    print(f"[gptq] calib nsamples={nsamples} seq_len={seq_len} hidden={hidden_size} stat_device={stat_dev}")

    layer_kwargs_flash: list[Dict[str, Any]] = []
    for kw in layer_kwargs:
        if "attention_mask" not in kw:
            layer_kwargs_flash.append(kw)
            continue
        safe_mask = _sanitize_attention_mask_for_flash(
            kw.get("attention_mask", None),
            batch_size=1,
            seq_len=seq_len,
            device=device,
        )
        if safe_mask is None:
            layer_kwargs_flash.append({k: v for k, v in kw.items() if k != "attention_mask"})
        else:
            kw2 = dict(kw)
            kw2["attention_mask"] = safe_mask
            layer_kwargs_flash.append(kw2)

    quant_state: Dict[str, torch.Tensor] = {}
    pbar = tqdm(layers, desc="GPTQ (layer-wise)")
    for i, layer in enumerate(pbar):
        layer.eval()
        use_flash_attn = _layer_uses_flash_attention(layer)
        kw_list = layer_kwargs_flash if use_flash_attn else layer_kwargs
        full = find_layers(layer, layers=(nn.Linear,))

        gptq: Dict[str, GPTQ] = {}
        handles = []

        for name, mod in full.items():
            if not isinstance(mod, nn.Linear):
                continue
            g = GPTQ(mod, dev=stat_device)
            g.quantizer.configure(w_bit, perchannel=True, sym=True)
            gptq[name] = g

            def _hook(module, inp, _out, name=name):
                x = inp[0].detach()
                mask = batches[j].get("attention_mask")
                if isinstance(mask, torch.Tensor) and x.shape[:2] == mask.shape:
                    x = x[mask.bool()]
                gptq[name].add_batch(x)

            handles.append(mod.register_forward_hook(_hook))

        # collect Hessian stats
        tick = max(1, nsamples // 20)
        for j in range(nsamples):
            if j % tick == 0:
                pbar.set_postfix_str(f"layer {i} stats {j}/{nsamples}")
            kw = kw_list[j]
            _ = _unwrap_hidden(layer(inps[j].unsqueeze(0), **kw))

        for h in handles:
            h.remove()

        # quantize weights for this layer
        for name, g in gptq.items():
            g.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder)
            g.free()

        # forward again (with quantized weights) to produce inputs for the next layer
        for j in range(nsamples):
            if j % tick == 0:
                pbar.set_postfix_str(f"layer {i} fwd {j}/{nsamples}")
            kw = kw_list[j]
            outs[j] = _unwrap_hidden(layer(inps[j].unsqueeze(0), **kw))

        # export quantized weights for caching
        layer_prefix = module_to_name.get(id(layer), None)
        if layer_prefix:
            for name, mod in full.items():
                full_name = f"{layer_prefix}.{name}" if name else layer_prefix
                quant_state[f"{full_name}.weight"] = mod.weight.detach().cpu()
                if mod.bias is not None:
                    quant_state[f"{full_name}.bias"] = mod.bias.detach().cpu()

        inps, outs = outs, inps
        pbar.set_postfix_str("")

    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = use_cache

    return {
        "config": {
            "w_bit": int(w_bit),
            "blocksize": int(blocksize),
            "percdamp": float(percdamp),
            "groupsize": int(groupsize),
            "actorder": bool(actorder),
        },
        "state_dict": quant_state,
    }


@torch.no_grad()
def apply_gptq_state(model: nn.Module, gptq_state: Dict[str, Any]) -> None:
    state_dict = gptq_state.get("state_dict", None)
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid GPTQ state: missing 'state_dict'.")
    cur = model.state_dict()
    for k, v in state_dict.items():
        if k not in cur or cur[k].shape != v.shape:
            raise ValueError(f"GPTQ cache does not match model tensor {k}")
        tgt = cur[k]
        if isinstance(v, torch.Tensor):
            tgt.copy_(v.to(device=tgt.device, dtype=tgt.dtype))
