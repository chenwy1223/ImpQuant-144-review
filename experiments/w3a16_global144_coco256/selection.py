"""Task-supervised global first-order code repair on the COCO-256 W3A16 cache."""

import math

import torch


LAYERS = (7, 11, 15, 19, 23, 27)
MODULES = tuple(f"model.layers.{layer}.self_attn.v_proj" for layer in LAYERS)
GROUPS = ("scienceqa", "aokvqa", "textvqa", "ocr")
EXAMPLES_PER_GROUP = 16
GROUP_SIZE = 128
CANDIDATES_PER_MODULE = 128
MODULE_CAP = 96
BUDGET = 144


def select_global_144(grids, load_gradient, supervision):
    """Rank adjacent code moves using aligned task gradients and answer losses.

    load_gradient(module, index) returns a native BF16 weight-gradient tensor.
    Each supervision row provides training_group and answer_nll in sample order.
    """
    if set(grids) != set(MODULES) or len(supervision) != len(GROUPS) * EXAMPLES_PER_GROUP:
        raise ValueError("Expected six native v_proj grids and 64 supervision rows")
    group_indices = {
        group: [i for i, row in enumerate(supervision) if row["training_group"] == group]
        for group in GROUPS
    }
    if any(len(indices) != EXAMPLES_PER_GROUP for indices in group_indices.values()):
        raise ValueError("Each supervision group must contain 16 rows")
    losses = [float(row["answer_nll"]) for row in supervision]
    if not all(math.isfinite(value) and value >= 0 for value in losses):
        raise ValueError("Answer losses must be finite and nonnegative")
    norms = {
        group: float(torch.tensor(
            max(math.fsum(losses[i] for i in indices) / len(indices), 1e-6),
            dtype=torch.float32, device="cpu",
        ))
        for group, indices in group_indices.items()
    }

    moves, projections = [], []
    for layer, module in zip(LAYERS, MODULES):
        codes, scales, zeros = _grid_tensors(grids[module], module)
        sums = {group: codes.new_zeros(codes.shape, dtype=torch.float64) for group in GROUPS}
        gradients = []
        for index, row in enumerate(supervision):
            gradient = load_gradient(module, index).detach().cpu()
            if gradient.dtype != torch.bfloat16 or gradient.shape != codes.shape:
                raise ValueError(f"Expected the native BF16 weight gradient for {module}")
            gradients.append(gradient)
            sums[row["training_group"]].add_(gradient.to(torch.float64))

        mean = torch.stack([
            sums[group] / (EXAMPLES_PER_GROUP * norms[group]) for group in GROUPS
        ]).mean(dim=0)
        block = _shortlist(module, layer, codes, scales, zeros, mean)
        rows = torch.tensor([move["row"] for move in block], device="cpu")
        columns = torch.tensor([move["column"] for move in block], device="cpu")
        deltas = torch.tensor(
            [move["delta_eff"] for move in block], dtype=torch.float64, device="cpu"
        )
        projections.append(torch.stack([g[rows, columns].to(torch.float64) * deltas for g in gradients]))
        moves.extend(block)
        del gradients

    # Compute the global scores from per-sample effective movements in FP64.
    projected = torch.cat(projections, dim=1)
    loss_changes = torch.stack([
        projected[indices].sum(dim=0) / (len(indices) * norms[group])
        for group, indices in group_indices.items()
    ]).mean(dim=0).tolist()
    ranked = sorted(range(len(moves)), key=lambda i: (loss_changes[i], _move_key(moves[i])))

    selected, used = [], set()
    counts = dict.fromkeys(MODULES, 0)
    for index in ranked:
        move = moves[index]
        module = move["module"]
        coordinate = (module, move["row"], move["column"])
        if coordinate in used or counts[module] == MODULE_CAP:
            continue
        selected.append(move)
        used.add(coordinate)
        counts[module] += 1
        if len(selected) == BUDGET:
            return selected
    raise ValueError("Insufficient eligible coordinates for the 144-code budget")


def repair_quantized_state(state, grids, edits):
    """Return a new ImpQuant cache with the selected BF16 weight replacements."""
    config = tuple(state[key] for key in ("method", "w_bit", "a_bit", "q_group_size"))
    if config != ("impquant", 3, 16, GROUP_SIZE):
        raise ValueError("Expected a W3A16 ImpQuant cache with group size 128")
    coordinates = {(edit["module"], edit["row"], edit["column"]) for edit in edits}
    if len(edits) != BUDGET or len(coordinates) != BUDGET:
        raise ValueError("Expected 144 distinct code edits")

    by_module = {name: [] for name in MODULES}
    for edit in edits:
        by_module[edit["module"]].append(edit)
    weights = [layer.copy() for layer in state["weights"]]
    for layer, name in zip(LAYERS, MODULES):
        codes, scales, zeros = _grid_tensors(grids[name], name)
        baseline = _decode(codes, scales, zeros)
        cached = weights[layer]["self_attn.v_proj"].detach().cpu().contiguous()
        if (cached.dtype != baseline.dtype or cached.shape != baseline.shape
                or not torch.equal(cached.view(torch.uint8), baseline.view(torch.uint8))):
            raise ValueError(f"The native grid is from a different cache: {name}")

        changes = by_module[name]
        if not changes:
            continue
        if len(changes) > MODULE_CAP:
            raise ValueError(f"Per-module code budget exceeded for {name}")
        patched_codes = codes.clone()
        for edit in changes:
            row, column = edit["row"], edit["column"]
            if not (0 <= row < codes.shape[0] and 0 <= column < codes.shape[1]):
                raise ValueError(f"Code coordinate out of range for {name}")
            old_q = int(codes[row, column])
            direction = edit["direction"]
            if (old_q != edit["old_q"] or not 1 <= old_q <= 6 or direction not in (-1, 1)
                    or edit["new_q"] != old_q + direction):
                raise ValueError(f"Expected a legal one-step code edit for {name}")
            patched_codes[row, column] = edit["new_q"]

        patched = _decode(patched_codes, scales, zeros)
        for edit in changes:
            row, column = edit["row"], edit["column"]
            step = float(patched[row, column]) - float(baseline[row, column])
            if step == 0 or not math.isfinite(step) or step != edit["delta_eff"]:
                raise ValueError(f"BF16 step differs from the ranked move for {name}")
        weights[layer]["self_attn.v_proj"] = patched
    return {**state, "weights": weights}


def _grid_tensors(grid, module):
    codes, scales, zeros = (grid[key].detach().cpu() for key in ("codes", "scales", "zeros"))
    if (grid["module"] != module or grid["group_size"] != GROUP_SIZE
            or codes.ndim != 2 or codes.numel() == 0 or codes.shape[1] % GROUP_SIZE
            or codes.dtype != torch.uint8
            or int(codes.max()) > 7):
        raise ValueError(f"Expected a native 3-bit group-128 integer grid for {module}")
    shape = (codes.shape[0], codes.shape[1] // GROUP_SIZE)
    if (scales.shape != shape or zeros.shape != shape or scales.dtype != torch.float32
            or zeros.dtype != torch.float32 or not bool(torch.isfinite(scales).all())
            or not bool((scales > 0).all()) or not bool(torch.isfinite(zeros).all())):
        raise ValueError(f"Invalid native scale or zero-point for {module}")
    return codes, scales, zeros


def _decode(codes, scales, zeros):
    grouped = codes.float().reshape(codes.shape[0], -1, GROUP_SIZE)
    value = scales.unsqueeze(-1) * (grouped - zeros.unsqueeze(-1))
    return value.reshape(codes.shape).to(torch.bfloat16)


def _shortlist(module, layer, codes, scales, zeros, mean):
    if not bool(torch.isfinite(mean).all()):
        raise ValueError(f"Nonfinite normalized gradient for {module}")
    base = _decode(codes, scales, zeros)
    scale = scales.repeat_interleave(GROUP_SIZE, dim=1)
    zero = zeros.repeat_interleave(GROUP_SIZE, dim=1)
    lower = (scale * (codes.float() - 1 - zero)).to(torch.bfloat16)
    upper = (scale * (codes.float() + 1 - zero)).to(torch.bfloat16)
    down = lower.to(torch.float64) - base.to(torch.float64)
    up = upper.to(torch.float64) - base.to(torch.float64)
    legal = ((codes > 0) & (codes < 7) & torch.isfinite(base)
             & torch.isfinite(lower) & torch.isfinite(upper) & (down != 0) & (up != 0))
    if int(legal.sum()) < CANDIDATES_PER_MODULE:
        raise ValueError(f"Insufficient eligible coordinates for {module}")

    best = torch.maximum(-mean * down, -mean * up).masked_fill(~legal, -math.inf)
    order = torch.argsort(best.reshape(-1), descending=True, stable=True)
    moves = []
    for flat in order[:CANDIDATES_PER_MODULE].tolist():
        row, column = divmod(flat, codes.shape[1])
        old_q = int(codes[row, column])
        for direction, delta in ((-1, down), (1, up)):
            moves.append(dict(
                module=module, module_index=layer, row=row, column=column,
                direction=direction, old_q=old_q, new_q=old_q + direction,
                delta_eff=float(delta[row, column]),
            ))
    return sorted(moves, key=_move_key)


def _move_key(move):
    return move["module_index"], move["row"], move["column"], move["direction"]
