"""Core selection and grid replay for the evaluated 8-image W3A16 patch."""

import hashlib
import math


BUDGET = 144
MODULE_CAP = 96
GROUP_SIZE = 128


def select_global_144(loss_changes, moves):
    """Return indices from the frozen candidate pool in exact selection order."""
    scores = [float(value) for value in loss_changes]
    tie_keys = [
        (int(move.get("module_index", move["module"].split(".")[2])),
         move["row"], move["column"], move["direction"])
        for move in moves
    ]
    if len(scores) != len(moves) or len(set(tie_keys)) != len(moves):
        raise ValueError("Expected one first-order score and unique key per move")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("First-order scores must be finite")

    # The first-order loss change is fixed. A single sort reproduces the
    # archived greedy order because chosen coordinates only remove candidates.
    ranked = sorted(range(len(moves)), key=lambda index: (scores[index], tie_keys[index]))
    selected = []
    used = set()
    counts = {}
    for index in ranked:
        move = moves[index]
        module = move["module"]
        coordinate = (module, move["row"], move["column"])
        count = counts.get(module, 0)
        if coordinate in used or count == MODULE_CAP:
            continue
        selected.append(index)
        used.add(coordinate)
        counts[module] = count + 1
        if len(selected) == BUDGET:
            return selected
    raise ValueError("Insufficient eligible moves for 144 edits")


def tensor_digest(value):
    import torch

    tensor = value.detach().cpu().contiguous()
    header = str((str(tensor.dtype), tuple(tensor.shape))).encode()
    data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + data).hexdigest()


def _decode(codes, scales, zeros):
    import torch

    grouped_codes = codes.float().reshape(codes.shape[0], -1, GROUP_SIZE)
    weights = scales.unsqueeze(-1) * (grouped_codes - zeros.unsqueeze(-1))
    return weights.reshape(codes.shape).to(torch.bfloat16)


def build_replacements(grids, patch):
    """Assemble and verify the four evaluated BF16 tensors on the CPU."""
    by_module = {name: [] for name in patch["tensor_receipts"]}
    coordinates = {(edit["module"], edit["row"], edit["column"]) for edit in patch["edits"]}
    if len(coordinates) != len(patch["edits"]) or len(coordinates) != BUDGET:
        raise ValueError("Expected 144 distinct edit coordinates")
    for edit in patch["edits"]:
        by_module[edit["module"]].append(edit)

    replacements = {}
    for name, edits in by_module.items():
        receipt = patch["tensor_receipts"][name]
        grid = grids[name]
        codes, scales, zeros = (grid[key].detach().cpu() for key in ("codes", "scales", "zeros"))
        if (tensor_digest(codes) != receipt["original_codes_sha256"]
                or tensor_digest(scales) != receipt["scales_sha256"]
                or tensor_digest(zeros) != receipt["zeros_sha256"]):
            raise ValueError(f"Original grid fingerprint differs for {name}")

        original = _decode(codes, scales, zeros)
        patched_codes = codes.clone()
        for edit in edits:
            row, column = edit["row"], edit["column"]
            old_code = int(codes[row, column])
            direction = edit["direction"]
            new_code = edit["new_q"]
            if (old_code != edit["old_q"] or not 1 <= old_code <= 6
                    or direction not in (-1, 1) or new_code != old_code + direction):
                raise ValueError(f"Invalid neighboring code for {name}[{row}, {column}]")
            patched_codes[row, column] = new_code

        replacement = _decode(patched_codes, scales, zeros)
        if (tensor_digest(original) != receipt["baseline_weight_sha256"]
                or tensor_digest(patched_codes) != receipt["modified_codes_sha256"]
                or tensor_digest(replacement) != receipt["modified_weight_sha256"]
                or int((replacement != original).sum()) != len(edits)):
            raise ValueError(f"Reconstructed tensor differs from the evaluated patch for {name}")
        replacements[name] = replacement
    return replacements
