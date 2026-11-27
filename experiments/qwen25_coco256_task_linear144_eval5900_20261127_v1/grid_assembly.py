"""The existing adjacent-neighbor BF16 grid assembly used for both 96- and 144-code states."""
import copy
from common import require
from native_contracts import decode_grid, tensor_digest, validate_actual_repaired_weight

def weight_for(grid, edits):
    """Assemble only original-grid adjacent code edits and audit actual BF16."""
    base = decode_grid(grid); codes = grid['codes'].detach().cpu().clone()
    edits = copy.deepcopy(edits); n = len(edits)
    require(len({(e['row'], e['column']) for e in edits}) == n, 'Distinct coordinates')
    # Algebra-only annotations never alter the native saved-edit schema.
    keys = ('row', 'column', 'direction', 'old_q', 'new_q', 'old_value', 'new_value', 'delta_eff')
    edits = [{key: e[key] for key in keys} for e in edits]
    for e in edits:
        r, c = e['row'], e['column']
        require(int(codes[r, c]) == e['old_q'] and 1 <= e['old_q'] <= 6 and
                e['direction'] in (-1, 1) and e['new_q'] == e['old_q'] + e['direction'] and
                0 <= e['new_q'] <= 7, 'Actual original-grid legal neighbor')
        codes[r, c] = e['new_q']
    value = decode_grid(grid, codes)
    receipt = dict(codes_changed=n, effective_BF16_changes=n,
        baseline_weight_sha256=tensor_digest(base), modified_weight_sha256=tensor_digest(value),
        original_codes_sha256=tensor_digest(grid['codes']), modified_codes_sha256=tensor_digest(codes),
        scales_sha256=tensor_digest(grid['scales']), zeros_sha256=tensor_digest(grid['zeros']),
        original_grid_unchanged=True, edits=edits)
    validate_actual_repaired_weight(grid, value, receipt, expected_count=n)
    return value, receipt
