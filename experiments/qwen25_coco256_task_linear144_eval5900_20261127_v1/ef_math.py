"""CPU-only legal-move empirical-Fisher algebra. No model or GPU calls.

The objective is a LOSS CHANGE: a.T v + gamma/2 * v.T K v.
Gradients and BF16-effective deltas enter Z before the one division by c.
"""
import math


def move_key(move):
    return (int(move.get('module_index', move['module'].split('.')[2])),
            int(move['row']), int(move['column']), int(move['direction']))


def make_pool(grid, mean_normalized_gradient, module, count=64):
    """Top count coordinates by best actual BF16 first-order improvement.

    Both effective interior +/-1 moves are returned, sorted by move_key.
    Mean gradients must already be the four-group scalar normalized mean.
    """
    import torch
    from native_contracts import decode_grid
    q = grid['codes'].detach().cpu()
    assert grid['module'] == module and type(count) is int and count > 0
    base = decode_grid(grid)
    g = mean_normalized_gradient.detach().cpu().to(torch.float64)
    assert g.shape == q.shape and bool(torch.isfinite(g).all())
    scales = grid['scales'].repeat_interleave(128, dim=1)
    zeros = grid['zeros'].repeat_interleave(128, dim=1)
    lower = (scales * (q.float() - 1 - zeros)).to(torch.bfloat16)
    upper = (scales * (q.float() + 1 - zeros)).to(torch.bfloat16)
    dm = lower.to(torch.float64) - base.to(torch.float64)
    dp = upper.to(torch.float64) - base.to(torch.float64)
    eligible = ((q > 0) & (q < 7) & torch.isfinite(base) &
                torch.isfinite(lower) & torch.isfinite(upper) & (dm != 0) & (dp != 0))
    score = torch.maximum(-g * dm, -g * dp).masked_fill(~eligible, -float('inf'))
    assert int(eligible.sum()) >= count
    chosen = torch.argsort(score.reshape(-1), descending=True, stable=True)[:count].tolist()
    coordinates, moves = [], []
    module_index = int(module.split('.')[2])
    for rank, flat in enumerate(chosen):
        row, column = divmod(flat, q.shape[1])
        old_q = int(q[row, column]); old_value = float(base[row, column])
        coordinates.append(dict(module=module, module_index=module_index,
            row=row, column=column, pool_rank=rank,
            first_order_improvement=float(score[row, column])))
        for direction, new_value in [(-1, float(lower[row, column])), (1, float(upper[row, column]))]:
            delta = new_value - old_value
            assert 1 <= old_q <= 6 and 0 <= old_q + direction <= 7 and delta != 0
            moves.append(dict(module=module, module_index=module_index,
                row=row, column=column, direction=direction, old_q=old_q,
                new_q=old_q + direction, old_value=old_value, new_value=new_value,
                delta_eff=delta, pool_rank=rank,
                first_order_improvement=-float(g[row, column]) * delta))
    moves.sort(key=move_key)
    return dict(module=module, coordinates=coordinates, moves=moves,
        eligible_coordinates=int(eligible.sum()), coordinate_count=count,
        move_count=2 * count, dtype='torch.float64',
        coordinate_tie='row,column ascending', move_tie='module_index,row,column,minus before plus')


def projection_stats(Z, groups, constants):
    """Fixed-S normalized a and EF Gram matrices; c is never squared."""
    import torch
    z = torch.as_tensor(Z, dtype=torch.float64, device='cpu')
    assert z.ndim == 2 and len(groups) == z.shape[0] and bool(torch.isfinite(z).all())
    names = list(constants)
    assert names and set(groups) == set(names)
    ag, kg, counts, ranks = {}, {}, {}, {}
    for group in names:
        c = float(constants[group]); assert math.isfinite(c) and c > 0
        indices = [i for i, value in enumerate(groups) if value == group]
        zg = z[indices]; n = len(indices); assert n > 0
        ag[group] = zg.sum(dim=0) / (n * c)
        kg[group] = (zg.T @ zg) / (n * c)
        assert bool(torch.isfinite(ag[group]).all()) and bool(torch.isfinite(kg[group]).all())
        assert torch.equal(kg[group], kg[group].T) and bool((kg[group].diag() >= 0).all())
        counts[group] = n
        ranks[group] = int(torch.linalg.matrix_rank(zg))
        assert ranks[group] <= n
    a = torch.stack([ag[g] for g in names]).mean(dim=0)
    k = torch.stack([kg[g] for g in names]).mean(dim=0)
    return dict(a_group=ag, K_group=kg, a=a, K=k, group_order=names,
        group_counts=counts, group_projected_gradient_ranks=ranks,
        normalization='a_g=sum Z_g/(n*c_g); K_g=Z_g.T@Z_g/(n*c_g); mean over groups')


def objective(a, K, mask, mode='interaction', gamma=1.):
    import torch
    a = torch.as_tensor(a, dtype=torch.float64, device='cpu')
    k = torch.as_tensor(K, dtype=torch.float64, device='cpu')
    v = torch.as_tensor(mask, dtype=torch.float64, device='cpu')
    assert mode in ('linear', 'diagonal', 'interaction') and math.isfinite(gamma) and gamma >= 0
    linear = float(a @ v)
    quadratic = (0. if mode == 'linear' else
                 .5 * gamma * float((k.diag() * v * v).sum()) if mode == 'diagonal' else
                 .5 * gamma * float(v @ k @ v))
    return dict(linear_loss_change=linear, quadratic_loss_change=quadratic,
                objective_loss_change=linear + quadratic)


def solve(a, K, moves, mode='linear', budget=96, per_module_cap=32, gamma=1.):
    """Greedy exact budget from an empty state with deterministic exact ties.

    A move's coordinate is excluded after either direction is chosen. Negative
    improvements remain eligible: the same exact budget is forced in all modes.
    """
    import torch
    a = torch.as_tensor(a, dtype=torch.float64, device='cpu')
    k = torch.as_tensor(K, dtype=torch.float64, device='cpu')
    n = len(moves)
    assert a.shape == (n,) and k.shape == (n, n) and bool(torch.isfinite(a).all())
    assert bool(torch.isfinite(k).all()) and torch.equal(k, k.T)
    assert type(budget) is int and budget > 0 and type(per_module_cap) is int and per_module_cap > 0
    assert mode in ('linear', 'diagonal', 'interaction') and math.isfinite(gamma) and gamma >= 0
    assert len({move_key(m) for m in moves}) == n
    order = sorted(range(n), key=lambda i: move_key(moves[i]))
    counts = {m['module']: 0 for m in moves}
    selected, trace, used = [], [], set()
    mask = torch.zeros(n, dtype=torch.float64)
    cross = torch.zeros(n, dtype=torch.float64)
    for step in range(budget):
        best, best_delta = None, float('inf')
        for i in order:
            m = moves[i]; coordinate = (m['module'], m['row'], m['column'])
            if coordinate in used or counts[m['module']] >= per_module_cap:
                continue
            delta = float(a[i])
            if mode != 'linear':
                delta += .5 * gamma * float(k[i, i])
            if mode == 'interaction':
                delta += gamma * float(cross[i])
            # Iteration is in the immutable exact-tie order. No tolerance ties.
            if delta < best_delta:
                best, best_delta = i, delta
        assert best is not None and math.isfinite(best_delta), 'Insufficient eligible moves for exact budget'
        m = moves[best]
        selected.append(best); mask[best] = 1.
        used.add((m['module'], m['row'], m['column'])); counts[m['module']] += 1
        cross += k[:, best]
        trace.append(dict(step=step, move_index=best, module=m['module'], row=m['row'],
            column=m['column'], direction=m['direction'], marginal_loss_change=best_delta,
            marginal_improvement=-best_delta, negative_marginal_improvement=best_delta > 0))
    result = objective(a, k, mask, mode, gamma)
    assert math.isclose(math.fsum(t['marginal_loss_change'] for t in trace),
                        result['objective_loss_change'], rel_tol=1e-11, abs_tol=1e-12)
    return dict(result, mode=mode, gamma=0. if mode == 'linear' else gamma,
        initialization='empty selection, original quantized baseline',
        selected_indices=selected, trace=trace, counts=counts, exact_budget=budget,
        negative_marginal_count=sum(t['negative_marginal_improvement'] for t in trace),
        forced_exact_budget=True, no_matrix_inversion=True, no_damping=True,
        tie_rule='minimum loss delta; exact ties module_index,row,column,minus before plus')
