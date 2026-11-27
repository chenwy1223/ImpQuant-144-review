"""Global first-order ranking in the six new native integer grids."""
from collections import Counter
from copy import deepcopy
import math

from common import *
from ef_math import make_pool, move_key, solve
from grid_assembly import weight_for
from native_contracts import decode_grid, state_root, tensor_digest


def choose(records, notes, baseline):
    import torch

    source = read(OUT/"source_snapshot.json")
    rows = read(OUT/"supervision_manifest.json")["rows"]
    groups = {g:[i for i,r in enumerate(rows) if r["training_group"] == g] for g in GROUPS}
    assert all(len(indices) == 16 for indices in groups.values())
    constants = {}
    for name, indices in groups.items():
        nll = math.fsum(float(notes[i]["answer_nll"]) for i in indices)/len(indices)
        constants[name] = float(torch.tensor(max(nll,1e-6),dtype=torch.float32))
        assert math.isfinite(constants[name]) and constants[name]>0
    records = {(r["module"],r["manifest_index"]):r for r in records}
    assert len(records) == len(MODULES)*SUPERVISION_N

    means, pools, grids, projected = {}, {}, {}, []
    moves = []
    for module in MODULES:
        g = source["grids"][module]
        check(g)
        grid = grids[module] = torch.load(g["path"], map_location="cpu", weights_only=True)
        base = decode_grid(grid)
        assert tensor_digest(base) == baseline[module+".weight"]["sha256"]
        gradients = []
        sums = {k:torch.zeros_like(base, dtype=torch.float64) for k in GROUPS}
        for i, row in enumerate(rows):
            entry = records[module,i]
            check(entry["artifact"])
            payload = torch.load(entry["artifact"]["path"], map_location="cpu", weights_only=True)
            assert all(payload[k] == entry[k] for k in ("module","manifest_index","training_group","id"))
            tensor = payload["native_gradient_BF16"]
            assert tensor.dtype == torch.bfloat16 and tensor.shape == base.shape
            assert tensor_digest(tensor) == entry["gradient_sha256"]
            gradients.append(tensor)
            sums[row["training_group"]].add_(tensor.to(torch.float64))
        means[module] = torch.stack([sums[k]/(len(groups[k])*constants[k]) for k in GROUPS]).mean(dim=0)
        pool = make_pool(grid, means[module], module, count=POOL_PER_MODULE)
        block = pool["moves"]
        pools[module] = {k:pool[k] for k in ("eligible_coordinates","coordinates")}
        moves.extend(block)
        rr = torch.tensor([item["row"] for item in block])
        cc = torch.tensor([item["column"] for item in block])
        delta = torch.tensor([item["delta_eff"] for item in block], dtype=torch.float64)
        projected.append(torch.stack([t[rr,cc].to(torch.float64)*delta for t in gradients]))
        del gradients, sums
    n = len(moves)
    assert n == 2*len(MODULES)*POOL_PER_MODULE and moves == sorted(moves,key=move_key)
    z = torch.cat(projected, dim=1)
    by_group = {g:z[groups[g]].sum(dim=0)/(len(groups[g])*constants[g]) for g in GROUPS}
    a = torch.stack([by_group[g] for g in GROUPS]).mean(dim=0)
    assert torch.allclose(-a, torch.tensor([m["first_order_improvement"] for m in moves],dtype=torch.float64),
                          rtol=2e-12,atol=2e-19)
    picked = solve(a, torch.zeros((n,n),dtype=torch.float64), moves,
                   mode="linear",budget=EDITS,per_module_cap=MODULE_CAP)
    edits = [moves[i] for i in picked["selected_indices"]]
    assert len({(m["module"],m["row"],m["column"]) for m in edits}) == EDITS
    by_module = Counter(m["module"] for m in edits)
    counts = {m:by_module[m] for m in MODULES}
    assert sum(counts.values()) == EDITS and max(counts.values()) <= MODULE_CAP
    restored = deepcopy(baseline)
    weights, receipts = {}, {}
    for m in by_module:
        value, proof = weight_for(grids[m], [edit for edit in edits if edit["module"] == m])
        weights[m], receipts[m] = value, proof
        restored[m+".weight"]["sha256"] = tensor_digest(value)
    for name, data in (("group_gradient_means.pt", means), ("coco256_task_linear144_weights.pt",weights)):
        with (OUT/name).open("xb") as f:
            torch.save(data, f)
    roots = {BASELINE:BASELINE_ROOT, OLD144:OLD_ROOT, LINEAR:state_root(restored)}
    assert len(set(roots.values())) == len(ARMS)
    record = dict(status="SEALED_BEFORE_TASK_GENERATION",roots=roots,
        supervision_examples=SUPERVISION_N, code_budget=EDITS, per_module_cap=MODULE_CAP,
        sources=source["grids"], normalized_means=ref(OUT/"group_gradient_means.pt"),
        weights=ref(OUT/"coco256_task_linear144_weights.pt"),gradient_batch=ref(OUT/"gradient_batch.json"),
        normalization_constants_FP32=constants,counts=counts,shortlists=pools,moves=moves,
        selected_edits=edits,receipts=receipts,greedy=picked,
        per_group_predicted_relative_loss_drop={g:-float(by_group[g][picked["selected_indices"]].sum()) for g in GROUPS},
        benchmark_calls_at_selection=0)
    save(OUT/"selection.json",record,compact=True)
    return weights, restored, record
