"""Replay S64 with the previous native answer-loss route on the new baseline."""
from collections import Counter
import gc
import importlib.util
import json

from common import (BASELINE, BASELINE_ROOT, GROUPS, MODULES, NAME, OUT, SUPERVISION_N,
                    check, read, ref, save, sha)
from inputs_native import load
from measure_mixed import gradient_one, measure
from native_contracts import tensor_digest


def collect(native, adapter, model, phase):
    import torch

    roster = OUT/"supervision_manifest.json"
    rows = read(roster)["rows"]
    assert len(rows) == SUPERVISION_N
    assert Counter(r["training_group"] for r in rows) == dict.fromkeys(GROUPS, 16)
    route_ref = read(OUT/"source_snapshot.json")["gradient_engine"]
    check(route_ref)
    spec = importlib.util.spec_from_file_location("_s64_frozen_native_gradient", route_ref["path"])
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)

    notes = []
    for i, row in enumerate(rows):
        phase("native_supervision_reference", task=row["task"], sample=i, expected=SUPERVISION_N)
        prefix = load(row)
        notes.append(measure(model, adapter, row, prefix, BASELINE, "S_native_reference",
                     BASELINE_ROOT, sha(roster), save_full=True))
        del prefix
    save(OUT/"supervision_references.json", notes, compact=True)

    files = []
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]["prompt_length"], i))
    with (OUT/"gradient_trace.jsonl").open("x", encoding="utf-8", buffering=1) as trace:
        for layer, module in enumerate(MODULES):
            directory = OUT/"native_gradients"/f"layer_{int(module.split('.')[2]):02d}"
            directory.mkdir(parents=True)
            for j, i in enumerate(order):
                row = rows[i]
                phase("native_supervision_gradient", module=module, task=row["task"], sample=i,
                      completed=layer*SUPERVISION_N+j, expected=SUPERVISION_N*len(MODULES))
                prefix = load(row)
                value, detail = gradient_one(model, row, prefix, notes[i], route, module)
                assert detail["gradient_sha256"] == tensor_digest(value)
                assert detail["objective_value"] == notes[i]["answer_nll"]
                assert detail["route_restoration"]["all196_native_forwards_restored"]
                assert detail["route_restoration"]["all28_decoder_forwards_restored"]
                p = directory/f"{i:03d}.pt"
                with p.open("xb") as f:
                    torch.save(dict(id=row["id"],module=module,training_group=row["training_group"],
                                    manifest_index=i,native_gradient_BF16=value), f)
                entry = dict(detail, attempt=NAME, module=module, task=row["task"], manifest_index=i,
                             training_group=row["training_group"], artifact=ref(p))
                trace.write(json.dumps(entry, separators=(",",":"), allow_nan=False)+"\n")
                files.append(entry)
                del prefix, value
                gc.collect()
                torch.cuda.empty_cache()
    assert len(files) == len({(x["module"],x["manifest_index"]) for x in files}) == SUPERVISION_N*len(MODULES)
    batch = dict(status="PASS",examples=SUPERVISION_N,route="prior_native_one_target_checkpointed",
        formal_backwards=len(files),references=ref(OUT/"supervision_references.json"),
        gradient_trace=ref(OUT/"gradient_trace.jsonl"),native_derivative_dtype="torch.bfloat16",
        accumulation_dtype="torch.float64")
    save(OUT/"gradient_batch.json",batch)
    return files, notes
