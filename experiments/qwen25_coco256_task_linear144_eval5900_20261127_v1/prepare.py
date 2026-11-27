"""Seal the already verified state, six grids and unchanged S64/benchmarks with CUDA hidden."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

from common import *
from inputs_native import load
from native_contracts import decode_grid, state_root, tensor_digest
from paired_inputs import check_manifest, verify_sources


def item(group, name):
    return group[name]["path"]


def load_lock(group):
    prior = read(item(group, "code/execution_lock.json"))
    entries = {r["path"]: r for r in prior["files"]}
    assert len(entries) == prior["bindings"] == len(prior["files"])
    return prior, entries


def verify_parent(group, count):
    lock, entries = load_lock(group)
    lock_id = group["code/execution_lock.json"]["sha256"]
    proof = read(item(group, "results/independent_verified.json"))
    exited = read(item(group, "results/scientific_exit_receipt.json"))
    verification = read(item(group, "results/final_lock_verification_receipt.json"))
    assert lock_id == proof["execution_lock_sha256"] == exited["lock_sha256"] == verification["lock_sha256"]
    assert proof["protocol_pass"] is True and proof["paired_questions"] == EVAL_N
    assert proof["actual_outcomes"] == count
    assert exited["exit_code"] == 0 and verification["full"] is True and verification["bindings"] == lock["bindings"]
    return proof, entries


def add(files, record):
    assert files.setdefault(record["path"], record) == record


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not (HERE/"execution_lock.json").exists() and not (OUT/"launched_once.json").exists()
    import torch

    torch.set_num_threads(2)
    pinned = read(HERE/"source_pins.json")
    for group in pinned.values():
        for record in group.values():
            check(record)
    base, repaired, supervision = (pinned[k] for k in ("baseline", "old_repair", "supervision"))
    base_proof, files = verify_parent(base, 2*EVAL_N)
    old_proof, _ = verify_parent(repaired, 2*EVAL_N)
    assert base_proof["roots"][BASELINE] == BASELINE_ROOT and old_proof["roots"][OLD144] == OLD_ROOT
    assert base_proof["independent_native_grids"] == len(MODULES)
    recovery = read(item(base, "results/independent_verification_recovery.json"))
    assert recovery["status"] == "PASS" and recovery["new_model_calls"] == recovery["new_GPU_calls"] == 0
    assert recovery["independent_proof"] == base["results/independent_verified.json"]
    for group in pinned.values():
        for record in group.values():
            add(files, record)

    meta = read(item(base, "results/calibration_complete.json"))
    assert meta["status"] == "PASS" and meta["images"] == MODEL_CALIBRATION_N
    assert meta["modules"] == LINEARS and meta["blocks"] == BLOCKS
    assert meta["state"]["sha256"] == BASELINE_SHA
    assert meta["state"] in base_proof["big_artifacts_retained"]
    assert meta["grids_sha256"] == base["results/grids/capture_manifest.json"]["sha256"]
    check(meta["state"])
    add(files, meta["state"])
    state_rec = read(item(base, "results/applied_model_hashes.json"))
    initial = state_rec["all_tensors"]
    assert state_root(initial) == state_rec["root"] == BASELINE_ROOT
    assert state_rec["modules"] == LINEARS and state_rec["state_sha256"] == BASELINE_SHA
    capture = read(item(base, "results/grids/capture_manifest.json"))
    assert capture["status"] == "NATIVE_MULTIPLE_GRID_CAPTURE_ROUNDTRIP_PASS"
    assert capture["targets"] == list(MODULES) and capture["target_calls"] == dict.fromkeys(MODULES, 1)
    assert capture["modules_observed"] == LINEARS and capture["captured_grid_modules"] == len(MODULES)
    grids = {}
    for module in MODULES:
        r = capture["modules"][module]
        p = Path(r["grid_path"])
        assert p.parent == BASE/"grids"
        g = dict(path=str(p),bytes=r["grid_bytes"],sha256=r["grid_sha256"])
        check(g)
        grid = torch.load(p, map_location="cpu", weights_only=True)
        assert grid["module"] == module and tensor_digest(decode_grid(grid)) == initial[module+".weight"]["sha256"]
        grids[module] = g
        add(files, g)

    s_lock, s_files = load_lock(supervision)
    s_proof = read(item(supervision, "results/independent_verified.json"))
    assert s_proof["protocol_pass"] is True and s_proof["new_gradients"] == SUPERVISION_N*len(MODULES)
    assert s_proof["execution_lock_sha256"] == supervision["code/execution_lock.json"]["sha256"]
    assert s_proof["bindings"] == s_lock["bindings"]
    for path in (ROUTE_HERE/"gradient_engine.py", RANK_HERE/"editable_grid.py", RANK_HERE/"checkpoint_suffix.py"):
        record = s_files[str(path)]
        check(record)
        add(files, record)
    assert s_files[item(supervision, "results/data_manifest.json")] == supervision["results/data_manifest.json"]
    S = read(item(supervision, "results/data_manifest.json"))["splits"]["S"]
    assert len(S) == len({r["id"] for r in S}) == SUPERVISION_N
    assert [r["split_index"] for r in S] == list(range(SUPERVISION_N))
    assert Counter(r["training_group"] for r in S) == dict.fromkeys(GROUPS, 16)
    assert Counter(r["answer_kind"] for r in S) == {"choice":32, "open":32}
    assert {r["source_split"] for r in S} == {"train"}
    for r in S:
        record = s_files[r["image"]]
        assert record["sha256"] == r["sha256"] and record["bytes"] == r["image_bytes"]
        check(record)
        add(files, record)
        load(r)  # Recreate the parent S64 prefix and literal target IDs.

    source_path = Path(item(base, "results/data_manifest.json"))
    assert base["results/data_manifest.json"]["sha256"] == repaired["results/data_manifest.json"]["sha256"]
    rows = check_manifest(read(source_path))
    assert not ({r["pixel_sha256"] for r in S} & {im["pixel_sha256"] for r in rows for im in r["images"]})
    references = {
        BASELINE: read(item(base, "results/outcomes.json"))[BASELINE],
        OLD144: read(item(repaired, "results/outcomes.json"))[OLD144],
    }
    verify_sources(rows, references)

    with (OUT/"data_manifest.json").open("xb") as dest:
        dest.write(source_path.read_bytes())
    save(OUT/"reference_outcomes.json", references, compact=True)
    save(OUT/"supervision_manifest.json", dict(status="FROZEN_PARENT_S64",rows=S,
         parent=supervision["results/data_manifest.json"],n=SUPERVISION_N,groups=list(GROUPS)), compact=True)
    save(OUT/"source_snapshot.json", dict(baseline_state=meta["state"],baseline_applied=base["results/applied_model_hashes.json"],
         baseline_calibration=base["results/calibration_complete.json"],grids=grids,
         gradient_engine=s_files[str(ROUTE_HERE/"gradient_engine.py")],parent_pins=pinned))
    assert not torch.cuda.is_initialized()
    report = dict(status="PASS",attempt=NAME,cuda_initialized=False,model_calls=0,
        baseline_and_grids_verified=True,data_seal_verified=True,supervision_examples=SUPERVISION_N,
        eval_examples=EVAL_N,reused_outcomes=2*EVAL_N,supervision_eval_pixel_overlap=0,
        artifacts={n:ref(OUT/n) for n in ("data_manifest.json","reference_outcomes.json","supervision_manifest.json","source_snapshot.json")})
    save(OUT/"cpu_preflight.json",report)
    for path in [*HERE.glob("*.py"), HERE/"source_pins.json", HERE/"PROTOCOL.md", *OUT.glob("*.json")]:
        add(files, ref(path))
    lock = dict(attempt=NAME,files=sorted(files.values(),key=lambda r:r["path"]),bindings=len(files),
        shared_queue_seconds=14400,initial_admission_wait_elapsed_seconds=0.0,initial_worker_elapsed_seconds=0.0,
        admission_RAM_GiB=40,admission_disk_GiB=ADMISSION_DISK_GIB,runtime_RAM_GiB=8,runtime_disk_GiB=RUNTIME_DISK_GIB,
        allocated_GiB=22.5,worker_seconds=WORKER_LIMIT,forced_RAM_reserve_waiver=False,
        forced_disk_reserve_waiver=False,auto_retry=False,scheduled_monitoring=False,**SCOPE_CONTRACT)
    save(HERE/"execution_lock.json",lock)
    print(json.dumps(dict(status="CPU_READY",execution_lock_sha256=sha(HERE/"execution_lock.json"),
        bindings=lock["bindings"],supervision_examples=SUPERVISION_N,paired_questions=EVAL_N,
        verified_reference_outcomes=2*EVAL_N,baseline_grids=len(MODULES))),flush=True)


if __name__ == "__main__":
    main()
