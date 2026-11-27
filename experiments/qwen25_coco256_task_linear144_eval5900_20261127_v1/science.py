"""Native S64 first-order code repair followed by one new matched inference arm."""
import json
import time

from common import *
from eval_inputs import decode, generation_kwargs, prefixes, processor, raw_digest, tensor_records
from native_contracts import fingerprint, state_root
from paired_inputs import replay_rows
from scorer import grade
from selection import choose
from statistics_fixed import summarize
from supervision import collect
from vision_route import DISPATCH, install_and_verify


def run_science(native, adapter, model, baseline, phase):
    import torch

    started = time.monotonic()
    cpu = read(OUT/"cpu_preflight.json")
    for artifact in cpu["artifacts"].values():
        check(artifact)
    data_hash = cpu["artifacts"]["data_manifest.json"]["sha256"]
    rows = read(OUT/"data_manifest.json")["rows"]
    refs = read(OUT/"reference_outcomes.json")
    indices = {arm:{r["id"]:r for r in refs[arm]} for arm in (BASELINE,OLD144)}
    audits = []

    def audit(stage, expected, observed=None):
        phase("full_state_audit",audit_name=stage)
        actual = fingerprint(model) if observed is None else observed
        assert actual == expected, stage
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(not p.requires_grad and p.grad is None for p in model.buffers())
        audits.append(dict(stage=stage,root=state_root(actual),all_tensors=actual))
        save(OUT/"state_audits.json",audits,compact=True)

    assert state_root(baseline) == BASELINE_ROOT
    audit("baseline_loaded",baseline,observed=baseline)
    trace, notes = collect(native,adapter,model,phase)
    audit("baseline_after_gradients",baseline)
    phase("rank_and_seal_144_legal_moves")
    patch, repaired, selection = choose(trace,notes,baseline)
    selection_sha = sha(OUT/"selection.json")
    roots = selection["roots"]
    originals = {m:model.get_submodule(m).weight.detach().cpu().clone() for m in patch}

    proc = processor()
    device = model.get_submodule(TARGET_MODULE).weight.device
    assert model.generation_config.repetition_penalty == 1.05
    install_and_verify(model,rows,phase)

    def infer(row,arm):
        tick = time.monotonic()
        prefix = prefixes(row,proc)
        identity = tensor_records(prefix)
        assert all(identity == i[row["id"]]["prefix_tensor_records"] for i in indices.values())
        kwargs = generation_kwargs(row,proc)
        assert kwargs == row["generation_kwargs"]
        x = {k:v.to(device) for k,v in prefix.items()}
        with torch.no_grad():
            output = model.generate(**x,**kwargs,return_dict_in_generate=True)
        length = x["input_ids"].shape[1]
        assert torch.equal(output.sequences[0,:length],x["input_ids"][0])
        tokens = output.sequences[0,length:].detach().cpu().tolist()
        assert 0 < len(tokens) <= kwargs["max_new_tokens"]
        answer = decode(tokens,proc,row["task"])
        return dict(attempt=NAME,id=row["id"],task=row["task"],arm=arm,target=row["target"],
            source_index=row["source_index"],image_sha256=raw_digest(row),prefix_tensor_records=identity,
            generation_kwargs=kwargs,generated_token_ids=tokens,response=answer,score=grade(row,answer),
            model_state_root=roots[arm],data_manifest_sha256=data_hash,selection_sha256=selection_sha,
            seconds=time.monotonic()-tick)

    replays = []
    for row in replay_rows(rows):
        phase("replay_coco256_baseline_anchor",task=row["task"],sample_id=row["id"])
        item = infer(row,BASELINE)
        ref = indices[BASELINE][row["id"]]
        for key in ("generated_token_ids","response","score","prefix_tensor_records","model_state_root"):
            assert item[key] == ref[key], (row["id"],key)
        replays.append(item)
    save(OUT/"continuity_replays.json",replays)
    with torch.no_grad():
        for module, value in patch.items():
            model.get_submodule(module).weight.copy_(value.to(device))
    audit("repaired_applied",repaired)

    records = []
    with (OUT/"measurement_trace.jsonl").open("x",encoding="utf-8",buffering=1) as stream:
        for task in RUN_ORDER:
            subset = [r for r in rows if r["task"] == task]
            n = TASK_COUNTS[task]
            assert len(subset) == n
            task_outputs = []
            for i,row in enumerate(subset):
                phase("paired_inference",task=task,arm=LINEAR,sample=i,expected=n)
                r = infer(row,LINEAR)
                task_outputs.append(r)
                stream.write(json.dumps(r,allow_nan=False,separators=(",",":"))+"\n")
            ratings = {arm:[indices[arm][r["id"]]["score"] for r in task_outputs] for arm in (BASELINE,OLD144)}
            fresh = [r["score"] for r in task_outputs]
            ratings[LINEAR] = fresh
            delta = [b-a for a,b in zip(ratings[BASELINE],fresh,strict=True)]
            save(OUT/"tasks"/f"{task}.json",dict(task=task,paired=n,
                 means={arm:100*sum(a)/n for arm,a in ratings.items()},baseline_delta_pp=100*sum(delta)/n,
                 improved=sum(x>1e-12 for x in delta),declined=sum(x< -1e-12 for x in delta)))
            records.extend(task_outputs)
    assert len(records) == len({r["id"] for r in records}) == EVAL_N
    audit("repaired_after_evaluation",repaired)
    with torch.no_grad():
        for module, value in originals.items():
            model.get_submodule(module).weight.copy_(value.to(device))
    audit("baseline_restored",baseline)

    outputs = {**refs, LINEAR:records}
    save(OUT/"outcomes.json",outputs,compact=True)
    save(OUT/"vision_dispatch.json",DISPATCH)
    assert DISPATCH["tiled_calls"]>0
    mapped = {arm:{r["id"]:r for r in outputs[arm]} for arm in ARMS}
    save(OUT/"statistics_fixed.json",summarize(rows,mapped))
    phase("saved_all_pairs_and_statistics",pairs=len(records))
    return dict(SCOPE_CONTRACT,benchmark_generations=len(records),native_supervision_generations=len(notes),
        baseline_continuity_replays=len(replays),whole_state_audits=len(audits),
        physical_states=len(ARMS),evaluation_examples_per_state=len(records),roots=roots,
        data_manifest_sha256=data_hash,selection_sha256=selection_sha,
        gradient_batch_sha256=sha(OUT/"gradient_batch.json"),seconds=time.monotonic()-started)
