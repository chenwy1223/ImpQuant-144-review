"""Independently replay native gradients, the legal ranking and benchmark pairs on CPU."""
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from statistics import mean

from common import *
from native_contracts import decode_grid, raw_state_changes, state_root, tensor_digest, validate_actual_repaired_weight
from paired_inputs import check_manifest, check_outcomes, replay_rows


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def verify_supervision(source, rows, selection):
    import numpy as np
    import torch

    batch = read(OUT/"gradient_batch.json")
    check(selection["gradient_batch"])
    assert batch["status"] == "PASS" and batch["examples"] == SUPERVISION_N
    assert batch["formal_backwards"] == SUPERVISION_N*len(MODULES)
    for artifact in (batch["references"],batch["gradient_trace"],selection["normalized_means"]):
        check(artifact)
    rpath = OUT/"supervision_manifest.json"
    roster = read(rpath)
    assert roster["rows"] == read(roster["parent"]["path"])["splits"]["S"]
    supervised = roster["rows"]
    assert not ({r["pixel_sha256"] for r in supervised} & {im["pixel_sha256"] for r in rows for im in r["images"]})
    notes = read(batch["references"]["path"])
    assert len(notes) == SUPERVISION_N
    for i, (sample, ref_record) in enumerate(zip(supervised,notes,strict=True)):
        assert ref_record["id"] == sample["id"] and ref_record["model_state_root"] == BASELINE_ROOT
        assert ref_record["prefix_tensor_records"] == sample["prefix_tensor_records"]
        assert ref_record["manifest_sha256"] == sha(rpath)
        assert ref_record["raw_native_prefix_bit_equal"] and ref_record["actual_processed_scores_bit_equal"]
        assert ref_record["generated_token_ids"][0] == ref_record["processed_argmax_token"]
        artifact = ref_record["firststep_vector_file"]
        assert artifact["path"] == str(OUT/"S_firststep_vectors"/f"{i:03d}.pt")
        check(artifact)
        vectors = torch.load(artifact["path"],map_location="cpu",weights_only=True)
        assert vectors["id"] == sample["id"]
        for key, ref_key in (("raw_logits_FP32","raw_logits_sha256"),("processed_scores_FP32","processed_scores_sha256")):
            assert vectors[key].dtype == torch.float32 and tensor_digest(vectors[key]) == ref_record[ref_key]
        if sample["answer_kind"] == "choice":
            assert ref_record["target_token_id"] == sample["answer_token_ids"][0]
            logits = vectors["raw_logits_FP32"].numpy().astype(np.float64)
            maximum = np.max(logits)
            nll = maximum + np.log(np.exp(logits-maximum).sum()) - logits[ref_record["target_token_id"]]
        else:
            assert ref_record["answer_tokens"] == len(sample["answer_token_ids"])
            nll = mean(z-t for z,t in zip(ref_record["logsumexp_fp32"],ref_record["target_logits_fp32"],strict=True))
        assert math.isclose(nll,ref_record["answer_nll"],rel_tol=5e-6,abs_tol=5e-6)
    groups = {g:[i for i,r in enumerate(supervised) if r["training_group"]==g] for g in GROUPS}
    assert all(len(v)==16 for v in groups.values())
    norm = {k:float(np.float32(max(math.fsum(notes[i]["answer_nll"] for i in v)/len(v),1e-6))) for k,v in groups.items()}
    assert norm == selection["normalization_constants_FP32"]
    table = lines(batch["gradient_trace"]["path"])
    by_key = {(r["module"],r["manifest_index"]):r for r in table}
    assert len(by_key) == len(table) == len(MODULES)*SUPERVISION_N
    means = torch.load(selection["normalized_means"]["path"],map_location="cpu",weights_only=True)
    assert set(means) == set(MODULES)
    options = {}
    all_blocks = []
    eligible_total = 0
    for module in MODULES:
        record = source["grids"][module]
        check(record)
        grid = torch.load(record["path"],map_location="cpu",weights_only=True)
        assert grid["module"] == module
        options[module] = grid
        base = decode_grid(grid).to(torch.float64).numpy()
        sums = {g:np.zeros(base.shape,dtype=np.float64) for g in GROUPS}
        projected_source = []
        for i,row in enumerate(supervised):
            r = by_key[module,i]
            assert r["id"] == row["id"] and r["training_group"] == row["training_group"]
            assert r["native_raw_bits_equal"] and r["native_processed_bits_equal"]
            if row["answer_kind"] == "open":
                assert r["native_teacher_logits_bit_equal"]
            restored = r["route_restoration"]
            assert all(restored[k] for k in (
                "all196_native_forwards_restored", "all28_decoder_forwards_restored",
                "all_tensor_gradient_flags_and_training_restored"))
            assert r["objective_value"] == notes[i]["answer_nll"]
            check(r["artifact"])
            value = torch.load(r["artifact"]["path"],map_location="cpu",weights_only=True)
            assert all(value[k] == r[k] for k in ("module","id","manifest_index","training_group"))
            g = value["native_gradient_BF16"]
            assert g.shape == base.shape and g.dtype == torch.bfloat16 and not g.requires_grad
            assert tensor_digest(g) == r["gradient_sha256"]
            numeric = g.to(torch.float64).numpy()
            assert np.isfinite(numeric).all()
            sums[row["training_group"]] += numeric
            projected_source.append(g)
        mean_g = np.stack([sums[g]/(len(groups[g])*norm[g]) for g in GROUPS]).mean(axis=0)
        assert means[module].dtype == torch.float64 and np.allclose(mean_g,means[module].numpy(),rtol=2e-15,atol=0)

        code = grid["codes"]
        scale = grid["scales"].repeat_interleave(128,dim=1)
        zero = grid["zeros"].repeat_interleave(128,dim=1)
        steps = [(scale*(code.float()+d-zero)).to(torch.bfloat16).to(torch.float64).numpy()-base for d in (-1,1)]
        q = code.numpy()
        lo,hi = steps
        eligible = (q>0)&(q<7)&np.isfinite(lo)&np.isfinite(hi)&(lo!=0)&(hi!=0)
        best = np.where(eligible,np.maximum(-mean_g*lo,-mean_g*hi),-np.inf)
        ids = np.argsort(-best.reshape(-1),kind="stable")[:POOL_PER_MODULE]
        assert np.isfinite(best.reshape(-1)[ids]).all()
        coords = [divmod(int(i),q.shape[1]) for i in ids]
        shortlist = selection["shortlists"][module]
        assert [(r["row"],r["column"]) for r in shortlist["coordinates"]] == coords
        assert int(eligible.sum()) == shortlist["eligible_coordinates"]
        eligible_total += int(eligible.sum())
        block = []
        for r,c in sorted(coords):
            for j,d in enumerate((-1,1)):
                step = steps[j][r,c]
                block.append((module,r,c,d,float(step)))
        rr,cc = zip(*[(r,c) for _,r,c,_,_ in block],strict=True)
        projection = torch.stack([g[list(rr),list(cc)].to(torch.float64) for g in projected_source])
        projection = projection*torch.tensor([m[4] for m in block],dtype=torch.float64)
        all_blocks.append((block,projection))
        del projected_source, sums

    candidates = [m for block,_ in all_blocks for m in block]
    p = torch.cat([v for _,v in all_blocks],dim=1)
    pgroup = torch.stack([p[groups[k]].sum(dim=0)/(len(groups[k])*norm[k]) for k in GROUPS])
    aggregate_loss = pgroup.mean(dim=0).tolist()
    assert candidates == [(x["module"],x["row"],x["column"],x["direction"],x["delta_eff"]) for x in selection["moves"]]
    ranking = sorted(range(len(candidates)),key=lambda i:(aggregate_loss[i],int(candidates[i][0].split(".")[2]),*candidates[i][1:4]))
    changes, used, picked = Counter(), set(), []
    for i in ranking:
        m,r,c,d,_ = candidates[i]
        if (m,r,c) in used or changes[m] == MODULE_CAP:
            continue
        used.add((m,r,c))
        changes[m] += 1
        picked.append(i)
        if len(picked)==EDITS:
            break
    assert picked == selection["greedy"]["selected_indices"]
    assert len(picked) == EDITS and int(sum(changes.values())) == EDITS
    assert selection["counts"] == {m:changes[m] for m in MODULES}
    assert [selection["moves"][i] for i in picked] == selection["selected_edits"]
    assert [aggregate_loss[i] for i in picked] == [r["marginal_loss_change"] for r in selection["greedy"]["trace"]]
    assert selection["per_group_predicted_relative_loss_drop"] == {g:-float(pgroup[i,picked].sum()) for i,g in enumerate(GROUPS)}
    return options, changes, eligible_total, len(table)


def verify_weights(source, selection, grids, changes):
    import torch

    check(selection["weights"])
    lookup = torch.load(selection["weights"]["path"],map_location="cpu",weights_only=True)
    baseline = read(source["baseline_applied"]["path"])["all_tensors"]
    assert state_root(baseline) == BASELINE_ROOT
    for m,grid in grids.items():
        assert tensor_digest(decode_grid(grid)) == baseline[m+".weight"]["sha256"]
    assert set(lookup) == set(changes) == set(selection["receipts"])
    after = deepcopy(baseline)
    for m,t in lookup.items():
        proof = selection["receipts"][m]
        assert len([e for e in selection["selected_edits"] if e["module"]==m]) == changes[m]
        validate_actual_repaired_weight(grids[m],t,proof,expected_count=changes[m])
        after[m+".weight"]["sha256"] = tensor_digest(t)
    assert raw_state_changes(baseline,after,lookup)
    roots = {BASELINE:BASELINE_ROOT,OLD144:OLD_ROOT,LINEAR:state_root(after)}
    assert selection["roots"] == roots
    expected = [("baseline_loaded",baseline),("baseline_after_gradients",baseline),
                ("repaired_applied",after),("repaired_after_evaluation",after),("baseline_restored",baseline)]
    for audit,(label,state) in zip(read(OUT/"state_audits.json"),expected,strict=True):
        assert audit["stage"]==label and audit["all_tensors"]==state and audit["root"]==state_root(state)
    return roots


def verify_intervals(rows, sets, stats):
    import numpy as np

    labels = (("repair_minus_coco256",BASELINE),("repair_minus_old8_repair",OLD144))
    d = 20_000
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    image_rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED+1))
    assert stats["bootstrap"]==dict(draws=d,seed=BOOTSTRAP_SEED,image_group_seed=BOOTSTRAP_SEED+1,common_within_task_indices=True)
    task_values = {a:[] for a in ARMS}
    overall = {label:(np.zeros(d),np.zeros(d)) for label,_ in labels}
    delta_macro = {label:[] for label,_ in labels}

    def close(a,b):
        assert np.allclose(a,b,rtol=0,atol=3e-10),(a,b)
    def interval(x):
        return np.quantile(x,(0.025,0.975))

    assert set(stats["tasks"]) == set(TASKS)
    for task in TASKS:
        samples=[r for r in rows if r["task"]==task]
        n=len(samples)
        found=stats["tasks"][task]
        assert n==found["n"]==TASK_COUNTS[task]
        values={}
        for a in ARMS:
            records=[sets[a][r["id"]] for r in samples]
            values[a]=scores=[r["score"] for r in records]
            entry=found["arms"][a]
            assert entry["full"]==scores.count(1) and entry["zero"]==scores.count(0)
            assert entry["partial"]==n-entry["full"]-entry["zero"]
            close(entry["pct"],100*mean(scores))
            task_values[a].append(entry["pct"])
            assert found["generation"][a]==dict(
                first_token_eos=sum(r["generated_token_ids"][0]==r["generation_kwargs"]["eos_token_id"] for r in records),
                empty=sum(not r["response"].strip() for r in records))
        groups={}
        for i,row in enumerate(samples):
            groups.setdefault(row["visual_group"],[]).append(i)
        blocks=[groups[k] for k in sorted(groups)]
        k=len(blocks)
        assert k==found["distinct_image_sets"]
        sizes=np.array([len(b) for b in blocks])
        picked=rng.integers(0,n,size=(d,n))
        chosen=image_rng.integers(0,k,size=(d,k))
        for label,arm in labels:
            delta=np.array(values[LINEAR])-np.array(values[arm])
            result=found["contrasts"][label]
            diff=100*mean(delta)
            close(result["delta_pp"],diff)
            delta_macro[label].append(diff)
            assert result["improved"]==int((delta>1e-12).sum())
            assert result["declined"]==int((delta< -1e-12).sum())
            assert result["unchanged"]==n-result["improved"]-result["declined"]
            qb=np.take(delta,picked).sum(axis=1)*(100/n)
            cell=np.array([math.fsum(delta[i] for i in b) for b in blocks])
            gb=100*np.take(cell,chosen).sum(axis=1)/np.take(sizes,chosen).sum(axis=1)
            close(result["paired_CI95_pp"],interval(qb))
            close(result["image_group_CI95_pp"],interval(gb))
            overall[label][0][:]+=qb
            overall[label][1][:]+=gb
        del picked, chosen
    for arm,scores in task_values.items():
        close(stats["macro"][arm],mean(scores))
    for label,_ in labels:
        x=stats["macro_contrasts"][label]
        close(x["delta_pp"],mean(delta_macro[label]))
        close(x["paired_CI95_pp"],interval(overall[label][0]/len(TASKS)))
        close(x["image_group_CI95_pp"],interval(overall[label][1]/len(TASKS)))


def main():
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(2)
    root_lock = HERE/"execution_lock.json"
    digest = sha(root_lock)
    lock = read(root_lock)
    marker = read(OUT/"pilot_complete.json")
    exited = read(OUT/"scientific_exit_receipt.json")
    final = read(OUT/"final_lock_verification_receipt.json")
    assert marker["attempt"] == NAME and marker["status"] == "PASS"
    assert exited["exit_code"]==0 and marker["lock_sha256"]==exited["lock_sha256"]==final["lock_sha256"]==digest
    assert exited["marker_sha256"]==sha(OUT/"pilot_complete.json")
    assert final["full"] is True and final["bindings"]==lock["bindings"]==len(lock["files"])
    assert final["bound_bytes"]==sum(r["bytes"] for r in lock["files"])
    assert marker["baseline_restored_and_verified"] is True and marker["peak_allocated_GiB"]<=22.5
    assert marker["network_calls"]==marker["external_judge_calls"]==marker["random_fallback_calls"]==0
    assert all(marker[k]==value for k,value in SCOPE_CONTRACT.items())
    assert all(marker[k]==value for k,value in completion_counts(len(ARMS)).items())

    preflight = read(OUT/"cpu_preflight.json")
    for r in preflight["artifacts"].values():
        check(r)
    source = read(OUT/"source_snapshot.json")
    rows = check_manifest(read(OUT/"data_manifest.json"))
    assert sha(OUT/"data_manifest.json")==marker["data_manifest_sha256"]
    selection = read(OUT/"selection.json")
    assert selection["status"]=="SEALED_BEFORE_TASK_GENERATION" and selection["benchmark_calls_at_selection"]==0
    assert selection["sources"] == source["grids"]
    assert marker["selection_sha256"]==sha(OUT/"selection.json")
    assert marker["gradient_batch_sha256"]==sha(OUT/"gradient_batch.json")
    grids,changes,eligible,n = verify_supervision(source,rows,selection)
    roots = verify_weights(source,selection,grids,changes)
    assert marker["roots"] == roots

    outcomes=read(OUT/"outcomes.json")
    refs=read(OUT/"reference_outcomes.json")
    assert all(outcomes[a]==refs[a] for a in refs)
    for arm, parent in ((BASELINE,"baseline"),(OLD144,"old_repair")):
        pinned=source["parent_pins"][parent]["results/outcomes.json"]
        check(pinned)
        assert refs[arm]==read(pinned["path"])[arm]
    assert outcomes[LINEAR]==lines(OUT/"measurement_trace.jsonl")
    sets=check_outcomes(rows,outcomes,roots)
    for r in outcomes[LINEAR]:
        assert r["attempt"]==NAME and r["data_manifest_sha256"]==marker["data_manifest_sha256"]
        assert r["selection_sha256"]==marker["selection_sha256"]
    for row,r in zip(replay_rows(rows),read(OUT/"continuity_replays.json"),strict=True):
        older=sets[BASELINE][row["id"]]
        assert all(r[k]==older[k] for k in ("id","model_state_root","generated_token_ids","response","score","prefix_tensor_records"))
    vision=read(OUT/"vision_equivalence.json")
    assert vision["status"]=="PASS" and len(vision["cases"])==2
    assert all(c["bytewise_equal"] for c in vision["cases"]) and read(OUT/"vision_dispatch.json")["tiled_calls"]>0
    verify_intervals(rows,sets,read(OUT/"statistics_fixed.json"))
    assert not torch.cuda.is_initialized()
    report=dict(status="PASS",protocol_pass=True,execution_lock_sha256=digest,bindings=final["bindings"],
                baseline_calibration_images=MODEL_CALIBRATION_N,native_gradients=n,eligible_coordinates=eligible,
                independent_global_ranking=True,legally_modified_codes=sum(changes.values()),native_grids=len(grids),
                paired_questions=EVAL_N,actual_outcomes=len(ARMS)*EVAL_N,verified_reused_outcomes=2*EVAL_N,
                new_outcomes=EVAL_N,roots=roots,statistics_sha256=sha(OUT/"statistics_fixed.json"),
                gradient_manifest_sha256=sha(OUT/"gradient_trace.jsonl"),cpu_only=True,independent_bootstrap=True,
                independent_legal_grid_reconstruction=True)
    target=OUT/"independent_verified.json"
    if target.exists():
        assert report == read(target)
    else:
        save(target,report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
