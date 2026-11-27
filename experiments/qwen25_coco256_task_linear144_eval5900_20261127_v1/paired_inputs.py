"""Check paired answer, source and generation identities for all three arms."""
from collections import Counter

from common import ARMS, BASELINE, BASELINE_ROOT, EVAL_N, OLD144, OLD_ROOT, TASKS, TASK_COUNTS
from eval_inputs import decode, processor, raw_digest
from scorer import grade


def check_manifest(manifest):
    rows = manifest["rows"]
    assert manifest["per_task"] == TASK_COUNTS
    assert manifest["dataset_receipts"]["mmmu_val"]["source_rows"] == TASK_COUNTS["mmmu_val"]
    assert len(rows) == len({r["id"] for r in rows}) == EVAL_N
    assert Counter(r["task"] for r in rows) == TASK_COUNTS
    return rows


def check_outcomes(rows, outcomes, roots):
    assert set(outcomes) == set(roots) and set(outcomes) <= set(ARMS)
    sets = {arm: {r["id"]: r for r in results} for arm, results in outcomes.items()}
    want = {r["id"] for r in rows}
    assert all(len(outcomes[arm]) == len(sets[arm]) == EVAL_N and set(sets[arm]) == want for arm in outcomes)
    proc = processor()
    for row in rows:
        native = sets[BASELINE][row["id"]]["prefix_tensor_records"]
        raw = raw_digest(row)
        for arm, index in sets.items():
            r = index[row["id"]]
            assert r["arm"] == arm and r["model_state_root"] == roots[arm], (arm, row["id"])
            assert r["prefix_tensor_records"] == native
            assert all(r[k] == row[k] for k in ("task", "target", "source_index", "generation_kwargs"))
            assert r["image_sha256"] == raw or (len(raw) == 1 and r["image_sha256"] == raw[0])
            tokens = r["generated_token_ids"]
            assert 0 < len(tokens) <= row["generation_kwargs"]["max_new_tokens"]
            assert decode(tokens, proc, row["task"]) == r["response"]
            assert grade(row, r["response"]) == r["score"]
    return sets


def verify_sources(rows, references):
    return check_outcomes(rows, references, {BASELINE: BASELINE_ROOT, OLD144: OLD_ROOT})


def replay_rows(rows):
    return [next(r for r in rows if r["task"] == task) for task in TASKS]
