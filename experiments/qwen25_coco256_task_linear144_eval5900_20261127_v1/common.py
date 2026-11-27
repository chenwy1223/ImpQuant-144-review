"""Repeat task-supervised 144-code repair on the COCO-256 uniform checkpoint."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("IMPQUANT_ROOT", Path(__file__).resolve().parents[2])).resolve()
NAME = "qwen25_coco256_task_linear144_eval5900_20261127_v1"
HERE = ROOT / "experiments" / NAME
OUT = ROOT / "results" / NAME
PY = sys.executable
MODEL = Path(os.environ.get("IMPQUANT_MODEL_DIR", ROOT / "models/Qwen2.5-VL-7B-Instruct"))
BASE = ROOT / "results/qwen25_coco256_uniform_recal_eval5900_20261127_v1"
OLD = ROOT / "results/qwen25_impquant6_linear144_eval5900_20261126_v3"
SUPERVISION = ROOT / "results/qwen25_fourtask_code96_20261111_v1"
ROUTE_HERE = ROOT / "experiments/qwen25_score_comparison_20261102_v2"
RANK_HERE = ROOT / "experiments/qwen25_module_rank_20261027_v1"
BASELINE_SHA = "7a2a2624c5bb54ab4dfcd9c6f309e8988085d02196779edb29e45efe157e917e"
BASELINE_ROOT = "f879777ef84d8280732b85bee958bdbbcfb41e0ae1de9cec05b948b5700de4b2"
OLD_ROOT = "7c72c7602febe67e039f69b7ba0a617bccb75227c8f519e4b30f6165fac0b024"
MODEL_CALIBRATION_N = 256
GPU_INDEX = int(os.environ.get("IMPQUANT_GPU_INDEX", "0"))
UUID = os.environ.get("IMPQUANT_GPU_UUID", "")
LINEARS = 196
BLOCKS = 28
MODULES = tuple(f"model.layers.{i}.self_attn.v_proj" for i in (7, 11, 15, 19, 23, 27))
TARGET_MODULE = MODULES[0]
GROUPS = ("scienceqa", "aokvqa", "textvqa", "ocr")
SUPERVISION_N = 64
SPLIT_COUNTS = {"S": SUPERVISION_N}
SUPERVISION_TASKS = ("scienceqa_img","mmbench_en_dev","textvqa_val","ocrbench")
EDITS = 144
POOL_PER_MODULE = 128
MODULE_CAP = 96
BASELINE, OLD144, LINEAR = "uniform_coco256", "global_linear144", "uniform_coco256_task_linear144"
ARMS = EVALUATED_ARMS = (BASELINE, OLD144, LINEAR)
TASK_COUNTS = {"vizwiz_vqa_val":1000, "scienceqa_img":1000, "mmbench_en_dev":1000,
               "textvqa_val":1000, "ocrbench":1000, "mmmu_val":900}
TASKS = tuple(TASK_COUNTS)
RUN_ORDER = ("scienceqa_img", "mmbench_en_dev", "mmmu_val", "ocrbench", "textvqa_val", "vizwiz_vqa_val")
EVAL_N = sum(TASK_COUNTS.values())
BOOTSTRAP_SEED = 20261127
ADMISSION_DISK_GIB = 55
RUNTIME_DISK_GIB = 40
WORKER_LIMIT = 36000
SCOPE_CONTRACT = dict(
    calibration_images=MODEL_CALIBRATION_N, requantization=False,
    supervision_examples=SUPERVISION_N, supervision_groups=list(GROUPS),
    fresh_native_gradients=SUPERVISION_N * len(MODULES), reused_gradients=0,
    selected_modules=list(MODULES), coordinate_shortlist_per_module=POOL_PER_MODULE,
    code_budget=EDITS, per_module_cap=MODULE_CAP, empirical_Fisher=False,
    fixed_scale_zero=True, per_coordinate_adjacent_only=True,
    evaluation_examples=EVAL_N, evaluation_per_task=TASK_COUNTS,
    evaluated_arms=list(ARMS), reused_outcomes=2*EVAL_N, new_outcomes=EVAL_N,
    selection_before_benchmark_generation=True, paired=True)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024**2), b""):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def check(item):
    p = Path(item["path"])
    before = p.stat()
    assert before.st_size == item["bytes"] and sha(p) == item["sha256"], str(p)
    after = p.stat()
    assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), str(p)


def save(path, value, compact=False):
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    tmp = p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(value, indent=None if compact else 2, ensure_ascii=True,
                              allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(p)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def completion_counts(physical_states):
    assert physical_states == len(ARMS)
    return dict(benchmark_generations=EVAL_N, native_supervision_generations=SUPERVISION_N,
                baseline_continuity_replays=len(TASKS), whole_state_audits=5)
