"""Protected matched W3A16 state loading, code repair and inference."""
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from common import *

START=time.monotonic();TORCH=None;STOP=threading.Event();NETWORK=[];GO=None;OWN=None
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'3rdparty/lmms-eval'),str(RANK_HERE)]

def identity(pid):
    p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),ppid=int(f[1]),pgrp=int(f[2]),start_ticks=int(f[19]),
                argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))

def stable_owner(actual,expected):return all(actual[k]==expected[k] for k in ['pid','pgrp','start_ticks','argv'])

def handshake():
    global OWN,GO
    OWN=identity(os.getpid())
    assert OWN['argv']==[PY,'-u',str(HERE/'worker.py')] and OWN['pgrp']==OWN['pid']
    save(OUT/'worker_ready.json',dict(attempt=NAME,**OWN))
    deadline=time.monotonic()+60
    while not (OUT/'worker_go.json').exists():
        assert time.monotonic()<deadline,'No verified owner GO within60sec'
        assert identity(os.getpid())==OWN,'Parent changed before GO'
        time.sleep(.05)
    GO=json.loads((OUT/'worker_go.json').read_text())
    assert GO['attempt']==NAME and GO['worker_identity']==OWN
    assert GO['owner_identity']['pid']==OWN['ppid']
    assert stable_owner(identity(OWN['ppid']),GO['owner_identity'])
    assert sha(HERE/'execution_lock.json')==GO['lock_sha256']
    save(OUT/'worker_handshake_verified.json',dict(status='READY_GO_PARENT_TICKS_ARGV_GROUP_PASS',utc=utc(),worker=OWN,owner=GO['owner_identity'],lock_sha256=GO['lock_sha256']))

def phase(name,**values):save(OUT/'worker_phase.json',dict(name=name,pid=os.getpid(),attempt=NAME,utc=utc(),**values))

def host_memory():
    d={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in Path('/proc/self/status').read_text().splitlines() if ':' in x}
    return {k+'_GiB':int(d[k].split()[0])/1024**2 for k in ['VmRSS','VmHWM','RssAnon','RssFile','VmSwap'] if k in d}

def self_guard():
    import shutil
    while not STOP.wait(2):
        try:
            owner=GO['owner_identity'];assert identity(os.getpid())==OWN,'Worker reparented or changed'
            assert stable_owner(identity(owner['pid']),owner),'Owner missing or identity changed'
            heartbeat=json.loads((OUT/'owner_heartbeat.json').read_text())
            assert heartbeat['attempt']==NAME and heartbeat['owner_identity']==owner,'Foreign owner heartbeat'
            assert heartbeat.get('status') not in ['FAILED','COMPLETE'],'Owner terminated while worker active'
            age=time.time()-datetime.datetime.fromisoformat(heartbeat['utc']).timestamp()
            assert -5<=age<=15,'Owner heartbeat stale'
            available=next(int(x.split()[1])/1024**2 for x in open('/proc/meminfo') if x.startswith('MemAvailable:'))
            disk=shutil.disk_usage(OUT).free/1024**3
            initialized=TORCH is not None and TORCH.cuda.is_initialized()
            allocated=TORCH.cuda.memory_allocated()/1024**3 if initialized else 0.
            peak=TORCH.cuda.max_memory_allocated()/1024**3 if initialized else 0.
            telemetry=dict(pid=os.getpid(),attempt=NAME,utc=utc(),allocated_GiB=allocated,peak_allocated_GiB=peak,
                           host_process=host_memory(),available_RAM_GiB=available,disk_free_GiB=disk,owner_heartbeat_age_seconds=age)
            save(OUT/'worker_memory.json',telemetry)
            assert available>=8,'HOST_RAM_BELOW_8GiB'
            assert disk>=RUNTIME_DISK_GIB,'DISK_BELOW_40GiB'
            assert max(allocated,peak)<=22.5,'ALLOCATED_ABOVE_22_5GiB'
            assert time.monotonic()-START<=WORKER_LIMIT,'WORKER_TIME_LIMIT'
        except BaseException as exc:
            try:save(OUT/'worker_self_protection.json',dict(status='WORKER_SELF_PROTECTION_STOP',utc=utc(),pid=os.getpid(),error_type=type(exc).__name__,error=str(exc)))
            finally:os._exit(78)

def offline(event,args):
    if event in ['socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr'] or (event=='socket.connect' and isinstance(args[1],tuple)):
        NETWORK.append(event);raise RuntimeError('Offline pilot forbids network/judge calls')

def legacy_digest(t):
    import torch
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode()+str(list(t.shape)).encode()+t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def fingerprint(model,legacy=False):
    from native_contracts import tensor_digest
    return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=(legacy_digest(v) if legacy else tensor_digest(v))) for k,v in model.state_dict().items()}

def state_root(records):return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_original():
    import torch
    from qmllm.cli import load_model
    phase('load_original_model')
    settings=SimpleNamespace(model='qwen2_5_vl',model_args='pretrained='+str(MODEL)+',device_map=cuda:0,min_pixels=200704,max_pixels=1003520',batch_size=1,device='cuda:0',seed=2026)
    native,adapter=load_model(settings);model=adapter.model
    assert native.model is model and native.processor is adapter.processor
    model.eval().requires_grad_(False);model.config.use_cache=False;adapter.low_memory_loss=False
    assert model.dtype==torch.bfloat16 and {str(p.device) for p in model.parameters()}=={'cuda:0'}
    expected=json.loads((ROOT/'results/visual_p2_quant_damage_n8_seed42_20261015_v1/formal/model_state_before.json').read_text())
    assert fingerprint(model,legacy=True)==expected,'Original model fingerprint changed'
    return native,adapter



def run_pilot():
    import torch
    from qmllm.methods.impquant.quantizer import apply_impquant
    from native_contracts import tensor_digest

    source = read(OUT/"source_snapshot.json")
    native, adapter = load_original()
    model = adapter.model
    phase("load_frozen_coco256_quantized_state")
    assert source["baseline_state"]["sha256"] == BASELINE_SHA
    check(source["baseline_state"])
    state = torch.load(source["baseline_state"]["path"], map_location="cpu", weights_only=True, mmap=True)
    apply_impquant(model, 3, 16, {"q_group_size": 128}, state)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    modules = {name: mod for name, mod in model.named_modules()
               if mod.__class__.__name__ == "ImpQuantLinear"}
    assert len(modules) == LINEARS
    for layer in range(BLOCKS):
        for name, value in state["weights"][layer].items():
            mod = modules[f"model.layers.{layer}.{name}"]
            assert tensor_digest(mod.weight) == tensor_digest(value)
            assert tensor_digest(mod.input_scale.float()) == tensor_digest(state["input_scales"][layer][name].float())
    del state, modules, mod, value
    gc.collect()
    torch.cuda.empty_cache()
    initial = fingerprint(model)
    expected = read(source["baseline_applied"]["path"])["all_tensors"]
    assert initial == expected and state_root(initial) == BASELINE_ROOT
    from science import run_science
    result = run_science(native, adapter, model, initial, phase)
    assert not NETWORK and all(not p.requires_grad and p.grad is None for p in model.parameters())
    for key, value in SCOPE_CONTRACT.items():
        assert type(result.get(key)) is type(value) and result[key] == value, (key, result.get(key))
    for key, value in completion_counts(result["physical_states"]).items():
        assert result[key] == value, (key, result[key], value)
    save(OUT/"pilot_complete.json",dict(status="PASS",utc=utc(),attempt=NAME,worker_identity=OWN,
        owner_identity=GO["owner_identity"],lock_sha256=GO["lock_sha256"],network_calls=0,
        external_judge_calls=0,random_fallback_calls=0,baseline_restored_and_verified=True,
        owner_handshake=True,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        quantized_state_sha256=source["baseline_state"]["sha256"],**result))
    print(json.dumps(dict(event="S64_FIRST_ORDER_144_PAIRED_EVALUATION_COMPLETE",outcomes=EVAL_N)),flush=True)


def main():
    global TORCH
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    handshake();threading.Thread(target=self_guard,daemon=True).start()
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.addaudithook(offline)
    import torch
    TORCH=torch;torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(2026)
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    try:run_pilot()
    except BaseException as exc:
        save(OUT/'worker_failed.json',dict(status='WORKER_FAILED_NO_AUTO_RETRY',attempt=NAME,utc=utc(),error_type=type(exc).__name__,error=str(exc),gradient_route_report=getattr(exc,'gradient_route_report',None)))
        raise
    finally:STOP.set()


if __name__=='__main__':main()
