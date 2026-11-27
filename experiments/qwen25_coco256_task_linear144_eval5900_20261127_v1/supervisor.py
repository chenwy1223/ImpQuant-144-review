"""One CPU owner, one GPU worker, explicit ready/go lifecycle and no retry.

The owner never imports torch. Worker self-protection/owner-heartbeat checking
remains required so a killed owner cannot leave scientific work unguarded.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import HERE,OUT,NAME,UUID,GPU_INDEX,PY,sha,save,utc,ADMISSION_DISK_GIB,RUNTIME_DISK_GIB,SCOPE_CONTRACT,completion_counts,EVAL_N,EVALUATED_ARMS,WORKER_LIMIT,BASELINE,BASELINE_ROOT,BASELINE_SHA,OLD144,OLD_ROOT,SUPERVISION_N

WAIT_LIMIT=14400
INITIAL_WORKER=0.0
INITIAL_WAIT=None
WAIT_SPENT=None
LOCK_SHA=None
LOCK_STATS={}
OWNER=None
ACTIVE=None
CLAIMED=False
STOP_REQUEST=None
EVER_SPAWNED=False
LAST_LOCK_HEARTBEAT=None
FIELDS=('pid','ppid','pgrp','start_ticks','argv')

def identity(pid):
    p=Path('/proc')/str(pid)
    raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=int(pid),ppid=int(fields[1]),pgrp=int(fields[2]),start_ticks=int(fields[19]),
        argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))

def bare(record):return {k:record[k] for k in FIELDS}

def owner_identity_matches(actual,expected):
    # The detached launcher's normal reparenting is irrelevant to owner identity.
    return bool(actual) and all(actual.get(k)==expected.get(k) for k in ['pid','pgrp','start_ticks','argv'])

def child_identity_matches(actual,expected,owner_pid):
    return bool(actual) and bare(actual)==bare(expected) and actual['ppid']==owner_pid and actual['pgrp']==actual['pid']

def expected_worker_argv():return [PY,'-u',str(HERE/'worker.py')]

def event(kind,**values):
    record=dict(event=kind,utc=utc(),attempt=NAME,**values)
    with (OUT/'supervisor_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)

def heartbeat(status):
    save(OUT/'owner_heartbeat.json',dict(attempt=NAME,utc=utc(),owner_identity=OWNER,status=status))

def interrupted(signum,frame):
    # Do not throw in the Popen-to-ACTIVE registration window. The next safe
    # boundary raises, and cleanup sees the child handle even after a signal.
    global STOP_REQUEST
    STOP_REQUEST='SUPERVISOR_INTERRUPTED_SIGNAL_'+str(int(signum))

def check_stop():
    if STOP_REQUEST:raise RuntimeError(STOP_REQUEST)


def durable_save(path,data):
    """Atomically persist a receipt before beginning a longer final hash pass."""
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    if os.name=='posix':
        fd=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def hash_with_progress(path,on_chunk=None,chunk_bytes=8*1024**2):
    """Hash without a multi-minute heartbeat gap on a large bound input."""
    digest=hashlib.sha256();done=0
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(chunk_bytes),b''):
            digest.update(block);done+=len(block);check_stop()
            if on_chunk:on_chunk(done)
    return digest.hexdigest()


def lock_progress(phase,full,completed,total,verified_bytes,total_bytes,
                  current_path=None,current_bytes=0,force_heartbeat=False):
    """Every completed binding is visible; large-file progress keeps owner live."""
    global LAST_LOCK_HEARTBEAT
    status='SCIENTIFIC_EXIT0_VERIFYING_FINAL_LOCK' if phase=='final' else 'VERIFYING_LOCK'
    record=dict(attempt=NAME,utc=utc(),phase=phase,status=status,full=full,lock_sha256=LOCK_SHA,
        bindings_completed=completed,bindings_total=total,verified_bytes=verified_bytes,
        total_bound_bytes=total_bytes,current_path=current_path,current_bytes_hashed=current_bytes,
        original_owner_identity=OWNER,protocol_complete=False)
    save(OUT/'lock_verification_progress.json',record)
    now=time.monotonic()
    if force_heartbeat or LAST_LOCK_HEARTBEAT is None or now-LAST_LOCK_HEARTBEAT>=1:
        heartbeat(status)
        checkpoint=dict(record,gpu_started=EVER_SPAWNED,scheduled_monitoring=False,
            admission_wait_elapsed_seconds=WAIT_SPENT)
        if phase=='final':
            receipt=OUT/'scientific_exit_receipt.json'
            assert receipt.is_file(),'Final hash pass requires durable scientific exit receipt'
            checkpoint['scientific_exit_receipt_sha256']=sha(receipt)
        save(OUT/'checkpoint.json',checkpoint);LAST_LOCK_HEARTBEAT=now


def verify_lock(full=True,phase='prelaunch'):
    global LAST_LOCK_HEARTBEAT
    assert LOCK_SHA and sha(HERE/'execution_lock.json')==LOCK_SHA,'Execution lock changed'
    lock=json.loads((HERE/'execution_lock.json').read_text())
    assert INITIAL_WAIT is not None and math.isfinite(INITIAL_WAIT) and 0 <= INITIAL_WAIT < WAIT_LIMIT,'Valid inherited queue wait budget'
    assert lock['initial_admission_wait_elapsed_seconds']==INITIAL_WAIT==0
    assert lock['initial_worker_elapsed_seconds']==INITIAL_WORKER==0
    assert lock['forced_RAM_reserve_waiver'] is False and lock['forced_disk_reserve_waiver'] is False
    assert lock['attempt']==NAME and lock['bindings']==len(lock['files'])
    required=dict(shared_queue_seconds=WAIT_LIMIT,admission_RAM_GiB=40,admission_disk_GiB=ADMISSION_DISK_GIB,
        runtime_RAM_GiB=8,runtime_disk_GiB=RUNTIME_DISK_GIB,allocated_GiB=22.5,worker_seconds=WORKER_LIMIT,
        auto_retry=False,scheduled_monitoring=False,**SCOPE_CONTRACT)
    for key,value in required.items():assert type(lock.get(key)) is type(value) and lock[key]==value,(key,lock.get(key),value)
    total_bytes=sum(row['bytes'] for row in lock['files']);verified_bytes=0
    # Lightweight admission verification cannot replace the admission checkpoint.
    report=full and phase in ['prelaunch','final']
    if report:
        LAST_LOCK_HEARTBEAT=None
        lock_progress(phase,full,0,len(lock['files']),0,total_bytes,force_heartbeat=True)
    for index,row in enumerate(lock['files']):
        p=Path(row['path']);st=p.stat()
        assert p.is_file() and st.st_size==row['bytes'],str(p)
        signature=(st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if full or row['bytes']<=16*1024**2:
            last_report=[time.monotonic()]
            def on_chunk(done):
                now=time.monotonic()
                if report and now-last_report[0]>=1:
                    lock_progress(phase,full,index,len(lock['files']),verified_bytes,total_bytes,str(p),done)
                    last_report[0]=now
            assert hash_with_progress(p,on_chunk=on_chunk if report else None)==row['sha256'],str(p)
            after=p.stat()
            assert (after.st_size,after.st_mtime_ns,after.st_ctime_ns)==signature,'Input changed while hashing: '+str(p)
            LOCK_STATS[str(p)]=signature
        else:assert LOCK_STATS.get(str(p))==signature,'Bound large input changed: '+str(p)
        verified_bytes+=row['bytes']
        if report:lock_progress(phase,full,index+1,len(lock['files']),verified_bytes,total_bytes,
            force_heartbeat=index+1==len(lock['files']))
        check_stop()
    return lock


def validate_scientific_marker(marker,expected):
    assert marker['attempt']==NAME and marker['worker_identity']==expected and marker['owner_identity']==OWNER
    assert marker['lock_sha256']==LOCK_SHA
    for key,value in SCOPE_CONTRACT.items():
        assert type(marker.get(key)) is type(value) and marker[key]==value,(key,marker.get(key))
    assert marker['quantized_state_sha256']==BASELINE_SHA
    assert marker['selection_sha256']==sha(OUT/'selection.json')
    assert marker['gradient_batch_sha256']==sha(OUT/'gradient_batch.json')
    assert marker['roots'][BASELINE]==BASELINE_ROOT and marker['roots'][OLD144]==OLD_ROOT
    assert set(marker['roots'])==set(EVALUATED_ARMS) and len(set(marker['roots'].values()))==3
    assert marker['physical_states']==len(EVALUATED_ARMS) and marker['evaluation_examples_per_state']==EVAL_N
    for key,value in completion_counts(len(EVALUATED_ARMS)).items():
        assert marker[key]==value,(key,marker[key],value)
    assert marker['external_judge_calls']==marker['random_fallback_calls']==0


def record_scientific_exit(expected,exit_code,wall_seconds):
    """Persist actual exit0 + existing scientific marker, never a protocol PASS."""
    assert exit_code==0,'A failed worker cannot receive a successful scientific exit receipt'
    assert 0 < wall_seconds <= WORKER_LIMIT - INITIAL_WORKER,'Scientific wall-time budget'
    marker_path=OUT/'pilot_complete.json';marker=json.loads(marker_path.read_text())
    assert marker['status']=='PASS' and marker['network_calls']==0
    assert marker['owner_handshake'] is True and marker['baseline_restored_and_verified'] is True
    validate_scientific_marker(marker,expected)
    receipt_path=OUT/'scientific_exit_receipt.json'
    assert not receipt_path.exists(),'Do not overwrite a scientific exit receipt'
    receipt=dict(attempt=NAME,utc=utc(),exit_code=exit_code,wall_seconds=wall_seconds,
        owner_identity=OWNER,worker_identity=expected,lock_sha256=LOCK_SHA,
        marker_path=str(marker_path),marker_bytes=marker_path.stat().st_size,marker_sha256=sha(marker_path),
        marker_utc=marker.get('utc'),scientific_computation_complete=True,
        protocol_complete=False,final_lock_rehash_pending_at_receipt=True,no_scientific_retry=True)
    durable_save(receipt_path,receipt)
    event('SCIENTIFIC_EXIT0_RECEIPT_DURABLY_SAVED',scientific_exit_receipt_sha256=sha(receipt_path),
        worker_identity=expected,protocol_complete=False)
    lock_progress('final',True,0,0,0,0,force_heartbeat=True)
    return receipt


def finalize_owner(verify_fn=None):
    """The expensive pass cannot hide a successful worker or fabricate COMPLETE."""
    if verify_fn is None:verify_fn=verify_lock
    receipt_path=OUT/'scientific_exit_receipt.json';receipt=json.loads(receipt_path.read_text())
    assert receipt['attempt']==NAME and receipt['exit_code']==0 and receipt['lock_sha256']==LOCK_SHA
    assert receipt['protocol_complete'] is False and receipt['scientific_computation_complete'] is True
    assert receipt['marker_sha256']==sha(OUT/'pilot_complete.json'),'Completion marker changed after exit'
    check_stop();lock_progress('final',True,0,0,0,0,force_heartbeat=True)
    lock=verify_fn(full=True,phase='final');check_stop()
    progress=json.loads((OUT/'lock_verification_progress.json').read_text())
    assert progress['phase']=='final' and progress['full'] is True
    assert progress['bindings_completed']==progress['bindings_total']==lock['bindings']
    assert progress['verified_bytes']==progress['total_bound_bytes']==sum(r['bytes'] for r in lock['files'])
    final_receipt=dict(attempt=NAME,utc=utc(),status='ALL_FINAL_EXECUTION_BINDINGS_REHASHED',
        lock_sha256=LOCK_SHA,bindings=lock['bindings'],bound_bytes=progress['verified_bytes'],
        owner_identity=OWNER,scientific_exit_receipt_sha256=sha(receipt_path),
        final_progress_sha256=sha(OUT/'lock_verification_progress.json'),full=True)
    durable_save(OUT/'final_lock_verification_receipt.json',final_receipt)
    durable_save(OUT/'checkpoint.json',dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION_AND_LOCAL_SYNC',attempt=NAME,
        utc=utc(),admission_wait_elapsed_seconds=WAIT_SPENT,gpu_started=EVER_SPAWNED,
        scientific_exit_receipt_sha256=sha(receipt_path),
        final_lock_verification_receipt_sha256=sha(OUT/'final_lock_verification_receipt.json'),
        scheduled_monitoring=False,no_scientific_retry=True))
    heartbeat('COMPLETE');event('COMPLETE_PENDING_INDEPENDENT_VERIFICATION_AND_LOCAL_SYNC',
        scientific_exit_receipt_sha256=sha(receipt_path),
        final_lock_verification_receipt_sha256=sha(OUT/'final_lock_verification_receipt.json'))

def resources():
    ram=next(int(line.split()[1])/1024**2 for line in open('/proc/meminfo') if line.startswith('MemAvailable:'))
    return dict(available_RAM_GiB=ram,disk_free_GiB=shutil.disk_usage(OUT).free/1024**3)

def gpu():
    line=subprocess.run(['nvidia-smi','--id='+UUID,'--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    fields=[v.strip() for v in line.split(',')]
    assert len(fields)==4 and fields[1]==UUID and int(fields[0])==GPU_INDEX
    rows=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
    pids=[]
    for line in rows:
        row=[v.strip() for v in line.split(',')]
        if row[0]==UUID:pids.append(int(row[1]))
    return dict(index=GPU_INDEX,uuid=UUID,used_MiB=int(fields[2]),utilization=int(fields[3]),compute_pids=pids)

def admission_allowed(r,g):
    return (r['available_RAM_GiB']>=40 and r['disk_free_GiB']>=ADMISSION_DISK_GIB and g['used_MiB']<=100
        and g['utilization']==0 and not g['compute_pids'])

def runtime_violation(r,mem,elapsed,telemetry_age=None):
    if r['available_RAM_GiB']<8:return 'HOST_RAM_BELOW_8GiB'
    if r['disk_free_GiB']<RUNTIME_DISK_GIB:return 'DISK_BELOW_40GiB'
    if mem is not None:
        values=[mem.get('allocated_GiB'),mem.get('peak_allocated_GiB')]
        if any(v is None or not math.isfinite(v) or v<0 for v in values):return 'INVALID_WORKER_MEMORY'
        if max(values)>22.5:return 'ALLOCATED_ABOVE_22_5GiB'
    if telemetry_age is not None and (not math.isfinite(telemetry_age) or telemetry_age< -5 or telemetry_age>30):return 'STALE_OR_INVALID_WORKER_TELEMETRY'
    if elapsed>120 and mem is None:return 'MISSING_WORKER_TELEMETRY'
    if elapsed + INITIAL_WORKER > WORKER_LIMIT:return 'WORKER_TIME_LIMIT_EXCEEDED'
    return None

def admit():
    global WAIT_SPENT
    stable=None;last=time.monotonic()
    while True:
        check_stop();heartbeat('WAIT_RESOURCES')
        r=resources();g=gpu();now=time.monotonic();WAIT_SPENT+=now-last;last=now
        if admission_allowed(r,g):
            if stable is None:stable=now
        else:stable=None
        save(OUT/'checkpoint.json',dict(status='WAIT_RESOURCES',attempt=NAME,utc=utc(),job='pilot',
            admission_wait_elapsed_seconds=WAIT_SPENT,admission_wait_remaining_seconds=max(0,WAIT_LIMIT-WAIT_SPENT),
            admission_RAM_GiB=40,admission_disk_GiB=ADMISSION_DISK_GIB,
            initial_admission_wait_elapsed_seconds=INITIAL_WAIT,forced_disk_reserve_waiver=False,
            forced_RAM_reserve_waiver=False,stable_seconds=0 if stable is None else now-stable,
            resources=r,gpu=g,gpu_child_launched=False))
        if WAIT_SPENT>=WAIT_LIMIT:raise RuntimeError('SHARED_4H_ADMISSION_WAIT_EXHAUSTED')
        if stable is not None and now-stable>=60:
            verify_lock(full=False);r=resources();g=gpu();now=time.monotonic();WAIT_SPENT+=now-last;last=now
            if WAIT_SPENT>=WAIT_LIMIT:raise RuntimeError('SHARED_4H_ADMISSION_WAIT_EXHAUSTED')
            if admission_allowed(r,g):
                event('RESOURCE_ADMITTED',job='pilot',resources=r,gpu=g,admission_wait_elapsed_seconds=WAIT_SPENT)
                return
            stable=None
        time.sleep(2)

def read_ready():
    p=OUT/'worker_ready.json'
    if not p.exists():return None
    return json.loads(p.read_text())

def wait_ready(child,owner_pid,argv,identity_fn=identity,ready_fn=read_ready,
               clock=time.monotonic,sleep=time.sleep,on_tick=None,timeout=5):
    """Wait for ready plus repeated /proc identity; transient pre-exec is harmless."""
    deadline=clock()+timeout;last=None;stable=0
    while clock()<=deadline:
        if child.poll() is not None:raise RuntimeError('Worker exited before ready; exit='+str(child.returncode))
        if on_tick:on_tick()
        try:actual=identity_fn(child.pid);ready=ready_fn()
        except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):actual=None;ready=None
        plausible=bool(actual and actual['pid']==child.pid and actual['ppid']==owner_pid
            and actual['pgrp']==child.pid and actual['argv']==argv and actual['start_ticks']>0)
        if plausible and ready and ready.get('attempt')==NAME and child_identity_matches(actual,ready,owner_pid):
            if last is not None and actual==last:stable+=1
            else:last=actual;stable=1
            if stable>=3:return actual
        else:last=None;stable=0
        sleep(.05)
    raise RuntimeError('Worker ready/identity handshake timed out; no go authorized')

def identity_differences(actual,expected):
    if actual is None:return {'identity':{'expected':bare(expected),'actual':None}}
    return {key:{'expected':expected.get(key),'actual':actual.get(key)} for key in FIELDS if actual.get(key)!=expected.get(key)}


def plausible_exit_sample(actual,expected):
    return actual is None or (all(actual.get(k)==expected.get(k) for k in FIELDS if k!='argv') and actual.get('argv') in [[],['']])


def observe_child(child,expected,owner_pid,identity_fn=identity,event_fn=event,
                  clock=time.monotonic,grace_seconds=.5,step_seconds=.05,on_tick=None):
    """Return actual child.wait() code or None for a verified live child.

    /proc stat and cmdline are not atomic. Missing /proc or empty argv may be
    exit transitions while poll still says None. Bound the wait/recheck; never
    infer exit0 from a marker, vanished PID, or mismatching identity.
    No signal is sent here, including for a foreign live identity.
    """
    if child.poll() is not None:return child.wait()
    try:actual=identity_fn(child.pid)
    except (FileNotFoundError,ProcessLookupError):actual=None
    if actual and child_identity_matches(actual,expected,owner_pid):return None
    differences=identity_differences(actual,expected)
    event_fn('WORKER_IDENTITY_SAMPLE_MISMATCH',worker_identity=bare(expected),actual_identity=actual,
        differing_fields=differences,grace_seconds=grace_seconds)
    if not plausible_exit_sample(actual,expected):
        event_fn('WORKER_IDENTITY_SAMPLE_UNRESOLVED',worker_identity=bare(expected),actual_identity=actual,
            differing_fields=differences,exit_code_independently_confirmed=False,reason='NONTRANSIENT_FOREIGN_IDENTITY')
        raise AssertionError('Running worker identity unresolved: nontransient foreign identity '+json.dumps(differences,sort_keys=True))
    deadline=clock()+grace_seconds;last_actual=actual
    while True:
        if on_tick:on_tick()
        remaining=max(0.,deadline-clock())
        # wait(), rather than a missing /proc record, supplies the exit code.
        try:
            code=child.wait(timeout=min(step_seconds,remaining))
        except subprocess.TimeoutExpired:code=None
        else:
            event_fn('WORKER_IDENTITY_SAMPLE_EXIT_CONFIRMED',worker_identity=bare(expected),
                last_actual_identity=last_actual,exit_code=code,exit_source='direct_child_wait')
            return code
        try:last_actual=identity_fn(child.pid)
        except (FileNotFoundError,ProcessLookupError):last_actual=None
        if last_actual and child_identity_matches(last_actual,expected,owner_pid):
            event_fn('WORKER_IDENTITY_SAMPLE_LIVE_REVERIFIED',worker_identity=bare(expected),actual_identity=last_actual)
            return None
        if clock()>=deadline or not plausible_exit_sample(last_actual,expected):
            event_fn('WORKER_IDENTITY_SAMPLE_UNRESOLVED',worker_identity=bare(expected),actual_identity=last_actual,
                differing_fields=identity_differences(last_actual,expected),exit_code_independently_confirmed=False)
            raise AssertionError('Running worker identity unresolved after bounded direct-child wait/recheck: '+json.dumps(identity_differences(last_actual,expected),sort_keys=True))


def cleanup(child,expected,owner_pid,identity_fn=identity,killpg=None,event_fn=event):
    """Only a currently reverified owned direct child group can be signaled."""
    if child.poll() is not None:
        code=child.wait();event_fn('GPU_CHILD_EXIT',job='pilot',exit_code=code,cleanup=True)
        return code
    if expected is not None:
        code=observe_child(child,expected,owner_pid,identity_fn=identity_fn,event_fn=event_fn)
        if code is not None:
            event_fn('GPU_CHILD_EXIT',job='pilot',exit_code=code,cleanup=True);return code
    # Re-read immediately before signaling, even after a successful observation.
    try:actual=identity_fn(child.pid)
    except (FileNotFoundError,ProcessLookupError):actual=None
    if actual is None or (expected is None and not actual.get('argv')):
        try:code=child.wait(timeout=.5)
        except subprocess.TimeoutExpired:raise AssertionError('Refusing unverified group termination: process identity unavailable')
        event_fn('GPU_CHILD_EXIT',job='pilot',exit_code=code,cleanup=True);return code
    if child.poll() is not None:
        code=child.wait();event_fn('GPU_CHILD_EXIT',job='pilot',exit_code=code,cleanup=True);return code
    if expected is None:
        assert actual['pid']==child.pid and actual['ppid']==owner_pid and actual['pgrp']==child.pid
        assert actual['argv']==expected_worker_argv() and actual['start_ticks']>0
        expected=actual
    assert child_identity_matches(actual,expected,owner_pid),'Refusing unverified group termination'
    event_fn('OWN_CHILD_STOP_IDENTITY_VERIFIED',identity=actual)
    if killpg is None:killpg=os.killpg
    killpg(child.pid,signal.SIGTERM)
    try:code=child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # The same exit transition handling applies before escalating a signal.
        code=observe_child(child,expected,owner_pid,identity_fn=identity_fn,event_fn=event_fn)
        if code is None:
            actual=identity_fn(child.pid)
            assert child_identity_matches(actual,expected,owner_pid),'Refusing SIGKILL after identity changed'
            killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
    event_fn('GPU_CHILD_EXIT',job='pilot',exit_code=code,cleanup=True)
    return code


def run_worker():
    global ACTIVE,EVER_SPAWNED
    assert owner_identity_matches(identity(os.getpid()),OWNER),'Owner identity changed'
    check_stop();admit();check_stop()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    argv=expected_worker_argv();start=time.monotonic();expected=None
    assert not (OUT/'worker_ready.json').exists() and not (OUT/'worker_go.json').exists(),'Do not reuse a prior handshake'
    with (OUT/'logs/pilot.log').open('xb') as f:
        child=subprocess.Popen(argv,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE=(child,None);EVER_SPAWNED=True
        def launch_tick():check_stop();heartbeat('WAIT_WORKER_READY')
        expected=wait_ready(child,os.getpid(),argv,on_tick=launch_tick)
        ACTIVE=(child,expected)
        save(OUT/'worker_pid.json',dict(attempt=NAME,job='pilot',**expected))
        event('GPU_CHILD_IDENTITY_READY_VERIFIED',job='pilot',identity=expected)
        check_stop();heartbeat('RUNNING')
        save(OUT/'worker_go.json',dict(attempt=NAME,owner_identity=OWNER,worker_identity=expected,
            lock_sha256=LOCK_SHA,authorized_utc=utc()))
        event('GPU_CHILD_LAUNCHED',job='pilot',identity=expected,ready_go_handshake=True)
        phase_token=None;last_gpu=0
        while True:
            check_stop();heartbeat('RUNNING')
            code=observe_child(child,expected,os.getpid(),on_tick=lambda:heartbeat('RUNNING'))
            if code is not None:break
            now=time.monotonic();r=resources();mem=None;age=None;phase=None
            p=OUT/'worker_phase.json'
            if p.exists():
                candidate=json.loads(p.read_text())
                if candidate.get('pid')==child.pid:
                    phase=candidate;token=(phase.get('name'),phase.get('split'),phase.get('arm'),phase.get('sample'),phase.get('trial'))
                    if token!=phase_token:phase_token=token;event('WORKER_PHASE',job='pilot',phase=phase)
            p=OUT/'worker_memory.json'
            if p.exists():
                candidate=json.loads(p.read_text())
                if candidate.get('pid')==child.pid:
                    age=time.time()-datetime.datetime.fromisoformat(candidate['utc']).timestamp();mem=candidate
            reason=runtime_violation(r,mem,now-start,age)
            record=dict(utc=utc(),attempt=NAME,job='pilot',identity=expected,resources=r,memory=mem,
                worker_memory_age_seconds=age,wall_seconds=now-start)
            with (OUT/'memory.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if now-last_gpu>=10:
                record['gpu']=gpu()
                with (OUT/'gpu.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                last_gpu=now
            save(OUT/'checkpoint.json',dict(status='RUNNING',attempt=NAME,utc=utc(),job='pilot',child=expected,
                phase=phase,wall_seconds=now-start,worker_limit_seconds=WORKER_LIMIT,
                admission_wait_elapsed_seconds=WAIT_SPENT,
                resources=r,memory=mem,gpu_child_launched=True,ready_go_handshake=True))
            if reason:event('RUNTIME_PROTECTION',reason=reason,record=record);raise RuntimeError(reason)
            time.sleep(2)
        code=child.wait();wall_seconds=time.monotonic()-start
        ACTIVE=None
        event('GPU_CHILD_EXIT',job='pilot',exit_code=code,wall_seconds=wall_seconds)
        if code!=0:raise RuntimeError('pilot failed exit='+str(code)+'; no automatic retry')
        record_scientific_exit(expected,code,wall_seconds)

def main():
    global LOCK_SHA,OWNER,CLAIMED,INITIAL_WAIT,WAIT_SPENT,INITIAL_WORKER
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU owner must hide CUDA'
    assert len(sys.argv)==2 and len(sys.argv[1])==64
    LOCK_SHA=sys.argv[1];assert sha(HERE/'execution_lock.json')==LOCK_SHA
    assert OUT.is_dir() and (OUT/'logs').is_dir()
    OWNER=identity(os.getpid());assert OWNER['pgrp']==OWNER['pid'] and OWNER['argv']==[PY,'-u',str(HERE/'supervisor.py'),LOCK_SHA]
    budget=json.loads((HERE/'execution_lock.json').read_bytes())
    INITIAL_WAIT=WAIT_SPENT=float(budget['initial_admission_wait_elapsed_seconds'])
    INITIAL_WORKER=float(budget['initial_worker_elapsed_seconds'])
    assert json.loads((OUT/'launched_once.json').read_bytes())['argv'] == OWNER['argv']
    assert not (OUT/'worker_pid.json').exists()
    with (OUT/'launcher_pid.json').open('x') as f:json.dump(dict(attempt=NAME,lock_sha256=LOCK_SHA,**OWNER),f,indent=2)
    CLAIMED=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    heartbeat('VERIFYING_LOCK');lock=verify_lock(full=True,phase='prelaunch');check_stop()
    cpu = json.loads((OUT/'cpu_preflight.json').read_bytes())
    assert cpu['status'] == 'PASS' and cpu['cuda_initialized'] is False and cpu['model_calls'] == 0
    assert cpu['baseline_and_grids_verified'] is True and cpu['data_seal_verified'] is True
    assert cpu['supervision_examples']==SUPERVISION_N and cpu['eval_examples']==EVAL_N
    assert cpu['reused_outcomes']==2*EVAL_N and cpu['supervision_eval_pixel_overlap']==0
    for filename in ('cpu_preflight.json','supervision_manifest.json','data_manifest.json','reference_outcomes.json','source_snapshot.json'):
        assert sha(OUT/filename) == next(row['sha256'] for row in lock['files']
                                         if row['path'] == str(OUT/filename))
    event('LOCK_AND_CPU_PREFLIGHT_VERIFIED',bindings=len(lock['files']),lock_sha256=LOCK_SHA)
    run_worker()
    finalize_owner()
    # This CPU verification is part of the finite experiment pipeline.
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (OUT/'independent_verify.log').open('xb') as log:
        status=subprocess.run([PY,'-u',str(HERE/'audit.py')],env=environment,
                              cwd=HERE,stdout=log,stderr=subprocess.STDOUT).returncode
    if status!=0:
        event('INFERENCE_COMPLETE_INDEPENDENT_VERIFICATION_FAILED',exit_code=status)
        return
    event('INDEPENDENT_VERIFICATION_PASS',receipt_sha256=sha(OUT/'independent_verified.json'))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        stopped=True
        if ACTIVE is not None:
            try:cleanup(ACTIVE[0],ACTIVE[1],os.getpid())
            except BaseException as stop_exc:
                stopped=False;event('OWN_CHILD_STOP_NOT_CONFIRMED',error_type=type(stop_exc).__name__,error=str(stop_exc))
        if CLAIMED:
            save(OUT/'checkpoint.json',dict(status='FAILED_TERMINAL_NO_AUTO_RETRY',attempt=NAME,utc=utc(),
                error_type=type(exc).__name__,error=str(exc),admission_wait_elapsed_seconds=WAIT_SPENT,
                gpu_started=EVER_SPAWNED,own_child_stop_confirmed=stopped,scheduled_monitoring=False))
            heartbeat('FAILED');event('FAILED_TERMINAL_NO_AUTO_RETRY',error_type=type(exc).__name__,error=str(exc),own_child_stop_confirmed=stopped)
        raise
