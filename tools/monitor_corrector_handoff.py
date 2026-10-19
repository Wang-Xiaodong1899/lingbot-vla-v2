"""Persistently monitor extraction, validate caches and hand off to four-GPU training.

The old shell controller is paused, but extraction workers are left running.
The monitor continues through training/evaluation and persists a heartbeat/status.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def process_alive(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z'
    except FileNotFoundError:
        return False


def atomic_json(path,value):
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    os.replace(temporary,path)


def scan(cache):
    records=[]
    for path in sorted(cache.glob('task*/complete.json')):
        records.append((path,json.loads(path.read_text())))
    return records


def validate_cache(cache,metadata):
    records=scan(cache)
    assert len(records)==metadata['total_episodes'] and not metadata.get('smoke_max_episodes')
    assert sum(info['frames'] for _,info in records)==metadata['total_frames']
    expected={f'task{task:02d}_episode{episode:06d}' for task,dataset in enumerate(metadata['training_data'])
              for episode in dataset['training_episodes']}
    assert {p.parent.name for p,_ in records}==expected,'Episode membership mismatch'
    for path,info in records:
        n,t,d=info['frames'],info['tokens'],info['token_dim']
        specs={'z_q':((n,t,d),np.int8),'z_scale':((n,t),np.float16),
               'actions':((n,55),np.float32),'mask':((55,),np.bool_)}
        for name,(shape,dtype) in specs.items():
            value=np.load(path.parent/f'{name}.npy',mmap_mode='r')
            assert value.shape==shape and value.dtype==dtype,(path,name,value.shape,value.dtype)
            del value
    return dict(episodes=len(records),frames=sum(info['frames'] for _,info in records),arrays_verified=4*len(records))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/vla-corrector-base10k')
    p.add_argument('--cache',type=Path,default=ROOT/'data/corrector_features_10k')
    p.add_argument('--interval',type=int,default=30)
    a=p.parse_args();a.output=a.output.resolve();a.cache=a.cache.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=a.output/'monitor_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    metadata=json.loads((a.cache/'metadata.json').read_text())
    control=json.loads((a.output/'handoff_control.json').read_text())
    old_wrapper=control['paused_wrapper_pid']
    extraction_pids=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:cmd=(entry/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if b'tasks/vla/extract_corrector_features.py' in cmd and b'python' in cmd:
            extraction_pids.append(int(entry.name))
    state.update(monitor_pid=os.getpid(),old_wrapper_pid=old_wrapper,extraction_pids=extraction_pids,
                 target_training_gpus=[0,1,2,3],interval_seconds=a.interval)
    previous_frames=None;last_progress=time.monotonic();last_report=0
    child=None

    def status(stage,**extra):
        state.update(stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
        atomic_json(state_path,state)
        if time.monotonic()-state.get('_last_print',0)>=60 or stage!=state.get('_printed_stage'):
            print(json.dumps({k:v for k,v in state.items() if not k.startswith('_')}),flush=True)
            state['_last_print']=time.monotonic();state['_printed_stage']=stage

    try:
        if state.get('new_pipeline_pid') and process_alive(state['new_pipeline_pid']):
            pipeline_pid=state['new_pipeline_pid']
        else:
            while True:
                records=scan(a.cache);frames=sum(info['frames'] for _,info in records)
                if frames!=previous_frames:
                    previous_frames=frames;last_progress=time.monotonic()
                living=[pid for pid in extraction_pids if process_alive(pid)]
                complete=len(records)==metadata['total_episodes'] and frames==metadata['total_frames']
                stage='waiting_for_extraction_exit' if complete else 'extracting_features'
                if time.monotonic()-last_progress>900:
                    stage='extraction_progress_stalled'
                status(stage,complete_episodes=len(records),total_episodes=metadata['total_episodes'],
                       complete_frames=frames,total_frames=metadata['total_frames'],live_extraction_pids=living)
                if not living:
                    if not complete:
                        raise RuntimeError(f'Extraction processes exited with incomplete cache: {len(records)} episodes, {frames} frames')
                    break
                time.sleep(a.interval)
            status('validating_complete_cache')
            validation=validate_cache(a.cache,metadata)
            atomic_json(a.output/'cache_validation.json',validation)
            # Retire the paused old shell only after its extraction workers exit.
            # SIGKILL targets just this shell; all finished workers are already gone.
            if process_alive(old_wrapper):
                os.kill(old_wrapper,signal.SIGKILL)
            pipeline_lock=(a.output/'pipeline.lock').open('w')
            for attempt in range(30):
                try:
                    fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(pipeline_lock,fcntl.LOCK_UN);break
                except BlockingIOError:time.sleep(1)
            else:raise RuntimeError('Old pipeline lock did not release')
            pipeline_lock.close()
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3','EXTRACT_CUDA_VISIBLE_DEVICES':'0,1,2,3',
                 'EXTRACT_BATCH_SIZE':'1024','SKIP_EXTRACTION':'1','MICRO_BATCH_SIZE':'128','PYTHONUNBUFFERED':'1',
                 'OUTPUT_DIR':str(a.output),'CACHE_DIR':str(a.cache)}
            with (a.output/'pipeline.log').open('a') as stream:
                stream.write('\nMonitor validated complete cache; launching four-GPU corrector training directly.\n');stream.flush()
                child=subprocess.Popen(['bash','experiment/robotwin/train_vla_corrector_10k.sh'],cwd=ROOT,env=env,
                                       stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            pipeline_pid=child.pid;state['new_pipeline_pid']=pipeline_pid
            (a.output/'pipeline.pid').write_text(str(pipeline_pid)+'\n')
            status('starting_four_gpu_training',cache_validation=validation)
        last_step=-1;last_update=time.monotonic()
        while True:
            pipeline_state=json.loads((a.output/'pipeline_status.json').read_text())
            stage=pipeline_state['stage']
            metrics=a.output/'train/metrics.jsonl';record=None
            if metrics.exists():
                for line in reversed(metrics.read_text().splitlines()):
                    try:
                        candidate=json.loads(line)
                    except json.JSONDecodeError:continue
                    if candidate.get('event')=='train':record=candidate;break
            if record:
                assert math.isfinite(record['loss']),'Nonfinite training loss'
                config=json.loads((a.output/'train/run_config.json').read_text())
                assert config['metadata']['world_size']==4,'Training did not launch with four ranks'
                if record['step']!=last_step:last_step=record['step'];last_update=time.monotonic()
                if stage=='training_corrector':stage='four_gpu_training_running'
                state.update(last_train_step=record['step'],last_train_loss=record['loss'],
                             training_world_size=4,training_global_batch=config['metadata']['batch_size']*4)
            if stage=='training_corrector' and time.monotonic()-last_update>900:
                stage='training_start_delayed'
            if not process_alive(pipeline_pid):
                code=child.poll() if child else None
                if pipeline_state['stage']=='complete' and code in (0,None):
                    status('complete',pipeline_exit_code=code);return
                raise RuntimeError(f'Pipeline exited in stage {pipeline_state["stage"]}, code={code}')
            status(stage,pipeline_pid=pipeline_pid)
            time.sleep(a.interval)
    except BaseException as exc:
        status('failed',reason=repr(exc))
        raise


if __name__=='__main__':
    main()
