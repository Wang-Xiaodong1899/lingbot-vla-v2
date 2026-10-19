"""Wait for idle GPUs; validate corrector smoke, then clean/randomized 10 episodes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
ANSI=re.compile(r'\x1b\[[0-9;]*m')


def validate_results(run, tasks, episodes, instructions):
    rates, failures={},{}
    if run is None:
        return rates,dict(launcher='No run directory was created')
    for task in tasks:
        result=run/'eval_results'/task/'_result.txt'
        log=run/'eval_logs'/f'{task}.log'
        if not result.exists() or not log.exists():
            failures[task]='Missing final result or task log'
            continue
        content=result.read_text()
        progress=re.findall(r'Success rate: (\d+)/(\d+) => ([\d.]+)%',ANSI.sub('',log.read_text(errors='replace')))
        if f'Instruction Type: {instructions}' not in content or not progress or int(progress[-1][1])!=episodes:
            failures[task]='Wrong instruction split or incomplete episode count'
            continue
        rate=float(content.strip().splitlines()[-1])
        if abs(rate-int(progress[-1][0])/episodes)>1e-6:
            failures[task]='Result and final success count disagree'
            continue
        rates[task]=dict(successes=int(progress[-1][0]),episodes=episodes,rate=rate)
    # Require actual corrector server startup and finite diagnostics.
    logs=list((run/'inference_logs').glob('*.log'))
    for path in logs:
        text=path.read_text(errors='replace')
        if 'CORRECTOR_LOADED ' not in text or 'CORRECTOR ' not in text:
            failures[path.name]='Corrector startup or per-step diagnostics absent'
    if not logs:
        failures['inference']='No inference logs'
    return rates,failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'evaluation/results/vla_corrector_10k')
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--start-port',type=int,default=9830)
    p.add_argument('--min-free-mib',type=int,default=75000)
    p.add_argument('--tasks',type=int,default=50)
    p.add_argument('--episodes',type=int,default=10)
    a=p.parse_args()
    devices=[int(v) for v in a.gpus.split(',')]
    if not devices or not 1<=a.tasks<=50 or a.episodes<1:
        p.error('Invalid device/task/episode selection')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # CPU checkpoint validation; do not allocate GPU memory while waiting.
    from lingbotvla.corrector.runtime import load_corrector, sha256
    model,payload=load_corrector(a.checkpoint)
    del model
    if payload['metadata']['smoke']:
        raise ValueError('Refuse formal evaluation with smoke model')
    base=payload['metadata']['feature_metadata']
    state_path=a.output/'status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(stages={})
    digest=sha256(a.checkpoint)
    if state.get('checkpoint_sha256',digest)!=digest:
        raise ValueError('Corrector changed; use a different evaluation output')
    state.update(driver_pid=os.getpid(),checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest,
                 corrector_step=payload['step'],gpus=devices,episodes=a.episodes,tasks=a.tasks)
    active=None
    def status(stage,value,**details):
        state.update(stage=stage,status=value,updated_at=datetime.now(timezone.utc).isoformat(),**details)
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));os.replace(temp,state_path)
        print(json.dumps(dict(stage=stage,status=value,**details)),flush=True)
    def stop(sig,_):
        if active is not None:
            active.terminate()
            try:active.wait(timeout=45)
            except subprocess.TimeoutExpired:active.kill()
        status(state.get('stage','driver'),'interrupted',signal=sig)
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    source=(ROOT/'experiment/robotwin/start_robotwin_infer_and_eval.sh').read_text()
    tasks=shlex.split(re.search(r'task_list_all=\((.*?)\)',source,re.S).group(1))[:a.tasks]
    # Snapshot the launcher to avoid altering concurrent experiments; explicit splits and GPU map.
    launcher=a.output/'launcher_snapshot.sh'
    text=source.replace('export CUDA_VISIBLE_DEVICES=${gpu_id}',
                        'gpu_map=('+ ' '.join(map(str,devices)) +')\n    export CUDA_VISIBLE_DEVICES=${gpu_map[$gpu_id]}')
    text=text.replace('--task_config ${task_config} \\', '--task_config ${task_config} \\\n        --instruction_type ${CORRECTOR_INSTRUCTION_TYPE} \\')
    text=text.replace('eval_policy_client_lingbotvla.py"','eval_policy_client_vla_corrector.py"')
    # Distinct client filename prevents overwriting a different active evaluation client.
    launcher.write_text(text)
    env=os.environ.copy()
    env.update(CORRECTOR_CHECKPOINT=str(a.checkpoint.resolve()),CORRECTOR_NORM_STATS=base['norm_stats'],
               QWEN3VL_PATH=base['qwen_path'],PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
               VK_ICD_FILENAMES=str(ROOT/'evaluation/nvidia_icd.json'),
               XDG_RUNTIME_DIR=str(ROOT/'evaluation/runtime'),TMPDIR=str(ROOT/'evaluation/tmp'),
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'data/lerobot_v21_source/src'))
    for key in ['XDG_RUNTIME_DIR','TMPDIR']:
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    stages=[('smoke','demo_clean','seen',tasks[:1],1),
            ('clean','demo_clean','seen',tasks,a.episodes),
            ('randomized','demo_randomized','unseen',tasks,a.episodes)]
    try:
        for name,config,instructions,expected,episodes in stages:
            previous=state['stages'].get(name,{})
            run=Path(previous['run']) if previous.get('run') else None
            if previous.get('status')=='complete':
                rates,failures=validate_results(run,expected,episodes,instructions)
                if not failures and len(rates)==len(expected):
                    continue
            while True:
                priority_queue=ROOT/'evaluation/results/queued_corrector9500_autohorizon18000/status.json'
                if priority_queue.exists():
                    queue_state=json.loads(priority_queue.read_text())
                    if queue_state.get('status')!='complete' or queue_state.get('stage')!='all':
                        status(name,'waiting_for_requested_evaluation_sequence',queue=str(priority_queue))
                        time.sleep(30)
                        continue
                query=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                free=[int(v) for v in query.splitlines()]
                if min(free[i] for i in devices)>=a.min_free_mib:
                    break
                status(name,'waiting_for_gpus',free_mib=free,minimum_free_mib=a.min_free_mib)
                time.sleep(30)
            env['CORRECTOR_INSTRUCTION_TYPE']=instructions
            # Reuse only same-checkpoint/same-stage completed tasks; launcher re-runs partial tasks.
            command=['bash',str(launcher),'--model_path',base['base_model'],
                     '--inference_script','deploy/lingbot_vla_corrector_policy.py',
                     '--inference_workdir',str(ROOT)+'/', '--eval_workdir',str(ROOT/'RoboTwin'),
                     '--output_base',str(a.output/name),'--conda_sh','/root/miniconda3/etc/profile.d/conda.sh',
                     '--inference_env','lingbotvla','--sim_env','RoboTwin','--task_config',config,
                     '--num_tasks',str(len(expected)),'--num_episodes',str(episodes),
                     '--num_gpus',str(len(devices)),'--num_per_gpu','1','--start_port',str(a.start_port),
                     '--use_bf16','False','--use_fp32','True','--use_compile','False','--no_video']
            if run and (run/'eval_logs').exists():
                command+=['--resume_from',str(run)]
            log=a.output/f'{name}_launcher_{time.time_ns()}.log'
            with log.open('w') as f:
                active=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,
                                        stderr=subprocess.STDOUT,start_new_session=True)
                status(name,'running',launcher_pid=active.pid,launcher_log=str(log),command=command)
                code=active.wait();active=None
            contents=ANSI.sub('',log.read_text(errors='replace'))
            directories=re.findall(r'Run directory: (.+)',contents)
            if directories:run=Path(directories[-1].strip())
            rates,failures=validate_results(run,expected,episodes,instructions)
            # Smoke also checks the OGG autograd branch on a real base/real observation beforehand.
            success=code==0 and not failures and len(rates)==len(expected)
            state['stages'][name]=dict(status='complete' if success else 'failed',run=str(run) if run else None,
                                       rates=rates,failures=failures,exit_code=code,instruction_type=instructions,
                                       tasks=len(rates),expected_tasks=len(expected),episodes_per_task=episodes,
                                       successes=sum(v['successes'] for v in rates.values()),
                                       episodes=sum(v['episodes'] for v in rates.values()))
            status(name,state['stages'][name]['status'])
            if not success:
                raise RuntimeError(f'{name} evaluation incomplete; inspect {log}')
        status('all','complete')
    except Exception as e:
        status(state.get('stage','driver'),'failed',reason=repr(e));raise


if __name__=='__main__':
    main()
