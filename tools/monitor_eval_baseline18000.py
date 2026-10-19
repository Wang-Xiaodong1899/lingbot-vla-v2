"""Persistent, resumable baseline18000 evaluation supervisor (4 GPUs, 8 slots)."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/results/baseline18000_clean_random_10ep'
REFERENCE = ROOT/'evaluation/results/corrector_step9500_clean10_4gpu8slots'
CURRENT = OUT/'corrector_clean'
MODEL = ROOT/'weights/lingbotvla-robotwin-clean-sft/global_step_18000'
CHECKPOINT = ROOT/'outputs/vla-corrector-base10k/train/snapshots/step16500.pt'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
SOURCE = (REFERENCE/'launcher_snapshot.sh').read_text()
TASKS = __import__('shlex').split(re.search(r'task_list_all=\((.*?)\)', SOURCE, re.S).group(1))
STAGES = [('baseline18000_clean', 'baseline', 'demo_clean', 'seen', OUT/'clean'),
          ('baseline18000_random', 'baseline', 'demo_randomized', 'unseen', OUT/'random')]
OUT.mkdir(parents=True, exist_ok=True)
LOCK = (OUT/'supervisor.lock').open('w')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
STATE_PATH = OUT/'status.json'
STATE = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {'stages': {}}
ACTIVE = None


def write(stage, status, **kw):
    STATE.update(supervisor_pid=os.getpid(), stage=stage, status=status,
                 updated_at=datetime.now(timezone.utc).isoformat(), **kw)
    tmp = STATE_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(STATE, indent=2)); os.replace(tmp, STATE_PATH)
    print(json.dumps({'stage': stage, 'status': status, **kw}), flush=True)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def find_run(output):
    runs = [p for p in output.glob('**/eval_logs') if p.is_dir()]
    return max(runs, key=lambda p: p.parent.stat().st_mtime).parent if runs else None


def inspect(run, method, instructions):
    rates, failures = {}, {}
    if run is None:
        return rates, {'run': 'not created'}
    for task in TASKS:
        log = run/'eval_logs'/f'{task}.log'
        result = run/'eval_results'/task/'_result.txt'
        if not log.exists() or not result.exists():
            failures[task] = 'incomplete'; continue
        matches = re.findall(r'Success rate: (\d+)/(\d+) => ([\d.]+)%', ANSI.sub('', log.read_text(errors='replace')))
        content = result.read_text()
        try:
            assert f'Instruction Type: {instructions}' in content
            assert matches and int(matches[-1][1]) == 10
            wins = int(matches[-1][0]); assert 0 <= wins <= 10
            assert abs(float(content.strip().splitlines()[-1]) - wins/10) < 1e-6
            rates[task] = {'successes': wins, 'episodes': 10}
        except (ValueError, AssertionError):
            failures[task] = 'invalid result or wrong split'
    logs = list((run/'inference_logs').glob('*.log'))
    if len(logs) != 8:
        failures['inference'] = f'expected 8 logs, found {len(logs)}'
    any_diagnostics = False
    for p in logs:
        t = p.read_text(errors='replace')
        if method == 'corrector':
            valid = bool(re.search(r'CORRECTOR_LOADED .*"step": 16500', t))
            any_diagnostics |= 'CORRECTOR ' in t
        else:
            valid = 'Model initialized' in t
            any_diagnostics |= 'Model initialized' in t
        if not valid:
            failures[p.name] = 'method diagnostics missing'
    if not any_diagnostics:
        failures['diagnostics'] = 'no actual method inference recorded'
    return rates, failures


def progress(run):
    episodes = wins = 0
    last = time.time() if run is None else 0
    stale_tasks = []
    if run:
        for p in (run/'eval_logs').glob('*.log'):
            last = max(last, p.stat().st_mtime)
            if time.time() - p.stat().st_mtime > 2700 and not (run/'eval_results'/p.stem/'_result.txt').exists():
                stale_tasks.append(p.stem)
            with p.open('rb') as f:
                f.seek(max(0, p.stat().st_size - 2_000_000))
                t = ANSI.sub('', f.read().decode(errors='replace'))
            r = re.findall(r'Success rate: (\d+)/(\d+) => ([\d.]+)%', t)
            if r:
                wins += int(r[-1][0]); episodes += int(r[-1][1])
        for p in (run/'inference_logs').glob('*.log'):
            last = max(last, p.stat().st_mtime)
    return dict(completed_episodes=episodes, successes=wins, total_episodes=500,
                last_log_activity=last, stale_tasks=stale_tasks)


def weights_ready():
    try:
        assert not list(MODEL.glob('*.aria2')), 'download still in progress'
        index = json.loads((MODEL/'model.safetensors.index.json').read_text())
        for name in set(index['weight_map'].values()):
            p = MODEL/name
            with p.open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                assert size < 100_000_000
                header = json.loads(f.read(size))
            end = max(v['data_offsets'][1] for k,v in header.items() if k != '__metadata__')
            assert p.stat().st_size == 8 + size + end, f'incomplete shard {name}'
        assert (MODEL/'config.json').exists()
        assert (MODEL.parent/'robotwin_clean_norm.json').exists()
        return True, 'all indexed shards complete'
    except (OSError, ValueError, AssertionError, KeyError, struct.error) as e:
        return False, str(e)


def launch(method, config, instructions, output, run):
    output.mkdir(parents=True, exist_ok=True)
    text = SOURCE
    if method == 'baseline':
        text = text.replace('eval_policy_client_vla_corrector.py', 'eval_policy_client_lingbotvla.py')
    launcher = output/'launcher_snapshot.sh'; launcher.write_text(text)
    env = os.environ.copy()
    # Use the proven current environment, including the corrector's frozen base metadata.
    current_driver = json.loads((REFERENCE/'status.json').read_text())
    if method == 'corrector':
        model = current_driver['command'][current_driver['command'].index('--model_path') + 1]
        infer = 'deploy/lingbot_vla_corrector_policy.py'
        env.update(CORRECTOR_CHECKPOINT=str(CHECKPOINT), CORRECTOR_NORM_STATS=str(Path(model)/'robotwin_clean_norm.json'))
    else:
        model = str(MODEL); infer = 'deploy/lingbot_vla_v2_policy.py'
        env.pop('CORRECTOR_CHECKPOINT', None); env.pop('CORRECTOR_NORM_STATS', None)
    env.update(CORRECTOR_INSTRUCTION_TYPE=instructions, QWEN3VL_PATH=str(ROOT/'evaluation/Qwen3-VL-4B-Instruct'),
        PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VK_ICD_FILENAMES=str(ROOT/'evaluation/nvidia_icd.json'), XDG_RUNTIME_DIR=str(ROOT/'evaluation/runtime'),
        TMPDIR=str(ROOT/'evaluation/tmp'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'data/lerobot_v21_source/src'))
    cmd = ['bash', str(launcher), '--model_path', model, '--inference_script', infer,
        '--inference_workdir', str(ROOT)+'/', '--eval_workdir', str(ROOT/'RoboTwin'), '--output_base', str(output),
        '--conda_sh', '/root/miniconda3/etc/profile.d/conda.sh', '--inference_env', 'lingbotvla', '--sim_env', 'RoboTwin',
        '--task_config', config, '--num_tasks', '50', '--num_episodes', '10', '--num_gpus', '4', '--num_per_gpu', '2',
        '--start_port', '19830', '--use_bf16', 'False', '--use_fp32', 'True', '--use_compile', 'False', '--no_video']
    if run:
        cmd += ['--resume_from', str(run)]
    log = output/f'launcher_{time.time_ns()}.log'
    with log.open('w') as f:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return child, log


def terminate(pid):
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(1)
        if alive(pid): os.kill(pid, signal.SIGKILL)


def stop(sig, frame):
    if ACTIVE: terminate(ACTIVE)
    write(STATE.get('stage'), 'interrupted', signal=sig)
    raise SystemExit(128+sig)


signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

def main():
    global ACTIVE
    for name, method, config, instructions, output in STAGES:
        if STATE['stages'].get(name, {}).get('status') == 'skipped':
            continue
        attempt = STATE['stages'].get(name, {}).get('attempts', 0)
        while True:
            run = find_run(output)
            rates, failures = inspect(run, method, instructions)
            if len(rates) == 50 and not failures:
                STATE['stages'][name] = dict(status='complete', run=str(run), successes=sum(v['successes'] for v in rates.values()), episodes=500, rates=rates, attempts=attempt)
                write(name, 'complete'); break
            # Adopt the current clean driver; subsequent stages never overlap it.
            adopted = None
            if not adopted and method == 'baseline':
                ready, reason = weights_ready()
                if not ready:
                    write(name, 'waiting_for_weights', reason=reason); time.sleep(30); continue
            child = None
            if adopted:
                ACTIVE = adopted
            else:
                attempt += 1
                child, log = launch(method, config, instructions, output, run)
                ACTIVE = child.pid
                STATE['stages'][name] = dict(status='running', attempts=attempt, launcher_pid=ACTIVE, launcher_log=str(log))
            start = time.time()
            while alive(ACTIVE):
                run = find_run(output); info = progress(run)
                STATE['stages'].setdefault(name, {}).update(status='running', run=str(run) if run else None, **info)
                write(name, 'running', active_pid=ACTIVE)
                # Long episodes may be slow; restart only after 45 minutes with no log activity.
                if time.time() - max(start, info['last_log_activity']) > 2700 or info['stale_tasks']:
                    write(name, 'restarting_stalled_run'); terminate(ACTIVE); break
                time.sleep(30)
            if child: child.wait()
            ACTIVE = None
            run = find_run(output); rates, failures = inspect(run, method, instructions)
            if len(rates) == 50 and not failures:
                continue
            STATE['stages'].setdefault(name, {}).update(status='retrying', failures=failures, attempts=attempt)
            write(name, 'retrying', reason='incomplete or failed evaluation; completed tasks will be reused')
            time.sleep(min(600, 30 * max(1, attempt)))
    write('all', 'complete', active_pid=None)

if __name__ == '__main__':
    main()
