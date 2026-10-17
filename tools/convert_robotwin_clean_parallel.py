#!/usr/bin/env python3
"""Convert 50 RoboTwin tasks concurrently and publish a ready-to-train path list."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--image-threads', type=int, default=2)
args = parser.parse_args()
raw = root / 'data/RoboTwin2.0/dataset'
tasks = sorted(path.name for path in raw.iterdir() if (path / 'aloha-agilex_clean_50').is_dir())
assert len(tasks) == 50, len(tasks)
logs = root / 'data/conversion_logs'
logs.mkdir(exist_ok=True)
manifest = root / 'assets/training_data/robotwin_clean50_local.txt'
complete = {}
failed = []
started = time.monotonic()

def save_progress():
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(''.join(f'robotwin {complete[task]["path"]}\n' for task in sorted(complete)))
    temporary.replace(manifest)
    status = logs / 'lerobot_status.json'
    temporary = status.with_suffix('.tmp')
    temporary.write_text(json.dumps({'total_tasks': len(tasks), 'completed_tasks': len(complete),
        'completed_episodes': len(complete) * 50, 'failed': failed,
        'ready_manifest': str(manifest), 'completed': complete,
        'elapsed_seconds': round(time.monotonic() - started, 1)}, indent=2) + '\n')
    temporary.replace(status)

def convert(task):
    with (logs / f'{task}.lerobot.log').open('w') as log:
        subprocess.run([sys.executable, '-u', str(root / 'tools/convert_robotwin_task_lerobot.py'),
                        task, '--image-threads', str(args.image_threads)], check=True,
                       stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PYTHONNOUSERSITE': '1'})
    output = root / 'data/lerobot_clean50' / f'{task}-aloha-agilex_clean_50-50'
    result = json.loads((output / '.conversion_complete.json').read_text())
    result['path'] = str(output)
    return result

if __name__ == '__main__':
    print(f'LeRobot v2.1 conversion: {len(tasks)} tasks, {args.workers} workers, '
          f'{args.image_threads} image threads per worker', flush=True)
    save_progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                complete[task] = future.result()
                print(f'[LeRobot {len(complete)}/{len(tasks)}] {task}: '
                      f'{complete[task]["frames"]} frames, 50 episodes', flush=True)
            except Exception as error:
                failed.append({'task': task, 'error': str(error)})
                print(f'[failed] {task}: {error}', flush=True)
            save_progress()
    print(f'Finished: {len(complete)}/{len(tasks)} tasks; training list: {manifest}', flush=True)
    if failed:
        raise SystemExit(1)
