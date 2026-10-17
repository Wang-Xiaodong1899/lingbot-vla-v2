#!/usr/bin/env python3
"""Parallel RoboTwin clean-50 HDF5 preprocessing, one isolated output per task."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=16)
args = parser.parse_args()
raw = root / 'data/RoboTwin2.0/dataset'
tasks = sorted(p.name for p in raw.iterdir() if (p / 'aloha-agilex_clean_50').is_dir())
logs = root / 'data/conversion_logs'
logs.mkdir(exist_ok=True)

def convert(task):
    name = f'{task}-aloha-agilex_clean_50-50'
    output = root / 'RoboTwin/policy/pi0/processed_data' / name
    marker = output / '.preprocess_complete.json'
    if not marker.exists():
        with (logs / f'{task}.preprocess.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(root / 'tools/process_robotwin_task.py'),
                            task, 'aloha-agilex_clean_50', '50'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        episodes = list(output.glob('episode_*/episode_*.hdf5'))
        if len(episodes) != 50:
            raise RuntimeError(f'{task}: expected 50 episodes, found {len(episodes)}')
        marker.write_text(json.dumps({'task': task, 'episodes': 50}) + '\n')
    training = root / 'RoboTwin/policy/pi0/training_data/demo_clean' / name
    if not training.exists():
        training.symlink_to(output, target_is_directory=True)
    return task

if __name__ == '__main__':
    print(f'Preprocessing {len(tasks)} tasks with {args.workers} workers', flush=True)
    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                done += 1
                print(f'[preprocess {done}/{len(tasks)}] {task} done', flush=True)
            except Exception as error:
                failures.append({'task': task, 'error': str(error)})
                print(f'[failed] {task}: {error}', flush=True)
    (logs / 'preprocess_status.json').write_text(json.dumps(
        {'total': len(tasks), 'completed': done, 'failed': failures}, indent=2) + '\n')
    if failures:
        raise SystemExit(1)
