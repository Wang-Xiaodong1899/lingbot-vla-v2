#!/usr/bin/env python3
"""Use RoboTwin's converter with a v2.1 LeRobot API and bounded image writers."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('--image-threads', type=int, default=2)
args = parser.parse_args()
name = f'{args.task}-aloha-agilex_clean_50-50'
output_base = root / 'data/lerobot_clean50'
output = output_base / name
if (output / '.conversion_complete.json').exists():
    print(f'[already complete] {output}', flush=True)
    raise SystemExit(0)
if output.exists():
    raise RuntimeError(f'Refusing to overwrite existing dataset: {output}')

staging_base = output_base / '.building'
staging = staging_base / name
if staging.exists():
    staging.rename(staging.with_name(name + f'.interrupted_{time.time_ns()}'))
staging_base.mkdir(parents=True, exist_ok=True)
os.environ['HF_LEROBOT_HOME'] = str(staging_base)
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(root / 'data/lerobot_v21_source/src'))

import cv2
import numpy as np
import torch
cv2.setNumThreads(1)
torch.set_num_threads(1)
np.random.seed(42)
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
assert CODEBASE_VERSION == 'v2.1', CODEBASE_VERSION

# Preserve the official camera decoding, state/action mapping, instruction
# sampling, feature definitions and 50 Hz rate. Only adapt the import namespace.
source_path = root / 'RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py'
source = source_path.read_text().replace(
    'from lerobot.common.datasets.lerobot_dataset import ',
    'from lerobot.datasets.lerobot_dataset import ',
).replace('import tyro\n', '')
source = source.replace(
    'dataset.add_frame(frame)',
    'dataset.add_frame({key: value for key, value in frame.items() if key != "task"}, task=frame["task"])',
)
module = types.ModuleType('robotwin_official_converter')
module.__file__ = str(source_path)
sys.modules[module.__name__] = module
exec(compile(source, str(source_path), 'exec'), module.__dict__)

processed = root / 'RoboTwin/policy/pi0/processed_data' / name
while not (processed / '.preprocess_complete.json').exists():
    print(f'[waiting for preprocessing] {args.task}', flush=True)
    time.sleep(5)
episodes = [processed / f'episode_{i}/episode_{i}.hdf5' for i in range(50)]
assert all(path.is_file() for path in episodes)
config = module.DatasetConfig(use_videos=False, image_writer_processes=0,
                              image_writer_threads=args.image_threads)
dataset = module.create_empty_dataset(name, robot_type='aloha', mode='image',
                                      dataset_config=config)
try:
    module.populate_dataset(dataset, episodes, task=args.task)
finally:
    dataset.stop_image_writer()
info = json.loads((staging / 'meta/info.json').read_text())
assert info['codebase_version'] == 'v2.1' and info['total_episodes'] == 50
assert len(list((staging / 'data').glob('chunk-*/episode_*.parquet'))) == 50
(staging / '.conversion_complete.json').write_text(json.dumps({
    'task': args.task, 'episodes': 50, 'frames': info['total_frames'],
    'codebase_version': 'v2.1', 'fps': info['fps'],
    'converter': str(source_path), 'lerobot_version': '0.3.3',
}, indent=2) + '\n')
staging.rename(output)
print(f'[complete] {output}: 50 episodes, {info["total_frames"]} frames', flush=True)
