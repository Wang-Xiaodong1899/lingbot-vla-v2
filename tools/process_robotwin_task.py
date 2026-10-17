#!/usr/bin/env python3
"""Run RoboTwin's original HDF5 preprocessing with bounded CPU threads."""
import os
from pathlib import Path
import runpy
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import cv2
cv2.setNumThreads(1)

root = Path(__file__).resolve().parents[1]
workdir = root / 'RoboTwin/policy/pi0'
os.chdir(workdir)
sys.argv = ['scripts/process_data.py', *sys.argv[1:]]
runpy.run_path(str(workdir / 'scripts/process_data.py'), run_name='__main__')
