#!/usr/bin/env python3
"""Export the current clean training loss snapshot from the rank-zero log."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log', type=Path, default=ROOT / 'outputs/robotwin-clean-from4k-bs768/train.log')
parser.add_argument('--output', type=Path, default=ROOT / 'robotwin_clean_training_loss.png')
parser.add_argument('--window', type=int, default=50)
args = parser.parse_args()
if args.window < 1:
    parser.error('--window must be positive')

pattern = re.compile(r' - Step (\d+)/\d+, Epoch (\d+), (.*)')
metrics = ['Loss', 'VLA_Loss', 'Depth_Loss', 'Future_Depth_Loss', 'FutureVideo_Loss']
rows = {}
with args.log.open(errors='replace') as log:
    for line in log:
        match = pattern.search(line)
        if not match:
            continue
        values = dict(re.findall(r'(\w+)\s+([-+\d.eE]+)(?=s?\s*(?:,|$))', match[3].strip()))
        if not all(key in values for key in metrics):
            continue
        row = {'step': int(match[1]), 'epoch': int(match[2])}
        row.update({key: float(values[key]) for key in metrics})
        rows[row['step']] = row  # Latest occurrence wins if a run was restarted.
if not rows:
    raise SystemExit('No completed training steps found in log')
data = [rows[step] for step in sorted(rows)]
steps = np.array([row['step'] for row in data])
if not all(np.isfinite(row[key]) for row in data for key in metrics):
    raise SystemExit('Non-finite loss values found; inspect the log before plotting')

csv_path = args.output.with_suffix('.csv')
with csv_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['step', 'epoch', *metrics])
    writer.writeheader()
    writer.writerows(data)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
panels = [('Total loss', [('Loss', 'Total', '#2563a6')]),
          ('Action prediction loss', [('VLA_Loss', 'VLA', '#2563a6')]),
          ('Depth alignment loss (unweighted)', [('Depth_Loss', 'Current depth', '#2563a6'),
                                                ('Future_Depth_Loss', 'Future depth', '#b96920')]),
          ('Future video alignment loss (unweighted)', [('FutureVideo_Loss', 'Future video', '#2563a6')])]
for ax, (title, series) in zip(axes.flat, panels):
    for key, label, color in series:
        raw = np.array([row[key] for row in data])
        # Trailing mean uses only actual logged steps and never crosses a gap.
        smooth = np.empty_like(raw)
        start = 0
        for i in range(len(raw)):
            if i and steps[i] != steps[i - 1] + 1:
                start = i
            smooth[i] = raw[max(start, i - args.window + 1):i + 1].mean()
        ax.plot(steps, raw, color=color, alpha=.20, linewidth=.65)
        ax.plot(steps, smooth, color=color, linewidth=1.8, label=label)
    ax.set_title(title, loc='left', fontsize=12, fontweight='semibold')
    ax.set_xlabel('New training step (initialized from step 4000 weights)')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)
    ax.set_xlim(steps[0], max(steps[-1], steps[0]+1))
    ticks = sorted(set([int(steps[0]), *np.linspace(steps[0], steps[-1], 5, dtype=int), int(steps[-1])]))
    ax.set_xticks(ticks)
    ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
    if len(series) > 1:
        ax.legend(frameon=False)
timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
fig.suptitle(f'RoboTwin clean training | 50 tasks | global batch 768\n'
             f'Steps {steps[0]}–{steps[-1]} | faint: raw loss; solid: trailing {args.window}-step mean | {timestamp}',
             fontsize=14)
fig.savefig(args.output, dpi=180, facecolor='white')
fig.savefig(args.output.with_suffix('.svg'), facecolor='white')
plt.close(fig)
print(f'Plotted {len(data)} completed steps, latest={steps[-1]}, loss={data[-1]["Loss"]:.4f}')
print(args.output)
print(csv_path)
