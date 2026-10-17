"""Compute clean-only stats with training-equivalent 50-frame action windows."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import yaml
from lingbotvla.utils.normalize import NormStats, save

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / 'assets/training_data/robotwin_clean50_local.txt'
output = ROOT / 'assets/norm_stats/robotwin_clean50.json'
robot_file = ROOT / 'configs/robot_configs/robotwin.yaml'
robot = yaml.safe_load(robot_file.read_text())
chunk_size = 50
entries = [line.split() for line in manifest.read_text().splitlines() if line.strip()]
assert len(entries) == 50 and len({p for _, p in entries}) == 50
assert all(r == 'robotwin' and '_clean_' in p and 'randomized' not in p for r, p in entries)
# Each action appears once for each start frame whose window covers it;
# beyond-episode timestamps are clamped to the final frame, as in LeRobot.
for length in [1, 7, 49, 50, 75]:
    ids = np.minimum(np.arange(length)[:, None] + np.arange(chunk_size), length - 1)
    expected = np.bincount(ids.ravel(), minlength=length)
    weights = np.minimum(np.arange(length) + 1, chunk_size)
    weights[-1] += length * chunk_size - weights.sum()
    np.testing.assert_array_equal(weights, expected)

values = {next(iter(entry)): [] for group in ['states', 'actions'] for entry in robot[group]}
records = []
total_frames = total_episodes = 0
for task_index, (_, path) in enumerate(entries):
    path = Path(path)
    info = json.loads((path / 'meta/info.json').read_text())
    assert info['total_episodes'] == 50
    files = sorted(path.glob('data/chunk-*/*.parquet'))
    assert len(files) == 50
    task_frames = 0
    for file in files:
        table = pq.read_table(file, columns=['observation.state', 'action', 'episode_index'])
        assert len(set(table['episode_index'].to_pylist())) == 1
        raw = {key: np.asarray(table[key].to_pylist(), dtype=np.float64) for key in ['observation.state', 'action']}
        n = len(table)
        assert n > 0 and raw['action'].shape == raw['observation.state'].shape == (n, 14)
        weights = np.minimum(np.arange(n) + 1, chunk_size)
        weights[-1] += n * chunk_size - weights.sum()
        for group in ['states', 'actions']:
            for entry in robot[group]:
                key, mapping = next(iter(entry.items()))
                assert not mapping.get('subtract_state', False)
                parts = []
                for part in mapping['origin_keys']:
                    source, bounds = next(iter(part.items()))
                    parts.append(raw[source][:, bounds['start']:bounds['end']])
                array = np.concatenate(parts, axis=-1)
                assert np.isfinite(array).all(), (file, key)
                if group == 'actions':
                    array = np.repeat(array, weights, axis=0)
                values[key].append(array)
        task_frames += n
    assert task_frames == info['total_frames']
    total_frames += task_frames
    total_episodes += len(files)
    records.append({'dataset': str(path), 'episodes': len(files), 'frames': task_frames})
    print(f'Read {task_index+1}/50 datasets; {total_frames} frames', flush=True)

stats = {}
counts = {}
for key, arrays in values.items():
    array = np.concatenate(arrays, axis=0)
    counts[key] = len(array)
    assert len(array) == total_frames * (chunk_size if key.startswith('action.') else 1)
    q01, q02, q98, q99 = np.quantile(array, [0.01, 0.02, 0.98, 0.99], axis=0)
    stats[key] = NormStats(mean=array.mean(axis=0), std=array.std(axis=0), min=array.min(axis=0), max=array.max(axis=0), q01=q01, q02=q02, q98=q98, q99=q99)
    print('Computed', key, array.shape, flush=True)
    del array
# The existing loader expects norm_stats + count; count is the state-frame count.
save(output, stats, total_frames)
provenance = {'dataset_list': str(manifest.relative_to(ROOT)), 'dataset_list_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'robot_config': str(robot_file.relative_to(ROOT)), 'robot_config_sha256': hashlib.sha256(robot_file.read_bytes()).hexdigest(), 'tasks': len(entries), 'episodes': total_episodes, 'frames': total_frames, 'chunk_size': chunk_size, 'data_ratio': 1.0, 'counts_by_feature': counts, 'action_rule': 'Absolute actions; all overlapping 50-frame windows, clamped at episode end; action chunk dimension merged, including padded entries, matching scripts/compute_norm_stats.py.', 'quantiles': 'Exact numpy.quantile with linear interpolation (repository script uses approximate running histograms).', 'datasets': records}
output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print('SAVED', output, flush=True)
