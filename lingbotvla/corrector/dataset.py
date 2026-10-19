"""Memory mapped, episode-separated visual dynamics cache."""
from collections import OrderedDict
import bisect
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .runtime import dequantize_tokens


class DynamicsDataset(Dataset):
    def __init__(self, episodes, k_step=10):
        self.episodes = [Path(p) for p in episodes]
        self.k_step = k_step
        self.ends, self.cache = [], OrderedDict()
        total = 0
        for path in self.episodes:
            frames = json.loads((path / 'complete.json').read_text())['frames']
            count = frames - k_step
            if count <= 0:
                raise ValueError(f'Episode too short: {path}')
            total += count
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        frame = index - (self.ends[episode - 1] if episode else 0)
        if episode not in self.cache:
            root = self.episodes[episode]
            self.cache[episode] = {k: np.load(root / f'{k}.npy', mmap_mode='r')
                                   for k in ['z_q', 'z_scale', 'actions', 'mask']}
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        arrays = self.cache[episode]
        z = dequantize_tokens(arrays['z_q'][frame], arrays['z_scale'][frame])
        future = dequantize_tokens(arrays['z_q'][frame + self.k_step], arrays['z_scale'][frame + self.k_step])
        action = np.array(arrays['actions'][frame], dtype=np.float32, copy=True) * arrays['mask']
        return dict(z=torch.from_numpy(z), action=torch.from_numpy(action), delta=torch.from_numpy(future - z))


def split_episodes(cache, val_episodes_per_task=5, seed=42, allow_smoke=False):
    cache = Path(cache)
    metadata = json.loads((cache / 'metadata.json').read_text())
    paths = sorted(p.parent for p in cache.glob('task*/complete.json'))
    if not allow_smoke and (metadata.get('smoke_max_episodes') or len(paths) != metadata['total_episodes']):
        raise ValueError(f'Feature extraction is incomplete: {len(paths)}/{metadata["total_episodes"]} episodes')
    by_task = {}
    for path in paths:
        task = json.loads((path / 'complete.json').read_text())['task']
        by_task.setdefault(task, []).append(path)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for task, episodes in sorted(by_task.items()):
        order = rng.permutation(len(episodes))
        n = min(val_episodes_per_task, max(1, len(episodes) // 10))
        if len(episodes) < 2:
            raise ValueError(f'Need at least two cached episodes for task {task}')
        val.extend(episodes[i] for i in order[:n])
        train.extend(episodes[i] for i in order[n:])
    return train, val, metadata
