"""Episode-local AHS state in the model's active normalized action coordinates."""
import os
import sys
from pathlib import Path
import numpy as np

root = Path(os.environ.get('CHUNKTRUST_ROOT', Path(__file__).resolve().parents[2] / 'ChunkTrust'))
sys.path.insert(0, str(root / 'src'))
from chunktrust import AHS, AHSConfig


class ChunkTrustRuntime:
    def __init__(self, horizon=50, seed=0):
        self.selector = AHS(AHSConfig(horizon=horizon, action_dim=14), seed=seed)
        self.history = np.empty((0, 14), dtype=np.float32)
        self.pending = None

    def acknowledge(self, count):
        if not isinstance(count, (int, np.integer)):
            raise ValueError('executed_count must be an integer')
        maximum = 0 if self.pending is None else len(self.pending)
        if not 0 <= count <= maximum:
            raise ValueError(f'executed_count {count} outside [0, {maximum}]')
        if self.pending is not None:
            self.history = np.concatenate([self.history, self.pending[:count]])[-60:]
        self.pending = None

    def select(self, velocity, actions):
        if self.pending is not None:
            raise ValueError('Previous prefix needs executed_count acknowledgement')
        k, info = self.selector.select(velocity, actions, self.history)
        k = int(k)
        self.pending = np.asarray(actions[:k], dtype=np.float32).copy()
        return k, info
