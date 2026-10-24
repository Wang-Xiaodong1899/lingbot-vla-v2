"""Causal, per-view/per-patch temporal adapter and timestamped inference cache."""
import math
from collections import deque
import torch
from torch import nn

HISTORY_SECONDS = (0.5, 1.0)


def history_frame_offsets(fps):
    """Use the latest frame at or before each requested time (never later)."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    return tuple(-math.ceil(seconds * fps - 1e-8) for seconds in HISTORY_SECONDS)


class PatchTemporalMemory(nn.Module):
    def __init__(self, width, bottleneck=256, heads=4):
        super().__init__()
        if bottleneck <= 0 or heads <= 0 or bottleneck % heads:
            raise ValueError('bottleneck must be positive and divisible by heads')
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.time = nn.Linear(1, bottleneck, bias=False)
        self.attention = nn.MultiheadAttention(bottleneck, heads, dropout=0., batch_first=True)
        self.up = nn.Linear(bottleneck, width)
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.down.reset_parameters()
        self.time.reset_parameters()
        self.attention._reset_parameters()
        self.attention.out_proj.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, current, history, valid, ages):
        # current B,V,P,D; history B,2,V,P,D; valid B,2,V; ages B,2 seconds.
        b, v, p, d = current.shape
        if history.shape != (b, 2, v, p, d) or valid.shape != (b, 2, v) or ages.shape != (b, 2):
            raise ValueError('invalid temporal input shapes')
        if not torch.isfinite(ages).all() or (ages <= 0).any():
            raise ValueError('history ages must be positive finite seconds')
        # Remove invalid entries BEFORE projection so even NaN padding cannot leak.
        history = torch.where(valid[..., None, None].bool(), history, torch.zeros_like(history))
        tokens = torch.cat((current[:, None], history), dim=1)
        x = self.down(self.norm(tokens))
        offsets = torch.cat((torch.zeros_like(ages[:, :1]), -ages), dim=1)
        x = x + self.time(offsets.to(x.dtype)[..., None])[:, :, None, None, :]
        x = x.permute(0, 2, 3, 1, 4).reshape(b*v*p, 3, -1)
        mask = torch.cat((torch.ones(b, 1, v, device=current.device, dtype=torch.bool), valid.bool()), dim=1)
        mask = ~mask.permute(0, 2, 1)[:, :, None, :].expand(b, v, p, 3).reshape(b*v*p, 3)
        y, _ = self.attention(x[:, :1], x, x, key_padding_mask=mask, need_weights=False)
        delta = self.up(y[:, 0]).reshape(b, v, p, d)
        # No available history means exact baseline even after adapter training.
        active = valid.any(dim=1)[:, :, None, None]
        return current + torch.where(active, delta, torch.zeros_like(delta))


class TemporalFeatureCache:
    """One episode/stream per instance. Call clear on reset; timestamps are env seconds.

    Missing/sparse observations are masked, never replaced with a newer frame.
    max_age_error limits how far before t-offset a cached observation may be.
    """
    def __init__(self, max_age_error=0.1, max_entries=256):
        if not math.isfinite(max_age_error) or max_age_error < 0 or max_entries < 3:
            raise ValueError('invalid cache limits')
        self.max_age_error = max_age_error
        self.entries = deque(maxlen=max_entries)
        self.episode = None
        self.encoded = {}
        self.current_encoding = None

    def clear(self):
        self.entries.clear()
        self.episode = None
        self.encoded = {}
        self.current_encoding = None

    def begin_episode(self, episode):
        if episode is None:
            raise ValueError('episode_id required')
        if self.episode != episode:
            self.clear()
            self.episode = episode

    def get_encoded(self, timestamp):
        """Exact frame lookup; observations at a timestamp must be immutable."""
        return self.encoded.get(float(timestamp))

    def put_encoded(self, timestamp, features, view_mask, grid, deepstack=None):
        t = float(timestamp)
        if not math.isfinite(t) or t < 0:
            raise ValueError('finite nonnegative environment timestamp required')
        self.encoded[t] = (features.detach(), view_mask.detach().bool(), grid.detach().clone())
        if deepstack is not None:
            # Only the latest current frame needs DeepStack, never history.
            self.current_encoding = (t, features.detach(), [x.detach() for x in deepstack])
        newest = max(self.encoded)
        for old in list(self.encoded):
            if newest-old > 1.+self.max_age_error+1e-6:
                del self.encoded[old]
        while len(self.encoded) > self.entries.maxlen:
            del self.encoded[min(self.encoded)]

    def push(self, timestamp, episode, features, view_mask):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0 or episode is None:
            raise ValueError('finite nonnegative environment timestamp and episode_id required')
        if self.episode != episode:
            self.clear()
            self.episode = episode
        if self.entries and timestamp < self.entries[-1][0]:
            raise ValueError('timestamps must be monotonic within an episode; reset on episode change')
        if self.entries and timestamp == self.entries[-1][0]:
            self.entries.pop()
        self.entries.append((timestamp, features.detach(), view_mask.detach().bool()))
        while self.entries and timestamp-self.entries[0][0] > 1.+self.max_age_error+1e-6:
            self.entries.popleft()

    def query(self, timestamp, reference, view_mask):
        history, masks, ages = [], [], []
        for offset in HISTORY_SECONDS:
            target = float(timestamp)-offset
            entry = next((x for x in reversed(self.entries) if x[0] <= target+1e-7), None)
            usable = entry is not None and target-entry[0] <= self.max_age_error+1e-7
            usable = usable and entry[1].shape == reference.shape
            history.append(entry[1] if usable else torch.zeros_like(reference))
            masks.append((entry[2] & view_mask.bool()) if usable else torch.zeros_like(view_mask, dtype=torch.bool))
            ages.append(float(timestamp)-entry[0] if usable else offset)
        return torch.stack(history), torch.stack(masks), reference.new_tensor(ages, dtype=torch.float32)
