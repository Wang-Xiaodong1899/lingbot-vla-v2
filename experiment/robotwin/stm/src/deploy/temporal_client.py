"""Client-side raw-frame buffer for action-chunk policies.

Call observe() at EACH camera/environment observation, even while executing an
open-loop action chunk. infer() sends only the two selected history observations
and the current observation in one request using the websocket client.
This avoids running ViT on every intermediate control frame.
"""
from collections import deque
import math


class TemporalObservationClient:
    def __init__(self, client, stream_id='default', max_age_error=0.1, max_entries=256):
        self.client = client
        self.stream_id = str(stream_id)
        self.max_age_error = float(max_age_error)
        if not math.isfinite(self.max_age_error) or self.max_age_error < 0 or max_entries < 3:
            raise ValueError('invalid history buffer limits')
        self.frames = deque(maxlen=max_entries)
        self.episode = None
        self.last_sent = -1.

    def reset(self, episode_id, robo_name='robotwin'):
        self.client.infer(dict(reset=True, robo_name=robo_name))
        self.frames.clear()
        self.episode = episode_id
        self.last_sent = -1.

    def observe(self, observation, timestamp):
        import copy
        t = float(timestamp)
        if self.episode is None:
            raise ValueError('call reset with a unique episode_id first')
        if not math.isfinite(t) or t < 0 or (self.frames and t < self.frames[-1][0]):
            raise ValueError('environment timestamps must be finite and monotonic')
        # Simulator observation arrays may be reused/mutated after returning.
        frame = copy.deepcopy(observation)
        frame.update(timestamp=t, episode_id=self.episode, stream_id=self.stream_id)
        if self.frames and self.frames[-1][0] == t:
            self.frames.pop()
        self.frames.append((t, frame))
        while self.frames and t-self.frames[0][0] > 1.+self.max_age_error+1e-6:
            self.frames.popleft()

    def infer(self):
        if not self.frames:
            raise ValueError('observe current frame before infer')
        now, current = self.frames[-1]
        selected = []
        for offset in (.5, 1.):
            target = now-offset
            entry = next((x for x in reversed(self.frames) if x[0] <= target+1e-7), None)
            usable = entry is not None and target-entry[0] <= self.max_age_error+1e-7
            selected.append(entry[1] if usable else None)
        return self.client.infer({**current, 'history_observations': selected})
