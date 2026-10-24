"""Explicit opt-in to the reference Z6b per-episode seen instruction pool."""
from functools import lru_cache
import gzip
import json
import random


@lru_cache(maxsize=8)
def load_seen_pool(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        pool = json.load(stream)
    if not isinstance(pool, dict) or any(
        not isinstance(choices, list) or not choices
        or any(not isinstance(text, str) or not text.strip() for text in choices)
        for choices in pool.values()
    ):
        raise ValueError('Seen instruction pool must map episode IDs to nonempty string lists')
    return pool


def sample_seen_instruction(path, episode_index):
    return random.choice(load_seen_pool(path)[str(episode_index)])
