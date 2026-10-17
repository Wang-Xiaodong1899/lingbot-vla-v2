"""Validate local DCP checkpoints before publishing or resuming them."""
import json
import os
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader


def validate_checkpoint(path, world_size=None, require_optimizer=True, require_extra_state=True):
    """Also supports legacy checkpoints without a completion marker.

    Reading metadata and checking every referenced byte range detects interrupted
    saves without loading multi-GB model tensors into RAM.
    """
    root = Path(path)
    marker = root / '_SUCCESS.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        saved_world_size = saved['world_size']
    else:
        saved_world_size = (len(list((root / 'extra_state').glob('extra_state_rank_*.pt')))
                            if require_extra_state else world_size)
    if world_size is not None and saved_world_size != world_size:
        raise ValueError(f'{root}: saved ranks={saved_world_size}, expected={world_size}')
    if saved_world_size < 1:
        raise ValueError(f'{root}: missing rank training states')
    for rank in range(saved_world_size if require_extra_state else 0):
        p = root / 'extra_state' / f'extra_state_rank_{rank}.pt'
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'Missing or empty training state: {p}')
    components = ['model'] + (['optimizer'] if require_optimizer else [])
    if (root / 'ema').exists():
        components.append('ema')
    for component in components:
        directory = root / component
        if not (directory / '.metadata').is_file():
            raise ValueError(f'Missing DCP metadata: {directory}')
        metadata = FileSystemReader(directory).read_metadata()
        if not metadata.storage_data:
            raise ValueError(f'Empty DCP storage map: {directory}')
        sizes = {}
        for storage in metadata.storage_data.values():
            p = directory / storage.relative_path
            if p not in sizes:
                sizes[p] = p.stat().st_size if p.is_file() else 0
            if sizes[p] < storage.offset + storage.length:
                raise ValueError(f'Truncated DCP shard: {p}')
    return {'world_size': saved_world_size, 'components': components}


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
