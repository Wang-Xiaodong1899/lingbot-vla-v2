"""Extract episode-separated spatial visual tokens from frozen Lingbot 10k."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tasks.vla.train_action_refiner import read_base_config, load_base, build_data, to_device
from lingbotvla.corrector.runtime import base_metadata, quantize_tokens, visual_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True)
    p.add_argument('--qwen-path', required=True)
    p.add_argument('--norm-stats', type=Path, required=True)
    p.add_argument('--train-list', required=True)
    p.add_argument('--robot-config-root', default=str(ROOT / 'configs/robot_configs'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-episodes', type=int, default=0, help='Only for isolated smoke extraction')
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    os.chdir(ROOT)
    a.base_dtype = 'float32'
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    config = read_base_config(a)
    config.precompute_grid_thw = False  # Last extraction batch can have a different size.
    config.use_qwen3_fixed_grid_cache = False
    dataset, collate, manifest = build_data(a, config)
    metadata = base_metadata(a.base_model, a.norm_stats, a.train_list, a.qwen_path)
    metadata.update(k_step=10, total_frames=len(dataset), total_episodes=len(dataset.episodes),
                    training_data=manifest, smoke_max_episodes=a.max_episodes)
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_file = a.output / 'metadata.json'
    if rank == 0:
        if manifest_file.exists() and json.loads(manifest_file.read_text()) != metadata:
            raise ValueError('Feature cache metadata changed; use a new directory')
        manifest_file.write_text(json.dumps(metadata, indent=2))
    base = load_base(a, config, device)
    core = base.policy.model
    # Only the visual tower is needed for extraction. Release language/action modules.
    del core.qwenvl_with_expert.qwenvl.model.language_model
    del core.qwenvl_with_expert.qwen_expert
    torch.cuda.empty_cache()
    report_first_batch = True
    episodes = dataset.episodes[:a.max_episodes or None]
    for index in range(rank, len(episodes), world):
        episode = episodes[index]
        task = dataset.tasks[episode['task']]
        name = f"task{episode['task']:02d}_episode{episode['episode_index']:06d}"
        output = a.output / name
        output.mkdir(exist_ok=True)
        done = output / 'complete.json'
        if done.exists():
            info = json.loads(done.read_text())
            if info['frames'] != episode['length']:
                raise ValueError(f'Episode length changed: {name}')
            print(json.dumps(dict(event='cache_reused', rank=rank, episode=name)), flush=True)
            continue
        loader = DataLoader(Subset(dataset, range(episode['start'], episode['start'] + episode['length'])),
                            batch_size=a.batch_size, collate_fn=collate, num_workers=0)
        position = 0
        arrays = None
        first_batch_started = time.monotonic()
        if report_first_batch:
            torch.cuda.reset_peak_memory_stats(device)
        for batch in loader:
            batch = to_device(batch, device)
            z = visual_tokens(core, batch).cpu().numpy()
            q, scale = quantize_tokens(z)
            if report_first_batch:
                print(json.dumps(dict(event='extraction_batch_verified', rank=rank,
                                      configured_batch=a.batch_size, actual_batch=len(z),
                                      peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                                      elapsed_seconds=time.monotonic()-first_batch_started)), flush=True)
                report_first_batch = False
            if arrays is None:
                n = episode['length']
                shapes = {'z_q': (np.int8, (n, *q.shape[1:])),
                          'z_scale': (np.float16, (n, *scale.shape[1:])),
                          'actions': (np.float32, (n, config.max_action_dim)),
                          'mask': (np.bool_, (config.max_action_dim,))}
                arrays = {key: np.lib.format.open_memmap(output / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
                          for key, (dtype, shape) in shapes.items()}
                arrays['mask'][:] = batch['action_joint_mask'][0].cpu().numpy()
            end = position + len(q)
            arrays['z_q'][position:end] = q
            arrays['z_scale'][position:end] = scale
            arrays['actions'][position:end] = (batch['actions'][:, 0] * batch['action_joint_mask']).cpu().numpy()
            position = end
        if position != episode['length']:
            raise ValueError(f'Incomplete extraction: {name}')
        for array in arrays.values():
            array.flush()
        info = dict(task=episode['task'], dataset=str(task['root']), episode=episode['episode_index'],
                    frames=position, tokens=int(arrays['z_q'].shape[1]), token_dim=int(arrays['z_q'].shape[2]))
        temporary = output / 'complete.tmp'
        temporary.write_text(json.dumps(info))
        os.replace(temporary, done)
        del arrays
        print(json.dumps(dict(event='episode_extracted', rank=rank, index=index, **info)), flush=True)
    print(json.dumps(dict(event='extraction_rank_complete', rank=rank, world=world)), flush=True)


if __name__ == '__main__':
    main()
