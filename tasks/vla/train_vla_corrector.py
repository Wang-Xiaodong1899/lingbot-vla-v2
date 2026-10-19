"""Train an external VLA-Corrector MLP on frozen Lingbot visual token caches."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from lingbotvla.corrector.dataset import DynamicsDataset, split_episodes
from lingbotvla.corrector.runtime import build_mlp, sha256
from lingbotvla.corrector.vendor.config import SiglipMLPConfig


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def loss_terms(pred, target):
    mse = F.mse_loss(pred, target)
    cosine = (1 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1)).mean()
    return mse + cosine, mse, cosine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=128, help='Per GPU; 3 GPUs => global 384')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true', help='Isolated test: smaller MLP, two updates, allow partial cache')
    p.add_argument('--smoke-steps', type=int, default=2, help='Testing only: number of updates in smoke mode')
    a = p.parse_args()
    if min(a.batch_size, a.epochs, a.save_every, a.patience) < 1 or a.workers < 0 or a.lr <= 0:
        p.error('Invalid training arguments')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed + rank)
    random.seed(a.seed + rank)
    train_paths, val_paths, feature_metadata = split_episodes(a.cache, seed=a.seed, allow_smoke=a.smoke)
    train_data, val_data = DynamicsDataset(train_paths), DynamicsDataset(val_paths)
    example = train_data[0]
    train_data.cache.clear()  # Spawn workers reopen memmaps instead of pickling a full episode.
    model_config = asdict(SiglipMLPConfig(token_dim=example['z'].shape[-1], action_dim=example['action'].shape[-1]))
    if a.smoke:
        model_config.update(scale='custom', custom_widths=[64, 64])
    model = build_mlp(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=a.seed, drop_last=True)
    if len(sampler) < a.batch_size:
        raise ValueError('Dataset smaller than a distributed training batch')
    batches_per_epoch = len(sampler) // a.batch_size
    total_steps = a.epochs * batches_per_epoch
    def lr_factor(step):
        warmup = min(100, max(1, total_steps // 10))
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total_steps - warmup)))) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    split = dict(train=[str(p.resolve()) for p in train_paths], validation=[str(p.resolve()) for p in val_paths])
    metadata = dict(feature_metadata=feature_metadata, feature_metadata_sha256=sha256(a.cache / 'metadata.json'),
                    split=split, batch_size=a.batch_size, world_size=world, epochs=a.epochs, lr=a.lr,
                    seed=a.seed, k_step=10, loss='mse_plus_cosine', smoke=a.smoke)
    start_epoch, start_batch, step, best, bad_epochs = 0, 0, 0, float('inf'), 0
    if a.resume:
        ckpt = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ckpt['metadata'] != metadata or ckpt['model_config'] != model_config:
            raise ValueError('Resume base/data/split/model/batch/world/schedule changed')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, start_batch, step = ckpt['next_epoch'], ckpt['next_batch'], ckpt['step']
        best, bad_epochs = ckpt['best_val'], ckpt['bad_epochs']
        restore_rng(ckpt['rng_by_rank'][rank])
        del ckpt
    wrapped = DDP(model, device_ids=[device.index]) if world > 1 else model
    a.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if (a.output / 'latest.pt').exists() and not a.resume:
            raise FileExistsError('Output already has a checkpoint; use --resume or a new directory')
        (a.output / 'run_config.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                                                                metadata=metadata, model_config=model_config), indent=2))
        print(json.dumps(dict(event='training_started', parameters=sum(p.numel() for p in model.parameters()),
                              train_pairs=len(train_data), validation_pairs=len(val_data),
                              train_episodes=len(train_paths), validation_episodes=len(val_paths),
                              world_size=world, global_batch=world*a.batch_size, model_config=model_config)), flush=True)
    loader_kwargs = dict(num_workers=a.workers, pin_memory=True,
                         generator=torch.Generator().manual_seed(a.seed + rank))
    if a.workers:
        loader_kwargs.update(multiprocessing_context='spawn', prefetch_factor=1)

    def save(next_epoch, next_batch, best_checkpoint=False, snapshot=False):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            payload = dict(format='lingbot-vla-corrector-v1', model=model.state_dict(), model_config=model_config,
                           metadata=metadata, step=step, next_epoch=next_epoch, next_batch=next_batch,
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng_by_rank=states,
                           best_val=best, bad_epochs=bad_epochs)
            atomic_save(payload, a.output / 'latest.pt')
            if snapshot:
                path = a.output / 'snapshots' / f'step{step}.pt'
                if path.exists():
                    raise FileExistsError(f'Snapshot already exists: {path}')
                path.parent.mkdir(exist_ok=True)
                os.link(a.output / 'latest.pt', path)
            if best_checkpoint:
                atomic_save(payload, a.output / 'best.pt')
            print(json.dumps(dict(event='checkpoint_saved', step=step, best=best_checkpoint, snapshot=snapshot)), flush=True)
        if world > 1:
            dist.barrier()

    started = time.monotonic()
    for epoch in range(start_epoch, a.epochs):
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))
        offset = start_batch * a.batch_size if epoch == start_epoch else 0
        loader = DataLoader(Subset(train_data, indices[offset:]), batch_size=a.batch_size, drop_last=True, **loader_kwargs)
        wrapped.train()
        for batch_index, batch in enumerate(loader, start=start_batch if epoch == start_epoch else 0):
            batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = wrapped(batch['z'], batch['action'])
            loss, mse, cosine = loss_terms(prediction, batch['delta'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite corrector loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 10 == 0 or step == 1 or a.smoke:
                values = torch.tensor([loss.item(), mse.item(), cosine.item()], device=device)
                if world > 1:
                    dist.all_reduce(values)
                if rank == 0:
                    record = dict(event='train', epoch=epoch+1, step=step, loss=float(values[0]/world),
                                  mse=float(values[1]/world), cosine=float(values[2]/world),
                                  grad_norm=float(grad), lr=scheduler.get_last_lr()[0], elapsed=time.monotonic()-started)
                    print(json.dumps(record), flush=True)
                    with (a.output / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record)+'\n')
            at_end = batch_index + 1 == batches_per_epoch
            next_epoch, next_batch = (epoch+1,0) if at_end else (epoch,batch_index+1)
            if step % a.save_every == 0 or (a.smoke and step >= a.smoke_steps):
                save(next_epoch, next_batch, snapshot=True)
            if a.smoke and step >= a.smoke_steps:
                if dist.is_initialized():
                    dist.destroy_process_group()
                return
        # Validation is disjoint by episode, never used as training samples.
        model.eval()
        sums = torch.zeros(4, dtype=torch.float64, device=device)
        val_indices = list(range(rank, len(val_data), world))
        with torch.no_grad():
            for batch in DataLoader(Subset(val_data, val_indices), batch_size=a.batch_size, **loader_kwargs):
                batch = {k:v.to(device,non_blocking=True) for k,v in batch.items()}
                prediction = model(batch['z'],batch['action'])
                loss, mse, cosine = loss_terms(prediction,batch['delta'])
                n = len(prediction)
                sums += torch.tensor([loss.item()*n,mse.item()*n,cosine.item()*n,n], device=device)
        if world > 1:
            dist.all_reduce(sums)
        val_loss = float(sums[0]/sums[3])
        if not math.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation loss')
        improved = val_loss < best
        best, bad_epochs = min(best,val_loss), 0 if improved else bad_epochs+1
        if rank == 0:
            record = dict(event='validation',epoch=epoch+1,step=step,loss=val_loss,
                          cosine=float(sums[2]/sums[3]),mse=float(sums[1]/sums[3]),best=best,bad_epochs=bad_epochs)
            print(json.dumps(record),flush=True)
            with (a.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
        save(epoch+1,0,best_checkpoint=improved)
        start_batch=0
        if bad_epochs >= a.patience:
            break
    if rank == 0:
        (a.output/'training_complete.json').write_text(json.dumps(dict(step=step,best_val=best,best_checkpoint=str((a.output/'best.pt').resolve()))))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
