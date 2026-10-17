"""Train only a candidate-to-GT flow refiner on a frozen Lingbot checkpoint.

Run with python (one GPU) or torchrun (one frozen base replica per GPU).
No depth/video teachers, original-policy training, or simulator are required.
"""

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
import yaml

from lingbotvla.models.vla.lingbot_vla.action_refinement import (
    ActionFlowRefiner, ActionRefinerConfig, FrozenLingbotProposal,
    action_valid_mask, masked_error,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="/root/autodl-tmp/models/bs512-step4k-ct-bs768-step2k")
    parser.add_argument("--norm-stats", type=Path, help="Base model normalization file; defaults to base-model/robotwin_clean_norm.json")
    parser.add_argument("--qwen-path", default="/root/autodl-tmp/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-list", default="/root/autodl-tmp/data/robotwin_clean50/robotwin_clean50.txt")
    parser.add_argument("--robot-config-root", default=str(ROOT / "configs/robot_configs"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/action-refiner"))
    parser.add_argument("--base-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1, help="Per GPU")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional optimizer update limit; omitted means run until stopped")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--loss-type", choices=["l1", "mse"], default="l1")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 avoids video-decoder worker teardown issues; increase after benchmarking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-data", action="store_true", help="Read one batch without loading base weights")
    parser.add_argument("--check-batch", action="store_true", help="One real proposal + refiner backward, no optimizer step or checkpoint")
    args = parser.parse_args()
    for name in ("batch_size", "grad_accum", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.max_grad_norm <= 0 or args.warmup_steps < 0:
        parser.error("Invalid workers/lr/max-grad-norm/warmup-steps")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive when specified")
    return args


def read_base_config(args):
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config

    model_dir = Path(args.base_model).resolve()
    raw = json.loads((model_dir / "config.json").read_text())
    config = LingbotVLAV2Config(**raw)
    # Config constructors set some defaults unconditionally. Honor saved fields.
    for key, value in raw.items():
        if key not in ("model_type", "architectures"):
            setattr(config, key, value)
    config.tokenizer_path = str(Path(args.qwen_path).resolve())
    config.attention_implementation = "eager"
    config.use_cache = True
    config.use_compile = False
    config.bias_update_speed = 0.0
    config.freeze_vision_encoder = True
    return config


def load_base(args, config, device):
    from safetensors.torch import load_file
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

    apply_lingbot_qwen3_vl_patch()
    dtype = getattr(torch, args.base_dtype)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        policy = LingbotVlaV2Policy(config, eval=True)
    finally:
        torch.set_default_dtype(previous_dtype)
    model_dir = Path(args.base_model)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = ["model.safetensors"]
    expected = set(policy.state_dict())
    loaded = set()
    for filename in filenames:
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Loading frozen base shard: {filename}", flush=True)
        shard = load_file(str(model_dir / filename), device="cpu")
        duplicates = loaded.intersection(shard)
        if duplicates:
            raise ValueError(f"Duplicate checkpoint keys: {sorted(duplicates)[:5]}")
        unexpected = set(shard) - expected
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {sorted(unexpected)[:5]}")
        policy.load_state_dict(shard, strict=False)
        loaded.update(shard)
        del shard
    if loaded != expected:
        raise ValueError(f"Missing checkpoint keys: {sorted(expected - loaded)[:10]}")
    return FrozenLingbotProposal(policy.to(device=device, dtype=dtype))


def build_data(args, config):
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.refinement_dataset import LocalV2RefinementDataset
    from lingbotvla.models import build_processor

    saved = yaml.safe_load((Path(args.base_model) / "lingbotvla_cli.yaml").read_text())
    data = saved["data"].copy()
    data.update(train_path=str(Path(args.train_list).resolve()), robot_config_root=args.robot_config_root,
                chunk_size=config.n_action_steps, use_future_image=False, image_augment=False,
                norm_stats_file=str((args.norm_stats or Path(args.base_model) / "robotwin_clean_norm.json").resolve()))
    # Saved CLI values are stringified mappings; also support ordinary YAML maps.
    for key in ("joints", "norm_type"):
        data[key] = [repr(value) if isinstance(value, dict) else value for value in data[key]]
    processor = build_processor(args.qwen_path)
    dataset = LocalV2RefinementDataset(args.train_list, SimpleNamespace(**data), config, processor)
    manifest = [{"dataset": str(task["root"]), "training_episodes":
                 [episode["episode_index"] for episode in task["episodes"]]} for task in dataset.tasks]
    return dataset, VLADataCollatorWithPacking(), manifest


class SkippedBatchSampler:
    """Resume at an epoch batch offset without decoding skipped video samples."""
    def __init__(self, sampler, skip):
        self.sampler, self.skip = sampler, skip

    def __iter__(self):
        for index, batch in enumerate(self.sampler):
            if index >= self.skip:
                yield batch

    def __len__(self):
        return max(0, len(self.sampler) - self.skip)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def reduce_sum(value):
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def save_checkpoint_files(output, payload):
    """Archive a complete save point before advancing latest; never overwrite it."""
    output = Path(output)
    snapshots = output / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    step = payload["step"]
    snapshot = snapshots / f"step{step}.pt"
    temporary = snapshots / f".step{step}.{os.getpid()}.tmp"
    latest_temporary = output / f".latest.{os.getpid()}.tmp"
    if snapshot.exists():
        raise FileExistsError(f"Checkpoint already exists and will not be overwritten: {snapshot}. "
                              "Use a new output directory for a new training branch.")
    try:
        torch.save(payload, temporary)
        # Atomic creation, including a second existence check by the filesystem.
        os.link(temporary, snapshot)
        # A hard link references the saved inode. Replacing latest at a later
        # step cannot change the archived file and needs no second 47 MB copy.
        os.link(snapshot, latest_temporary)
        os.replace(latest_temporary, output / "latest.pt")
    finally:
        temporary.unlink(missing_ok=True)
        latest_temporary.unlink(missing_ok=True)
    return snapshot


def main():
    args = arguments()
    os.chdir(ROOT)  # Robot YAML norm_stats paths are relative to the checkout.
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the frozen Lingbot model")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    config = read_base_config(args)
    train_data, collate, data_manifest = build_data(args, config)
    loader_options = dict(num_workers=args.workers, pin_memory=True, collate_fn=collate,
                          generator=torch.Generator().manual_seed(args.seed + rank))
    if args.workers:
        loader_options["multiprocessing_context"] = "spawn"
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=args.seed, drop_last=True)
    batches_per_epoch = len(sampler) // args.batch_size
    if not batches_per_epoch:
        raise ValueError("Training dataset is smaller than one distributed batch")

    def train_batches(start_micro_step):
        epoch, skip = divmod(start_micro_step, batches_per_epoch)
        while True:
            sampler.set_epoch(epoch)
            batches = SkippedBatchSampler(BatchSampler(sampler, args.batch_size, drop_last=True), skip)
            yield from DataLoader(train_data, batch_sampler=batches, **loader_options)
            epoch, skip = epoch + 1, 0

    if args.check_data:
        batch = next(train_batches(0))
        print(json.dumps({"train_frames": len(train_data), "all_episodes": True,
                          "batch": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}}, indent=2))
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    base = load_base(args, config, device)
    context_dim = base.policy.model.qwenvl_with_expert.qwenvl.config.text_config.hidden_size
    refiner_config = ActionRefinerConfig(
        action_dim=config.max_action_dim, state_dim=config.max_state_dim,
        horizon=config.n_action_steps, context_dim=context_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads, num_steps=args.refine_steps,
        loss_type=args.loss_type,
    )
    refiner = ActionFlowRefiner(refiner_config).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(step):
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.max_steps is None:
            return 1.0
        progress = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    metadata = {
        "base_model": str(Path(args.base_model).resolve()),
        "base_config_sha256": hashlib.sha256((Path(args.base_model) / "config.json").read_bytes()).hexdigest(),
        "norm_stats_path": str((args.norm_stats or Path(args.base_model) / "robotwin_clean_norm.json").resolve()),
        "norm_stats_sha256": hashlib.sha256((args.norm_stats or Path(args.base_model) / "robotwin_clean_norm.json").read_bytes()).hexdigest(),
        "data_config_sha256": hashlib.sha256((Path(args.base_model) / "lingbotvla_cli.yaml").read_bytes()).hexdigest(),
        "base_dtype": args.base_dtype, "base_num_steps": config.num_steps,
        "qwen_path": str(Path(args.qwen_path).resolve()), "train_list": str(Path(args.train_list).resolve()),
        "train_list_sha256": hashlib.sha256(Path(args.train_list).read_bytes()).hexdigest(),
        "robot_config_root": str(Path(args.robot_config_root).resolve()),
        "robot_config_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(args.robot_config_root).glob("*.yaml"))},
        "training_data": data_manifest, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "world_size": world, "seed": args.seed, "max_steps": args.max_steps,
        "lr": args.lr, "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
    }
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["metadata"] != metadata or checkpoint["refiner_config"] != refiner.config_dict():
            raise ValueError("Resume requires the same base/data/model/batch/world-size/schedule settings")
        refiner.load_state_dict(checkpoint["refiner"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"]
        restore_rng(checkpoint["rng_by_rank"][rank])
        if rank == 0:
            print(json.dumps({"event": "resume_loaded", "checkpoint": str(args.resume.resolve()),
                              "completed_steps": start_step, "next_step": start_step + 1,
                              "optimizer_state_entries": len(optimizer.state),
                              "lr": scheduler.get_last_lr()[0], "rng_ranks": len(checkpoint["rng_by_rank"])}),
                  flush=True)
        del checkpoint
    model = DistributedDataParallel(refiner, device_ids=[device.index]) if world > 1 else refiner
    batches = iter(train_batches(start_step * args.grad_accum))

    if args.check_batch:
        batch = to_device(next(batches), device)
        proposal, context = base(batch)
        loss, _ = refiner.flow_loss(proposal, batch["actions"], batch["state"], context,
                                    batch["joint_mask"], batch["action_is_pad"])
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None for p in base.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in refiner.parameters())
        refined = refiner.refine(proposal, batch["state"], context, batch["joint_mask"])
        if not args.resume:
            assert torch.equal(refined, proposal), "Zero-initialized refiner should preserve base actions"
        print(json.dumps({"check_batch": "passed", "proposal_shape": list(proposal.shape),
                          "context_shape": list(context.shape), "flow_loss": loss.item(),
                          "base_frozen": True, "optimizer_steps": 0}))
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "latest.pt").exists() and not args.resume:
            raise FileExistsError(f"{output}/latest.pt exists; use --resume or a new output directory")
        (output / "run_config.json").write_text(json.dumps(
            {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "metadata": metadata, "refiner_config": refiner.config_dict()}, indent=2))
        print(f"Frozen base; {sum(p.numel() for p in refiner.parameters()):,} trainable refiner parameters. "
              f"Global batch={world * args.batch_size * args.grad_accum}; "
              f"train={len(train_data)} frames; all episodes, no training-time evaluation.", flush=True)

    def save(step):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            payload = {"format_version": 1, "step": step, "refiner_config": refiner.config_dict(),
                       "refiner": refiner.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "metadata": metadata,
                       "rng_by_rank": states}
            snapshot = save_checkpoint_files(output, payload)
            print(json.dumps({"event": "checkpoint_saved", "step": step,
                              "snapshot": str(snapshot.resolve()),
                              "latest": str((output / "latest.pt").resolve())}), flush=True)
        if world > 1:
            dist.barrier()

    started = time.monotonic()
    steps = itertools.count(start_step) if args.max_steps is None else range(start_step, args.max_steps)
    for step in steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        for micro in range(args.grad_accum):
            batch = to_device(next(batches), device)
            proposal, context = base(batch)
            target = batch["actions"].float()
            mask = action_valid_mask(target, batch["joint_mask"], batch["action_is_pad"])
            valid_count = reduce_sum(mask.sum().double())
            if valid_count == 0:
                raise ValueError("No valid action targets in distributed batch")
            delta = torch.where(mask, target - proposal, 0.0)
            flow_time = torch.rand(proposal.shape[0], device=device)
            x_t = proposal + flow_time[:, None, None] * delta
            sync = model.no_sync() if world > 1 and micro + 1 < args.grad_accum else contextlib.nullcontext()
            with sync:
                velocity = model(x_t, flow_time, proposal, batch["state"], context, batch["joint_mask"])
                loss_sum, local_count = masked_error(velocity, delta, mask, args.loss_type)
                loss = loss_sum * world / valid_count / args.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite flow loss at step {step + 1}")
                loss.backward()
            totals += torch.stack([loss_sum.detach().double(), local_count.double()])
        norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        current = step + 1
        if current % args.log_every == 0 or current == start_step + 1:
            reduce_sum(totals)
            record = {"step": current, "train_flow_loss": (totals[0] / totals[1]).item(),
                      "grad_norm": norm.item(), "lr": scheduler.get_last_lr()[0],
                      "elapsed_seconds": time.monotonic() - started}
            if rank == 0:
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        if current % args.save_every == 0 or current == args.max_steps:
            save(current)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
