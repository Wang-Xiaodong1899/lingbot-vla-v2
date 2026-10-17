"""Pin a saved refiner step and evaluate its matching base on RoboTwin.

Default: latest checkpoint, clean + randomized, 50 tasks x 10 episodes,
4 GPUs x 2 slots. Waits for GPU memory; never stops an active training run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_checkpoint(source, archive_dir, expected_base=None):
    """Capture one inode's contents even if source is replaced during the copy."""
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint has not been saved: {source}")
    archive_dir = Path(archive_dir).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".capture_", suffix=".pt", dir=archive_dir)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        payload = torch.load(temporary, map_location="cpu", weights_only=False)
        if payload.get("format_version") != 1 or "refiner" not in payload:
            raise ValueError("Expected a refiner checkpoint produced by train_action_refiner.py")
        metadata = payload["metadata"]
        base = Path(metadata["base_model"]).resolve()
        if expected_base is not None and base != Path(expected_base).resolve():
            raise ValueError(f"Checkpoint was trained on {base}, not the expected {expected_base}")
        if file_digest(base / "config.json") != metadata["base_config_sha256"]:
            raise ValueError("Base model configuration differs from checkpoint metadata")
        if metadata["base_dtype"] != "float32":
            raise ValueError("This evaluation launcher currently uses the FP32 benchmark setting")
        step = int(payload["step"])
        digest = file_digest(temporary)
        pinned = archive_dir / f"step{step}_{digest[:12]}.pt"
        try:
            os.link(temporary, pinned)
        except FileExistsError:
            if file_digest(pinned) != digest:
                raise ValueError("Pinned checkpoint name conflicts with different contents")
        return dict(step=step, base_model=str(base), checkpoint=str(pinned), sha256=digest,
                    original_source=str(source), base_dtype=metadata["base_dtype"],
                    pinned_at_utc=datetime.now(timezone.utc).isoformat())
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--expected-base", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--step", type=int, help="Saved step, e.g. 500 or 1500")
    selection.add_argument("--checkpoint", type=Path, help="Explicit checkpoint; default is latest.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--settings", choices=["both", "clean", "randomized"], default="both")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-per-gpu", type=int, default=2)
    parser.add_argument("--start-port", type=int, default=9730)
    parser.add_argument("--prepare-only", action="store_true", help="Validate/pin checkpoint without starting evaluation")
    args = parser.parse_args()
    if args.step is not None and args.step <= 0:
        parser.error("--step must be positive")
    if args.num_gpus <= 0 or args.num_per_gpu <= 0:
        parser.error("GPU counts must be positive")
    train_output = args.train_output.resolve()
    source = args.checkpoint or (train_output / "snapshots" / f"step{args.step}.pt" if args.step is not None
                                 else train_output / "latest.pt")
    pinned = pin_checkpoint(source, train_output / "eval_checkpoints", args.expected_base)
    if args.step is not None and pinned["step"] != args.step:
        raise ValueError(f"Requested step {args.step}, but checkpoint contains step {pinned['step']}")
    output = args.output_dir or train_output / f"eval_step{pinned['step']}_10episodes"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "selected_checkpoint.json"
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != pinned["sha256"]:
        raise ValueError("Evaluation output already belongs to another checkpoint; use a new --output-dir")
    manifest.write_text(json.dumps(pinned, ensure_ascii=False, indent=2))
    command = [sys.executable, "-u", str(ROOT / "experiment/robotwin/eval_action_refiner.py"),
               "--checkpoint", pinned["checkpoint"], "--base-model", pinned["base_model"],
               "--output-dir", str(output), "--settings", args.settings,
               "--num-gpus", str(args.num_gpus), "--num-per-gpu", str(args.num_per_gpu),
               "--start-port", str(args.start_port)]
    print(json.dumps({"selected": pinned, "output_dir": str(output), "command": command},
                     ensure_ascii=False, indent=2), flush=True)
    if not args.prepare_only:
        os.chdir(ROOT)
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
