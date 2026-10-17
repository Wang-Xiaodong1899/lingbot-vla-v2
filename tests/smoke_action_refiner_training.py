"""Exercise DDP, checkpointing and exact resume with synthetic data.

Run: python -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/smoke_action_refiner_training.py --test-dir /tmp/refiner-smoke-UNIQUE --phase initial
Then repeat with --phase resume to check exact resumption.
No real base-model weights or robot training data are loaded.
"""

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("train_action_refiner", ROOT / "tasks/vla/train_action_refiner.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class SyntheticData(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        state = torch.randn(3, generator=generator)
        return {"state": state, "actions": state[None].expand(4, 3) * 0.1,
                "joint_mask": torch.ones(4, 3, dtype=torch.bool),
                "action_is_pad": torch.tensor([False, False, False, index % 2 == 0])}


class SyntheticBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = SimpleNamespace(model=SimpleNamespace(qwenvl_with_expert=SimpleNamespace(
            qwenvl=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=6))))))

    def forward(self, batch):
        state = batch["state"]
        return torch.randn(state.shape[0], 4, 3, device=state.device) * 0.1, state.repeat(1, 2)


def run(output, resume=None):
    fixture = output.parent / "fixture"
    sys.argv = ["train_action_refiner.py", "--base-model", str(fixture), "--train-list", str(fixture / "train.txt"), "--norm-stats", str(fixture / "robotwin_clean_norm.json"), "--output-dir", str(output), "--workers", "0",
                "--batch-size", "2", "--grad-accum", "2", "--max-steps", "2",
                "--warmup-steps", "0", "--save-every", "1",
                "--log-every", "1", "--hidden-dim", "16",
                "--num-heads", "2", "--num-layers", "1"]
    if resume:
        sys.argv += ["--resume", str(resume)]
    trainer.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["initial", "resume"], required=True)
    args = parser.parse_args()
    fixture = args.test_dir / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "lingbotvla_cli.yaml", "robotwin_clean_norm.json", "train.txt"):
        (fixture / name).write_text("{}")
    trainer.read_base_config = lambda _: SimpleNamespace(max_action_dim=3, max_state_dim=3,
                                                         n_action_steps=4, num_steps=10)
    trainer.build_data = lambda *_: (SyntheticData(),
                                     torch.utils.data.default_collate, [{"synthetic": True}])
    trainer.load_base = lambda *_: SyntheticBase()
    original_replace = trainer.os.replace

    def snapshot(source, destination):
        original_replace(source, destination)
        if torch.load(destination, map_location="cpu", weights_only=False)["step"] == 1:
            shutil.copyfile(destination, args.test_dir / "step1.pt")

    if args.phase == "initial":
        trainer.os.replace = snapshot
        run(args.test_dir / "uninterrupted")
        return
    run(args.test_dir / "resumed", args.test_dir / "step1.pt")
    if int(trainer.os.environ.get("RANK", 0)) == 0:
        first = torch.load(args.test_dir / "uninterrupted/latest.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(args.test_dir / "resumed/latest.pt", map_location="cpu", weights_only=False)
        for name, weight in first["refiner"].items():
            torch.testing.assert_close(weight, resumed["refiner"][name], rtol=0, atol=0)
        assert first["scheduler"] == resumed["scheduler"]
        print("PASS: DDP training, checkpoints, and bit-exact step-1 resume", flush=True)


if __name__ == "__main__":
    main()
