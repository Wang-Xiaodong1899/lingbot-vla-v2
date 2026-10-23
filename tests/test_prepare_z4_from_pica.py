"""CPU regression checks for HF-only initialization and resume isolation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_z3a", ROOT / "tools/prepare_z3a_official.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparePicaZ4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "source/global_step_2000"
        self.checkpoint.mkdir(parents=True)
        self.align = yaml.safe_load((ROOT / "configs/vla/robotwin/robotwin.yaml").read_text())["train"]["align_params"]
        self.align["depth_loss_weight"] = 0.007  # Must come from the checkpoint.
        (self.checkpoint / "config.json").write_text(json.dumps({
            "align_params": self.align, "pica_params": {"weight": 0.05},
        }))
        self.output = self.root / "output"
        self.env = dict(
            OUTPUT_DIR=str(self.output), INIT_MODEL=str(self.checkpoint),
            LINGBOT_TRAIN_RECIPE="z4", LINGBOT_Z4_CONFIG=str(ROOT / "configs/augmentation/z4.yaml"),
            ROBOT_CONFIG=str(ROOT / "configs/robot_configs/robotwin.yaml"),
            NORM_FILE=str(self.root / "source/robotwin_clean_norm.json"),
            DATA_ROOT=str(self.root / "data"), MICRO_BATCH_SIZE="32", GRAD_ACCUM="16", NGPU="2",
            LR="1e-5", LR_MIN="5e-6", MAX_STEPS="1000", SAVE_STEPS="100",
            LINGBOT_STATE_NOISE_STD="0", DRY_RUN="1",
        )

    def run_prepare(self, **overrides):
        with patch.dict(os.environ, {**self.env, **overrides}), contextlib.redirect_stdout(io.StringIO()):
            prepare.main()
        return yaml.safe_load((self.output / "launch/config.yaml").read_text())

    def test_hf_only_export_retains_auxiliary_architecture_without_pica(self):
        config = self.run_prepare()
        self.assertEqual(config["model"]["model_path"], str(self.checkpoint))
        self.assertEqual(config["train"]["pica_params"], {})
        self.assertEqual(config["train"]["align_params"]["depth_loss_weight"], 0.007)
        self.assertEqual(config["train"]["lr"], 1e-5)
        self.assertEqual(config["train"]["lr_min"], 5e-6)
        self.assertEqual(config["train"]["global_batch_size"], 1024)
        self.assertEqual(config["train"]["optimizer"], "muon")
        self.assertFalse((self.checkpoint / "lingbotvla_cli.yaml").exists())
        source = json.loads((self.checkpoint / "config.json").read_text())
        self.assertEqual(source["pica_params"], {"weight": 0.05})

    def test_existing_run_cannot_switch_initial_checkpoint(self):
        self.run_prepare()
        (self.output / "checkpoints/global_step_100").mkdir(parents=True)
        previous = (self.output / "launch/recipe.json").read_bytes()
        self.run_prepare()  # Same checkpoint/recipe may resume.
        with self.assertRaisesRegex(ValueError, "different initial checkpoint"):
            self.run_prepare(INIT_MODEL=str(self.root / "another_checkpoint"))
        self.assertEqual((self.output / "launch/recipe.json").read_bytes(), previous)

    def test_existing_run_cannot_switch_augmentation(self):
        self.run_prepare()
        (self.output / "checkpoints/global_step_100").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "different augmentation recipe"):
            self.run_prepare(LINGBOT_TRAIN_RECIPE="z3a")

    def test_legacy_cli_alignment_fallback(self):
        (self.checkpoint / "config.json").write_text("{}")
        (self.checkpoint / "lingbotvla_cli.yaml").write_text(yaml.safe_dump({"model": {"align_params": self.align}}))
        config = self.run_prepare()
        self.assertEqual(config["train"]["align_params"]["depth_loss_weight"], 0.007)

    def test_invalid_learning_rates_are_rejected(self):
        for lr, low in [("0", "0"), ("nan", "0"), ("1e-5", "5e-5"), ("1e-5", "-1")]:
            with self.subTest(lr=lr, low=low), self.assertRaisesRegex(ValueError, "Learning rates"):
                self.run_prepare(LR=lr, LR_MIN=low)


if __name__ == "__main__":
    unittest.main()
