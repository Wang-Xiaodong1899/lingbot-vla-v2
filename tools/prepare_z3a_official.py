"""Prepare and validate the local official-data/aug1/Muon continuation recipe.

Exact upstream Z3a settings are unverified: the cached experiments branch has
only Z1/Z2 and B_aug1_n01. Defaults here combine official-data training with
the implemented level-1 augmentation and 0.01 normalized-state noise.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SIZE = 4100134542
ARCHIVE_SHA256 = "ed5948c2bb693e098e0edc9c693614f8621948b3fc5953748932b34ee5bf0419"


def prepare_data(dataset):
    marker = dataset / ".download_complete.json"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if (record.get("archive_sha256") == ARCHIVE_SHA256
                and record.get("sha256_verified") and record.get("extracted_crc_verified")):
            return
    if dataset.exists():
        raise RuntimeError(f"Dataset has no verified completion marker (extraction may still be running): {dataset}")
    archive = Path(os.environ.get("DATA_ARCHIVE", ROOT / "data/downloads/RoboTwin_lerobot_v30.zip.part"))
    if archive.suffix == ".part" and archive.with_suffix("").is_file():
        archive = archive.with_suffix("")
    if not archive.is_file() or archive.stat().st_size != ARCHIVE_SIZE:
        raise RuntimeError(f"Download incomplete: {archive}; expected {ARCHIVE_SIZE} bytes. Retry after download finishes.")
    print(f"Verifying SHA-256: {archive}", flush=True)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("RoboTwin archive SHA-256 mismatch")
    dataset.parent.mkdir(parents=True, exist_ok=True)
    # Stage extraction; never publish a half-extracted dataset or rename a download.
    with tempfile.TemporaryDirectory(prefix=".z3a-extract-", dir=dataset.parent) as staging:
        destination = Path(staging)
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                target = (destination / entry.filename).resolve()
                if (not target.is_relative_to(destination.resolve())
                        or not entry.filename.startswith("RoboTwin_lerobot_v30/")):
                    raise RuntimeError(f"Unexpected archive entry: {entry.filename}")
            bundle.extractall(destination)  # ZipFile validates each file's CRC.
        extracted = destination / "RoboTwin_lerobot_v30"
        (extracted / marker.name).write_text(json.dumps({
            "archive_sha256": ARCHIVE_SHA256, "sha256_verified": True,
            "extracted_crc_verified": True,
        }, indent=2) + "\n")
        extracted.rename(dataset)


def main():
    output = Path(os.environ["OUTPUT_DIR"]).resolve()
    launch = output / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(os.environ.get("INIT_MODEL", "/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000")).resolve()
    dataset = Path(os.environ.get("DATA_ROOT", ROOT / "data/RoboTwin_lerobot_v30")).resolve()
    norm = Path(os.environ.get("NORM_FILE", checkpoint.parent / "robotwin_clean_norm.json")).resolve()
    # Start from the native Muon recipe, never from the PICA configuration.
    config = yaml.safe_load((ROOT / "configs/vla/robotwin/robotwin.yaml").read_text())
    config["model"].update(model_path=str(checkpoint), config_path=str(checkpoint),
                           tokenizer_path=str(ROOT / "evaluation/Qwen3-VL-4B-Instruct"))
    robot_dir = launch / "robot_configs"
    robot_dir.mkdir(exist_ok=True)
    source_robot = checkpoint.parent / "launch/robot_configs/robotwin_clean.yaml"
    robot = yaml.safe_load(source_robot.read_text())
    robot["norm_stats"] = str(norm)
    (robot_dir / "robotwin_clean.yaml").write_text(yaml.safe_dump(robot, sort_keys=False))
    manifest = launch / "train.txt"
    if any(c.isspace() for c in str(dataset)):
        raise ValueError("Dataset path must not contain whitespace (training manifest format)")
    manifest.write_text(f"robotwin_clean {dataset}\n")
    config["data"].update(train_path=str(manifest), robot_config_root=str(robot_dir),
                          norm_stats_file=str(norm), image_augment=True,
                          num_workers=int(os.environ.get("DATA_NUM_WORKERS", 4)), prefetch_factor=1)
    train = config["train"]
    train.update(output_dir=str(output), optimizer="muon", use_moe_expert_lr=False,
                 micro_batch_size=int(os.environ["MICRO_BATCH_SIZE"]), global_batch_size=1024,
                 gradient_accumulation_steps=int(os.environ["GRAD_ACCUM"]),
                 data_parallel_shard_size=int(os.environ["NGPU"]), data_parallel_replicate_size=1,
                 enable_full_shard=True, enable_gradient_checkpointing=True, use_compile=False,
                 lr=float(os.environ.get("LR", "1e-4")), lr_min=float(os.environ.get("LR_MIN", "5e-5")),
                 lr_warmup_ratio=0.0, max_steps=int(os.environ.get("MAX_STEPS", 1000)),
                 save_steps=int(os.environ.get("SAVE_STEPS", 100)), save_hf_weights=True,
                 async_save_hf_weights=False, enable_resume=True, pica_params={})
    if train["max_steps"] < 1 or train["save_steps"] < 1:
        raise ValueError("MAX_STEPS and SAVE_STEPS must be positive")
    # Preserve the checkpoint's depth/video auxiliary objectives and architecture settings.
    saved = yaml.safe_load((checkpoint / "lingbotvla_cli.yaml").read_text())
    train["align_params"] = saved["model"]["align_params"]
    align = train["align_params"]
    align.pop("visual_dir", None)
    align["depth"].update(moge_path=str(ROOT / "weights/depth/moge2-vitb-normal.pt"),
                          morgbd_path=str(ROOT / "weights/depth/model.pt"))
    align["video"].update(ckpt_path=str(ROOT / "weights/dino_video/teacher_step_10000.pth"),
                          config_path=str(ROOT / "weights/dino_video/config.yaml"))
    (launch / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    provenance = {
        "recipe": "local official-data aug1 adaptation; exact upstream Z3a unverified",
        "initial_checkpoint": str(checkpoint), "dataset": str(dataset),
        "optimizer": "muon", "global_batch_size": 1024,
        "augmentation_level": 1, "state_noise_std": float(os.environ["LINGBOT_STATE_NOISE_STD"]),
        "initial_optimizer_state": "not supplied; new Muon state, training steps start at 0",
    }
    (launch / "recipe.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Configuration: {launch / 'config.yaml'}", flush=True)
    if os.environ.get("DRY_RUN") == "1":
        print("DRY_RUN: assets and data not validated", flush=True)
        return
    errors = []
    try:
        prepare_data(dataset)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        errors.append(str(exc))
    paths = [norm, checkpoint / "config.json", checkpoint / "model.safetensors.index.json"]
    for section, names in [("depth", ["moge_path", "morgbd_path"]), ("video", ["ckpt_path", "config_path"])]:
        paths.extend(Path(align[section][name]) for name in names)
    tokenizer = Path(config["model"]["tokenizer_path"])
    paths.extend(tokenizer / name for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"])
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        paths.extend(checkpoint / shard for shard in set(json.loads(index.read_text())["weight_map"].values()))
    errors.extend(f"Missing asset (download may still be running): {p}" for p in paths if not p.is_file() or p.stat().st_size == 0)
    info_path = dataset / "meta/info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if info["codebase_version"] != "v3.0" or info["total_episodes"] != 2500:
            errors.append(f"Unexpected dataset version/episodes: {info['codebase_version']}/{info['total_episodes']}")
    else:
        errors.append(f"Dataset metadata missing: {info_path}")
    for module in ["lerobot.datasets.lerobot_dataset", "torchcodec", "mdm.model.v2", "moge.model.v2", "moge.utils.vis"]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"Dependency {module}: {type(exc).__name__}: {exc}")
    if errors:
        raise SystemExit("Preflight failed; training not started:\n  " + "\n  ".join(errors))
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    metadata = LeRobotDatasetMetadata(dataset.name, root=dataset)
    print(f"Preflight passed: {metadata.total_episodes} episodes; native Muon; no PICA.", flush=True)


if __name__ == "__main__":
    main()
