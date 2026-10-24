"""Prepare and validate the local official-data/aug1/Muon continuation recipe.

Exact upstream Z3a settings are unverified: the cached experiments branch has
only Z1/Z2 and B_aug1_n01. Defaults here combine official-data training with
the implemented level-1 augmentation and 0.01 normalized-state noise.
"""
import hashlib
import importlib
from dataclasses import asdict
import json
import math
import shutil
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
    recipe = os.environ.get("LINGBOT_TRAIN_RECIPE", "z3a")
    if recipe not in ("z3a", "z4", "z6b"):
        raise ValueError("Unknown training recipe")
    checkpoint = Path(os.environ.get("INIT_MODEL", "/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000")).resolve()
    z6b_config = None
    instruction_source = None
    if recipe == "z6b":
        from lingbotvla.data.vla_data.z6b_augmentation import _AUG5
        from lingbotvla.data.vla_data.instruction_pool import load_seen_pool
        import cv2  # JPEG is mandatory for this recipe, never silently skipped.
        z6b_config = json.loads(json.dumps(_AUG5))
        state_noise = float(os.environ["LINGBOT_STATE_NOISE_STD"])
        if not math.isfinite(state_noise) or state_noise < 0:
            raise ValueError("Z6b state noise must be finite and nonnegative")
        instruction_source = Path(os.environ["LINGBOT_INSTRUCTION_POOL"]).resolve()
        pool = load_seen_pool(str(instruction_source))
        if set(pool) != {str(i) for i in range(2500)}:
            raise ValueError("Z6b requires the official 2500-episode v1 seen instruction pool")
        pool_sha256 = hashlib.sha256(instruction_source.read_bytes()).hexdigest()
        expected_pool = json.loads((ROOT / "experiment/robotwin/records/z6b_source.json").read_text())["source_sha256"]["data/official_instruction_pool_v1.json.gz"]
        if pool_sha256 != expected_pool:
            raise ValueError("Z6b instruction pool differs from the reference v1 release")
    z4_config = None
    if recipe == "z4":
        from lingbotvla.data.vla_data.z4_augmentation import load_config
        z4_config = json.loads(json.dumps(asdict(load_config(os.environ.get("LINGBOT_Z4_CONFIG")))))
    # A continuation must not silently switch augmentation on an existing run.
    previous_path = launch / "recipe.json"
    if any((output / "checkpoints").glob("global_step_*")):
        if not previous_path.is_file():
            raise ValueError("Existing checkpoints have no recipe record; use a new OUTPUT_DIR")
        previous = json.loads(previous_path.read_text())
        old_policy = previous.get("augmentation_policy", "z3a")
        if old_policy != recipe or (recipe == "z4" and previous.get("z4_config") != z4_config):
            raise ValueError("Existing run has a different augmentation recipe; use a new OUTPUT_DIR")
        if recipe == "z6b" and (previous.get("z6b_config") != z6b_config
                                or previous.get("instruction_pool_sha256") != pool_sha256
                                or previous.get("state_noise_std") != float(os.environ["LINGBOT_STATE_NOISE_STD"])):
            raise ValueError("Existing run has a different Z6b recipe; use a new OUTPUT_DIR")
        if previous.get("initial_checkpoint") != str(checkpoint):
            raise ValueError("Existing run has a different initial checkpoint; use a new OUTPUT_DIR")
    if z4_config is not None:
        (launch / "z4_augmentation.yaml").write_text(yaml.safe_dump(z4_config, sort_keys=False))
    if instruction_source is not None:
        destination = launch / "instruction_pool_v1.json.gz"
        if instruction_source != destination.resolve():
            shutil.copyfile(instruction_source, destination)
    dataset = Path(os.environ.get("DATA_ROOT", ROOT / "data/RoboTwin_lerobot_v30")).resolve()
    norm = Path(os.environ.get("NORM_FILE", checkpoint.parent / "robotwin_clean_norm.json")).resolve()
    # Start from the native Muon recipe, never from the PICA configuration.
    config = yaml.safe_load((ROOT / "configs/vla/robotwin/robotwin.yaml").read_text())
    config["model"].update(model_path=str(checkpoint), config_path=str(checkpoint),
                           tokenizer_path=os.environ.get("QWEN3VL_PATH", str(ROOT / "evaluation/Qwen3-VL-4B-Instruct")))
    robot_dir = launch / "robot_configs"
    robot_dir.mkdir(exist_ok=True)
    source_robot = Path(os.environ.get("ROBOT_CONFIG", os.environ.get("ROBOT_CONFIG_FILE", checkpoint.parent / "launch/robot_configs/robotwin_clean.yaml"))).resolve()
    robot = yaml.safe_load(source_robot.read_text())
    robot["norm_stats"] = str(norm)
    (robot_dir / "robotwin_clean.yaml").write_text(yaml.safe_dump(robot, sort_keys=False))
    manifest = launch / "train.txt"
    if any(c.isspace() for c in str(dataset)):
        raise ValueError("Dataset path must not contain whitespace (training manifest format)")
    manifest.write_text(f"robotwin_clean {dataset}\n")
    config["data"].update(train_path=str(manifest), robot_config_root=str(robot_dir),
                          norm_stats_file=str(norm), image_augment=recipe != "z4",
                          num_workers=int(os.environ.get("DATA_NUM_WORKERS", 4)), prefetch_factor=1)
    train = config["train"]
    train.update(output_dir=str(output), optimizer="muon", use_moe_expert_lr=False,
                 micro_batch_size=int(os.environ["MICRO_BATCH_SIZE"]), global_batch_size=int(os.environ.get("GLOBAL_BATCH_SIZE", "1024")),
                 gradient_accumulation_steps=int(os.environ["GRAD_ACCUM"]),
                 data_parallel_shard_size=int(os.environ["NGPU"]), data_parallel_replicate_size=1,
                 enable_full_shard=True, enable_gradient_checkpointing=True, use_compile=False,
                 lr=float(os.environ.get("LR", "1e-4")), lr_min=float(os.environ.get("LR_MIN", "5e-5")),
                 lr_warmup_ratio=0.0, max_steps=int(os.environ.get("MAX_STEPS", 1000)),
                 save_steps=int(os.environ.get("SAVE_STEPS", 100)), save_hf_weights=True,
                 async_save_hf_weights=False, enable_resume=True, pica_params={})
    train["enable_fp32"] = os.environ.get("ENABLE_FP32", str(train.get("enable_fp32", True))).lower() in ("1", "true", "yes")
    train["lr_decay_style"] = os.environ.get("LR_DECAY_STYLE", train.get("lr_decay_style", "cosine"))
    if recipe == "z6b":
        train["lr_decay_ratio"] = float(os.environ.get("LR_DECAY_RATIO", "2.0"))
        if train["lr_decay_style"] != "cosine" or not math.isfinite(train["lr_decay_ratio"]) or train["lr_decay_ratio"] < 1:
            raise ValueError("Z6b requires cosine LR with finite LR_DECAY_RATIO >= 1")
    if train["max_steps"] < 1 or train["save_steps"] < 1:
        raise ValueError("MAX_STEPS and SAVE_STEPS must be positive")
    if not (math.isfinite(train["lr"]) and math.isfinite(train["lr_min"])
            and 0 <= train["lr_min"] <= train["lr"] and train["lr"] > 0):
        raise ValueError("Learning rates must be finite and satisfy 0 <= LR_MIN <= LR, LR > 0")
    # Preserve the checkpoint's depth/video auxiliary objectives and architecture settings.
    # HF-only exports may have no per-step CLI YAML. Their config.json carries
    # the alignment architecture; do not inherit the source run's PICA loss.
    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    if "align_params" in checkpoint_config:
        train["align_params"] = checkpoint_config["align_params"]
    else:
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
        "recipe": ("Z4: 10% original / 90% combined image corruptions" if recipe == "z4" else
                   "Z6b reference level-5 whitelist + seen v1 instruction pool" if recipe == "z6b" else
                   "local official-data aug1 adaptation; exact upstream Z3a unverified"),
        "initial_checkpoint": str(checkpoint), "dataset": str(dataset),
        "source_robot_config": str(source_robot), "normalization_file": str(norm),
        "optimizer": "muon", "global_batch_size": train["global_batch_size"],
        "lr": train["lr"], "lr_min": train["lr_min"], "pica_alignment_enabled": False,
        "augmentation_policy": recipe,
        "augmentation_level": {"z4": 0, "z3a": 1, "z6b": 5}[recipe],
        "state_noise_std": float(os.environ["LINGBOT_STATE_NOISE_STD"]),
        "initial_optimizer_state": "not supplied; new Muon state, training steps start at 0",
    }
    if z6b_config is not None:
        provenance.update(z6b_config=z6b_config, instruction_pool_sha256=pool_sha256,
                          instruction_pool=str(launch / "instruction_pool_v1.json.gz"),
                          lr_decay_ratio=train["lr_decay_ratio"], image_augment=True,
                          reference=json.loads((ROOT / "experiment/robotwin/records/z6b_source.json").read_text()))
    if z4_config is not None:
        provenance["z4_config"] = z4_config
    source_manifest = checkpoint.parent / "download_manifest.json"
    if source_manifest.is_file():
        source_record = json.loads(source_manifest.read_text())
        if isinstance(source_record, dict) and source_record.get("all_files_verified"):
            provenance["initial_checkpoint_download"] = {
                key: source_record[key] for key in ("repo", "revision", "checkpoint")
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
        if (info["codebase_version"] != "v3.0" or info["total_episodes"] != 2500
                or (recipe == "z6b" and (info["fps"] != 15 or info["total_frames"] != 548893
                    or info["features"]["observation.images.cam_high"]["shape"][-2:] != [240, 320]))):
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
