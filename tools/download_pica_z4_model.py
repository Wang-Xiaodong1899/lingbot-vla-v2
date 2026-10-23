"""Download only PICA step 2000 and its training metadata at a pinned HF revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests

from download_z3a_assets import download

REPO = "Xiaodong/lingbotvla-robotwin-clean-sft-20k-PICA"
REVISION = "892824afa4cf7d77ad5d3916045d884f14d37170"
DEFAULT_DEST = "/root/autodl-tmp/models/Xiaodong-lingbotvla-robotwin-clean-sft-20k-PICA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=Path(DEFAULT_DEST))
    args = parser.parse_args()
    destination = args.local_dir.resolve()
    response = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}",
                            params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION:
        raise RuntimeError("Hugging Face revision mismatch")
    files = [f for f in metadata["siblings"]
             if f["rfilename"].startswith("global_step_2000/")
             or f["rfilename"] in ("lingbotvla_cli.yaml", "robotwin_clean_norm.json")
             or f["rfilename"].startswith("launch/robot_configs/")]
    if not any(f["rfilename"] == "global_step_2000/model.safetensors.index.json" for f in files):
        raise RuntimeError("Requested step 2000 checkpoint is missing")
    records = []
    large = []
    for entry in files:
        remote = entry["rfilename"]
        target = destination / remote
        if not target.resolve().is_relative_to(destination):
            raise RuntimeError("Invalid repository file path")
        asset = dict(repo=REPO, revision=REVISION, remote=remote, local=str(target), size=entry["size"])
        if "lfs" in entry:
            asset["sha256"] = entry["lfs"]["sha256"]
            large.append(asset)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        content = target.read_bytes() if target.is_file() else b""
        def git_hash(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(content) != entry["size"] or git_hash(content) != entry["blobId"]:
            response = requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}", timeout=60)
            response.raise_for_status()
            content = response.content
            if len(content) != entry["size"] or git_hash(content) != entry["blobId"]:
                raise RuntimeError(f"Git blob verification failed: {remote}")
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(content)
            partial.replace(target)
        asset.update(sha256=hashlib.sha256(content).hexdigest(), git_blob=entry["blobId"])
        records.append(asset)
        print(f"Git blob verified: {remote}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records.extend(pool.map(download, large))
    record = dict(repo=REPO, revision=REVISION, checkpoint="global_step_2000",
                  all_files_verified=True, files=sorted(records, key=lambda x: x["remote"]))
    manifest = destination / "download_manifest.json"
    partial = manifest.with_suffix(".json.part")
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.replace(manifest)
    print(f"All files verified; manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
