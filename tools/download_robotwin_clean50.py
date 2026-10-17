#!/usr/bin/env python3
"""Download and extract only dataset/**/aloha-agilex_clean_50.zip."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi, hf_hub_download


def download_and_extract(filename, output_dir, revision):
    task = filename.split("/")[1]
    print(f"[下载] {task}", flush=True)
    archive = Path(hf_hub_download(
        repo_id="TianxingChen/RoboTwin2.0",
        repo_type="dataset",
        filename=filename,
        revision=revision,
        local_dir=str(output_dir),
    ))
    destination = archive.parent
    marker = destination / ".aloha-agilex_clean_50.extracted.json"
    if marker.exists():
        try:
            previous = json.loads(marker.read_text())
            if previous.get("revision") == revision and all(
                (destination / name).is_file() for name in previous["files"]
            ):
                print(f"[跳过已解压] {task}", flush=True)
                return task
        except (ValueError, KeyError):
            pass

    print(f"[解压] {task}: {archive}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        members = zf.infolist()
        for member in members:
            if not (destination / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        # extractall verifies each extracted file's CRC. Interrupted extraction
        # has no completion marker and is safely retried on the next run.
        zf.extractall(destination)
        files = [member.filename for member in members if not member.is_dir()]
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(json.dumps({
        "revision": revision,
        "archive": filename,
        "files": files,
    }, ensure_ascii=False, indent=2) + "\n")
    temporary_marker.replace(marker)
    print(f"[完成] {task}: {len(files)} 个文件", flush=True)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=
                        Path(__file__).resolve().parents[1] / "data" / "RoboTwin2.0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision = api.repo_info("TianxingChen/RoboTwin2.0", repo_type="dataset",
                             revision=args.revision).sha
    filenames = sorted(filename for filename in api.list_repo_files(
        "TianxingChen/RoboTwin2.0", repo_type="dataset", revision=revision
    ) if filename.startswith("dataset/")
        and filename.rsplit("/", 1)[-1] == "aloha-agilex_clean_50.zip")
    if not filenames:
        raise RuntimeError("No matching clean-50 archives found")
    (args.output_dir / "download_manifest.json").write_text(json.dumps({
        "repo_id": "TianxingChen/RoboTwin2.0",
        "revision": revision,
        "filenames": filenames,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"共 {len(filenames)} 个 ZIP，并发 {args.workers}，目录 {args.output_dir}", flush=True)
    started = time.monotonic()
    failed = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_and_extract, filename,
                                   args.output_dir, revision): filename
                   for filename in filenames}
        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
                completed += 1
                print(f"[进度] {completed}/{len(filenames)} 已下载并解压", flush=True)
            except Exception as exc:
                failed.append({"filename": filename, "error": str(exc)})
                print(f"[失败] {filename}: {exc}", flush=True)
    (args.output_dir / "download_status.json").write_text(json.dumps({
        "revision": revision,
        "total": len(filenames),
        "completed": completed,
        "failed": failed,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"结束：完成 {completed}/{len(filenames)}，失败 {len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
