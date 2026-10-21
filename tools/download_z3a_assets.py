"""Resume and SHA-256 verify the official depth/video teacher checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    dict(repo="Ruicheng/moge-2-vitb-normal", revision="ca5f0e07ff01d3e5a364c1d954ed12ee1814b368",
         remote="model.pt", local="weights/depth/moge2-vitb-normal.pt", size=419110160,
         sha256="16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6"),
    dict(repo="robbyant/lingbot-vla-v2-6b", revision="11c703bf6a5c1f45b3b69168482da11fdbba53d7",
         remote="depth/model.pt", local="weights/depth/model.pt", size=1316220456,
         sha256="d70c5191eab853d436763b35d40ff99d13534b4bcd43e4d02823656968159e5b"),
    dict(repo="robbyant/lingbot-vla-v2-6b", revision="11c703bf6a5c1f45b3b69168482da11fdbba53d7",
         remote="dino_video/teacher_step_10000.pth", local="weights/dino_video/teacher_step_10000.pth", size=1401509792,
         sha256="086285efd8d65bc66e96b363807c4010ee5c790b7452b765edaa23837a63705b"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset):
    target = ROOT / asset["local"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == asset["size"] and sha256(target) == asset["sha256"]:
        print(f"Verified existing: {asset['local']}", flush=True)
        return asset
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset["size"]:
            partial.rename(partial.with_name(partial.name + f".invalid-{time.time_ns()}"))
            offset = 0
        try:
            if offset < asset["size"]:
                url = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{asset['remote']}"
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                print(f"Downloading {asset['local']}: {offset}/{asset['size']} bytes", flush=True)
                with requests.get(url, params={"download": "true", "cache_bust": time.time_ns()},
                                  headers=headers, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Server returned an incorrect resume offset")
                    elif response.status_code == 200:
                        offset = 0  # Range unsupported: restart rather than append a full response.
                    else:
                        raise RuntimeError(f"Unexpected HTTP status {response.status_code}")
                    last_report = time.monotonic()
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            stream.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last_report > 20:
                                print(f"{asset['local']}: {offset / asset['size']:.1%}", flush=True)
                                last_report = time.monotonic()
            if partial.stat().st_size != asset["size"]:
                raise RuntimeError("Download size mismatch")
            if sha256(partial) != asset["sha256"]:
                partial.rename(partial.with_name(partial.name + f".invalid-{time.time_ns()}"))
                raise RuntimeError("SHA-256 mismatch; saved invalid file and will retry from zero")
            partial.replace(target)
            print(f"SHA-256 verified: {asset['local']}", flush=True)
            return asset
        except (requests.RequestException, RuntimeError) as exc:
            # Avoid logging signed redirect URLs or authorization material.
            print(f"Retry {attempt}/8 for {asset['local']}: {type(exc).__name__}", flush=True)
            if attempt == 8:
                raise RuntimeError(f"Could not download {asset['local']}") from None
            time.sleep(min(attempt * 2, 10))


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, ASSETS))
    # The small DINO config is a regular Git blob rather than an LFS object.
    revision = ASSETS[1]["revision"]
    response = requests.get(f"https://huggingface.co/robbyant/lingbot-vla-v2-6b/resolve/{revision}/dino_video/config.yaml", timeout=60)
    response.raise_for_status()
    data = response.content
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != "9223ab64a8e0081cd46446381dc4e6dce3a23080":
        raise RuntimeError("DINO configuration Git blob hash mismatch")
    config = ROOT / "weights/dino_video/config.yaml"
    config.write_bytes(data)
    verified.append(dict(local=str(config.relative_to(ROOT)), revision=revision,
                         size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (ROOT / "weights/z3a_assets_verified.json").write_text(json.dumps(verified, indent=2) + "\n")
    print("All depth/video assets verified.", flush=True)


if __name__ == "__main__":
    main()
