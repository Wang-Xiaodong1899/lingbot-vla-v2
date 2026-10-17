"""Read local LeRobot v2.1 episodes without depending on installed LeRobot API.

Uses the repository's existing FeatureTransform and video decoder so normalized
actions and Qwen inputs retain the base policy's semantics.
"""

from collections import OrderedDict
import bisect
import json
import io
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

from .utils import FeatureTransform
from .video_utils import decode_video_frames


class LocalV2RefinementDataset(Dataset):
    def __init__(self, repo_file, data_config, config, processor):
        self.horizon = config.n_action_steps
        self.tasks = []
        self.episodes = []
        self.ends = []
        self.cache = OrderedDict()
        self.resize = Resize((data_config.img_size, data_config.img_size))
        total = 0
        for line in Path(repo_file).read_text().splitlines():
            if not line.strip():
                continue
            name, path = line.split(maxsplit=1)
            root = Path(path).expanduser().resolve()
            info = json.loads((root / "meta/info.json").read_text())
            if info["codebase_version"] != "v2.1":
                raise ValueError(f"Expected local LeRobot v2.1 data: {root}")
            transform = FeatureTransform(
                str(Path(data_config.robot_config_root) / f"{name}.yaml"), data_config,
                config, processor, chunk_size=self.horizon, use_depth_align=False,
                image_augment=False, use_future_image=False,
                norm_stats_path=data_config.norm_stats_file,
            )
            if transform.actions_convert_from_state:
                raise ValueError("This adapter expects recorded actions, not actions converted from states")
            task_names = {row["task_index"]: row["task"] for row in self._jsonl(root / "meta/tasks.jsonl")}
            task = {"root": root, "info": info, "transform": transform, "task_names": task_names,
                    "episodes": sorted(self._jsonl(root / "meta/episodes.jsonl"), key=lambda row: row["episode_index"])}
            task_index = len(self.tasks)
            self.tasks.append(task)
            for episode in task["episodes"]:
                self.episodes.append({**episode, "task": task_index, "start": total})
                total += episode["length"]
                self.ends.append(total)
            if sum(row["length"] for row in task["episodes"]) != info["total_frames"]:
                raise ValueError(f"Invalid episode lengths: {root}")

    @staticmethod
    def _jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = self.episodes[bisect.bisect_right(self.ends, index)]
        task = self.tasks[episode["task"]]
        episode_id = episode["episode_index"]
        frame = index - episode["start"]
        info, root, transform = task["info"], task["root"], task["transform"]
        substitutions = dict(episode_index=episode_id, episode_chunk=episode_id // info["chunks_size"])
        parquet_path = root / info["data_path"].format(**substitutions)
        key = str(parquet_path)
        if key not in self.cache:
            self.cache[key] = pq.read_table(parquet_path).to_pydict()
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        columns = self.cache[key]
        self.cache.move_to_end(key)
        if len(columns["timestamp"]) != episode["length"]:
            raise ValueError(f"Parquet length mismatch: {parquet_path}")
        raw = {key: torch.as_tensor(value[frame]) for key, value in columns.items()
               if key not in transform.org_features["images"]}
        future = np.arange(frame, frame + self.horizon)
        pad = torch.from_numpy(future >= episode["length"])
        future = np.minimum(future, episode["length"] - 1)
        for action_key in transform.org_features["actions"]:
            raw[action_key] = torch.as_tensor(np.asarray(columns[action_key], dtype=np.float32)[future])
            raw[action_key + "_is_pad"] = pad
        raw["task"] = task["task_names"][int(raw["task_index"])]
        for image_key in transform.org_features["images"]:
            image_type = info["features"][image_key]["dtype"]
            if image_type == "video":
                video_path = root / info["video_path"].format(video_key=image_key, **substitutions)
                image = decode_video_frames(video_path, [float(raw["timestamp"])], tolerance_s=1e-4)[0]
            elif image_type == "image":
                entry = columns[image_key][frame]
                source = io.BytesIO(entry["bytes"]) if entry.get("bytes") is not None else root / entry["path"]
                with Image.open(source) as decoded:
                    image = pil_to_tensor(decoded.convert("RGB")).float() / 255.0
            else:
                raise ValueError(f"Unsupported image feature {image_key}: {image_type}")
            raw[image_key] = self.resize(image)
        return transform.apply(raw)
