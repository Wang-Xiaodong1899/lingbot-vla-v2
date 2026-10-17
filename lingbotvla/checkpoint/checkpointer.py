# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import gc
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from ..utils.import_utils import is_torch_version_greater_than
from ..utils.logging import get_logger
from pathlib import Path
from .integrity import validate_checkpoint, fsync_directory

if is_torch_version_greater_than("2.4"):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import (
        FileSystemReader,
        FileSystemWriter,
    )
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed._tensor import DTensor, Shard as DTensorShard
else:
    Stateful = ABC

logger = get_logger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_MODEL_DIR = "model"
_EMA_DIR = "ema"
_OPTIMIZER_DIR = "optimizer"
_EXTRA_STATE_DIR = "extra_state"


def _checkpoint_phase(path, phase):
    """Keep phase/memory evidence on disk even if a process is SIGKILLed."""
    record = {'time': time.time(), 'rank': dist.get_rank(), 'phase': phase, 'checkpoint': str(path)}
    for name in ('memory.current', 'memory.high', 'memory.max', 'memory.events'):
        p = Path('/sys/fs/cgroup') / name
        if p.exists():
            record[name] = p.read_text().strip()
    if torch.cuda.is_available():
        record['cuda_allocated'] = torch.cuda.memory_allocated()
        record['cuda_reserved'] = torch.cuda.memory_reserved()
    root = Path(path).parent
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'checkpoint_memory_rank_{dist.get_rank()}.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()
        os.fsync(f.fileno())


class ModelState(Stateful):
    """
    A wrapper around a model to make it stateful.
    EP-aware: restores/drops EP dimension for DCP save/load when FSDP2+EP is active.
    Args:
        model (Model): model to wrap.
    """

    def __init__(self, model):
        self.model = model
        self._fqn2spec_info = getattr(model, '_fqn2spec_info', None)
        self._parallel_state = None
        self._should_ep_aware = False
        if self._fqn2spec_info is not None:
            from ..distributed.parallel_state import get_parallel_state
            self._parallel_state = get_parallel_state()
            self._should_ep_aware = (
                self._parallel_state.dp_mode == 'fsdp2'
                and self._parallel_state.ep_enabled
            )

    @torch.no_grad()
    def state_dict(self):
        model_state_dict = get_model_state_dict(model=self.model)
        if self._should_ep_aware:
            logger.info_rank0("ModelState: restoring EP dim for DCP save")
            model_state_dict = self._preprocess_ep_dim(model_state_dict, action='restore')
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        model_state_dict = state_dict["model"]
        if self._should_ep_aware:
            logger.info_rank0("ModelState: dropping EP dim for DCP load")
            model_state_dict = self._preprocess_ep_dim(model_state_dict, action='drop')
        set_model_state_dict(model=self.model, model_state_dict=model_state_dict)

    def _preprocess_ep_dim(self, state_dict, action):
        """Process EP dimension for save/load.
        restore: local tensor -> DTensor with EP+FSDP placements (for save)
        drop: DTensor with EP+FSDP placements -> local tensor (for load)
        """
        fqn2spec_info = self._fqn2spec_info
        ep_fsdp_mesh = self._parallel_state.ep_fsdp_device_mesh

        for name in sorted(state_dict.keys()):
            if name not in fqn2spec_info:
                continue
            spec = fqn2spec_info[name]
            if not isinstance(spec.placement, DTensorShard):
                continue
            tensor = state_dict[name]
            if action == 'restore':
                state_dict[name] = _restore_ep_dim(tensor, ep_fsdp_mesh, ep_fsdp_mesh["ep_fsdp"])
            elif action == 'drop':
                state_dict[name] = _drop_ep_dim(tensor, ep_fsdp_mesh["ep_fsdp"])
        return state_dict


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.
    EP-aware: restores/drops EP dimension for optimizer state DCP save/load.

    Args:
        model (Model): model to wrap.
        optimizer (Optimizer): optimizer to wrap.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self._fqn2spec_info = getattr(model, '_fqn2spec_info', None)
        self._parallel_state = None
        self._should_ep_aware = False
        if self._fqn2spec_info is not None:
            from ..distributed.parallel_state import get_parallel_state
            self._parallel_state = get_parallel_state()
            self._should_ep_aware = (
                self._parallel_state.dp_mode == 'fsdp2'
                and self._parallel_state.ep_enabled
            )

    def state_dict(self):
        optimizers = getattr(self.optimizer, "optimizers", self.optimizer)
        optimizer_state_dict = get_optimizer_state_dict(model=self.model, optimizers=optimizers)
        if self._should_ep_aware:
            logger.info_rank0("OptimizerState: restoring EP dim for DCP save")
            optimizer_state_dict = self._preprocess_ep_dim(optimizer_state_dict, action='restore')
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        optim_state_dict = state_dict["optim"]
        if self._should_ep_aware:
            logger.info_rank0("OptimizerState: dropping EP dim for DCP load")
            optim_state_dict = self._preprocess_ep_dim(optim_state_dict, action='drop')
        optimizers = getattr(self.optimizer, "optimizers", self.optimizer)
        set_optimizer_state_dict(model=self.model, optimizers=optimizers, optim_state_dict=optim_state_dict)

    def _preprocess_ep_dim(self, state_dict, action):
        """Process EP dimension for optimizer state save/load."""
        fqn2spec_info = self._fqn2spec_info
        ep_fsdp_mesh = self._parallel_state.ep_fsdp_device_mesh
        ep_fqn_keys = list(fqn2spec_info.keys())

        for name in sorted(state_dict.keys()):
            # Match optimizer state keys to EP FQNs (e.g. "state.model.layers.0.mlp.experts.gate_proj.step")
            matches = [k for k in ep_fqn_keys if k in name]
            if not matches:
                continue
            assert len(matches) == 1, f"Ambiguous EP spec match for optimizer key '{name}': {matches}"
            spec = fqn2spec_info[matches[0]]
            if not isinstance(spec.placement, DTensorShard):
                continue
            tensor = state_dict[name]
            if not torch.is_tensor(tensor) or tensor.ndim == 0:
                continue
            if action == 'restore':
                state_dict[name] = _restore_ep_dim(tensor, ep_fsdp_mesh, ep_fsdp_mesh["ep_fsdp"])
            elif action == 'drop':
                state_dict[name] = _drop_ep_dim(tensor, ep_fsdp_mesh["ep_fsdp"])
        return state_dict


def _drop_ep_dim(loaded_tensor, ep_fsdp_mesh):
    """Drop EP dim after loading from DCP so that EP-FSDP would not be confused."""
    if isinstance(loaded_tensor, DTensor):
        if len(loaded_tensor.placements) == 2:
            # EP+FSDP: keep only FSDP shard
            return DTensor.from_local(
                loaded_tensor._local_tensor, device_mesh=ep_fsdp_mesh, placements=[DTensorShard(1)]
            )
        elif len(loaded_tensor.placements) == 1:
            # EP only: unwrap to local
            return loaded_tensor.to_local()
    # Already a local tensor
    return loaded_tensor


def _restore_ep_dim(orig_tensor, global_mesh, ep_fsdp_mesh):
    """Restore EP dim so that DCP can be aware about EP ranks.
    global_mesh: 2D mesh with (ep, ep_fsdp) dimensions
    ep_fsdp_mesh: 1D mesh for ep_fsdp dimension
    """
    if isinstance(orig_tensor, DTensor):
        # EP+FSDP: restore both dimensions
        return DTensor.from_local(
            orig_tensor._local_tensor, device_mesh=global_mesh, placements=[DTensorShard(0), DTensorShard(1)]
        )
    elif torch.is_tensor(orig_tensor):
        # EP only (no FSDP): single EP shard
        return DTensor.from_local(orig_tensor, device_mesh=ep_fsdp_mesh, placements=[DTensorShard(0)])
    else:
        raise RuntimeError(f"orig_tensor {orig_tensor} is not a tensor!")


def build_checkpointer(
    dist_backend: str = "fsdp1",
    ckpt_manager: str = "bytecheckpoint",
):
    """
    create a checkpointer manager with given mode.
    Args:
        dist_backend (str, optional): checkpoint mode. Defaults to "fsdp1".
            fsdp1: FSDP1 checkpoint from bytecheckpoint
            fsdp2-vescale: FSDP2 checkpoint from bytecheckpoint
            fsdp2: FSDP2 checkpoint from bytecheckpoint
            ddp: DDP checkpoint from bytecheckpoint
            dcp: DCP checkpoint from torch.distributed.checkpoint
        ckpt_manager (str, optional): checkpoint manager. Defaults to "bytecheckpoint".
            bytecheckpoint: bytecheckpoint checkpoint manager
            dcp: torch dcp checkpoint manager
    Raises:
        ValueError: if ckpt_manager is not supported

    Returns:
        Checkpointer: checkpointer with given mode.
    """

    if ckpt_manager == "bytecheckpoint":
        if dist_backend == "ddp":
            from bytecheckpoint import DDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp1":
            from bytecheckpoint import FSDPCheckpointer as Checkpointer
        elif dist_backend == "fsdp2-vescale":
            from bytecheckpoint import VeScaleCheckpointer as Checkpointer
        elif dist_backend == "fsdp2":
            from bytecheckpoint import FSDP2Checkpointer as Checkpointer
    elif ckpt_manager == "dcp":
        if not is_torch_version_greater_than("2.4"):
            raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
        if dist_backend not in ["ddp", "fsdp1", "fsdp2"]:
            raise ValueError(
                f"Unsupported distributed backend: {dist_backend} for DCP checkpoint manager, supported modes are: ddp, fsdp1, fsdp2"
            )
        Checkpointer = DistributedCheckpointer
    else:
        raise ValueError(
            f"Unknown checkpoint manager: {ckpt_manager}, supported modes are: bytecheckpoint, dcp, native"
        )

    return Checkpointer


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return

    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        allow_partial_load: bool = False,
    ):
        return


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    @classmethod
    def save(
        cls, path: str, state: Dict[str, Any], global_steps: int = None, save_async=False,
    ) -> None:
        # Do not report completion for fire-and-forget asynchronous futures.
        if save_async:
            raise ValueError('Transactional DCP saving requires save_async=False')
        final = Path(f'{path}/global_step_{global_steps}' if global_steps else path)
        rank = dist.get_rank()
        payload = [None]
        if rank == 0:
            payload[0] = {'error': f'Refusing to overwrite checkpoint: {final}'} if final.exists() else {
                'staging': str(final.with_name(f'.{final.name}.incomplete-{uuid.uuid4().hex}'))}
        dist.broadcast_object_list(payload, src=0)
        if 'error' in payload[0]:
            raise FileExistsError(payload[0]['error'])
        staging = Path(payload[0]['staging'])
        staging.mkdir(parents=True, exist_ok=True)
        gc.collect()
        cls._save_uncommitted(str(staging), state, global_steps=None, save_async=False)
        dist.barrier()
        result = [None]
        if rank == 0:
            try:
                info = validate_checkpoint(staging, dist.get_world_size(), 'optimizer' in state, 'extra_state' in state)
                # torch.save(extra_state) does not fsync on its own.
                for p in (staging / _EXTRA_STATE_DIR).glob('*.pt'):
                    with p.open('rb') as f:
                        os.fsync(f.fileno())
                for p in staging.iterdir():
                    if p.is_dir():
                        fsync_directory(p)
                with (staging / '_SUCCESS.json').open('w') as f:
                    json.dump({**info, 'global_step': global_steps, 'saved_at': time.time()}, f)
                    f.flush()
                    os.fsync(f.fileno())
                fsync_directory(staging)
                os.replace(staging, final)
                fsync_directory(final.parent)
                result[0] = {'ok': True}
            except Exception as exc:
                result[0] = {'error': repr(exc)}
        dist.broadcast_object_list(result, src=0)
        if 'error' in result[0]:
            raise RuntimeError(f'Checkpoint not committed; original checkpoints preserved: {result[0]["error"]}')
        dist.barrier()
        logger.info_rank0(f'Committed complete checkpoint to {final}')

    @classmethod
    def _save_uncommitted(
        cls,
        path: str,
        state: Dict[str, Any],
        global_steps: int = None,
        save_async=False,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            global_steps: global steps
            save_async: whether to save asynchronously
        return:
            None
        """

        checkpoint_dir = f"{path}/global_step_{global_steps}" if global_steps else path
        os.makedirs(checkpoint_dir, exist_ok=True)

        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        if save_async:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.async_save(
                state_dict={"state": ModelState(state["model"])},
                storage_writer=FileSystemWriter(
                    model_dir,
                    thread_count=int(os.environ.get('LINGBOT_DCP_THREADS', '1')),
                    single_file_per_rank=True,
                    sync_files=True,
                ),
            )
            if "ema" in state and state["ema"] is not None:
                ema_dir = os.path.join(checkpoint_dir, _EMA_DIR)
                dcp.async_save(
                    state_dict={"state": ModelState(state["ema"])},
                    storage_writer=FileSystemWriter(
                        ema_dir,
                        thread_count=int(os.environ.get('LINGBOT_DCP_THREADS', '1')),
                        single_file_per_rank=True,
                        sync_files=True,
                    ),
                )
            if "optimizer" in state:
                optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
                dcp.async_save(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_writer=FileSystemWriter(
                        optimizer_dir,
                        thread_count=int(os.environ.get('LINGBOT_DCP_THREADS', '1')),
                        single_file_per_rank=True,
                        sync_files=True,
                    ),
                )
        else:
            def safe_create_writer(output_dir):
                tmp_path = Path(output_dir) / ".metadata.tmp"
                if tmp_path.exists():
                    print(f"Warning: removing existing tmp file: {tmp_path}")
                    tmp_path.unlink()  # remove .metadata.tmp
                return FileSystemWriter(
                    output_dir,
                    thread_count=int(os.environ.get('LINGBOT_DCP_THREADS', '1')),
                    single_file_per_rank=True,
                    sync_files=True,
                )
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            storage_writer = safe_create_writer(model_dir)
            _checkpoint_phase(checkpoint_dir, 'model_start')
            dcp.save(
                state_dict={"state": ModelState(state["model"])},
                storage_writer=storage_writer,
            )
            gc.collect()
            _checkpoint_phase(checkpoint_dir, 'model_done')
            if "ema" in state and state["ema"] is not None:
                ema_dir = os.path.join(checkpoint_dir, _EMA_DIR)
                storage_writer = safe_create_writer(ema_dir)
                dcp.save(
                    state_dict={"state": ModelState(state["ema"])},
                    storage_writer=storage_writer,
                )
            if "optimizer" in state:
                optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
                _checkpoint_phase(checkpoint_dir, 'optimizer_start')
                dcp.save(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])},
                    storage_writer=FileSystemWriter(
                        optimizer_dir,
                        thread_count=int(os.environ.get('LINGBOT_DCP_THREADS', '1')),
                        single_file_per_rank=True,
                        sync_files=True,
                    ),
                )
                gc.collect()
                _checkpoint_phase(checkpoint_dir, 'optimizer_done')
                # dist.barrier()

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            torch.save(
                state["extra_state"],
                extra_state_path,
            )
            _checkpoint_phase(checkpoint_dir, 'extra_state_done')

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        allow_partial_load: bool = False,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            allow_partial_load: if True, skip missing keys in checkpoint (useful for model structure changes)

        return:
            state: state loaded
        """
        checkpoint_dir = path

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        model_planner = DefaultLoadPlanner(allow_partial_load=True) if allow_partial_load else None

        if "ema" in state and state["ema"] is not None:
            ema_dir = os.path.join(checkpoint_dir, _EMA_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["ema"])},
                storage_reader=FileSystemReader(ema_dir),
                planner=model_planner,
                process_group=process_group,
            )

        if "optimizer" in state:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["model"])},
                storage_reader=FileSystemReader(model_dir),
                planner=model_planner,
                process_group=process_group,
            )

            optimizer_dir = os.path.join(checkpoint_dir, _OPTIMIZER_DIR)
            try:
                dcp.load(
                    state_dict={"state": OptimizerState(model=state["model"], optimizer=state["optimizer"])}, # 1043
                    storage_reader=FileSystemReader(optimizer_dir), # 1027
                    planner=DefaultLoadPlanner(allow_partial_load=allow_partial_load),
                    process_group=process_group,
                )
            except Exception:
                if not allow_partial_load:
                    raise
                logger.warning_rank0(f"Skip loading Optimizer from {checkpoint_dir} (partial load explicitly enabled)")
        else:
            model_dir = os.path.join(checkpoint_dir, _MODEL_DIR)
            dcp.load(
                state_dict={"state": ModelState(state["model"])},
                storage_reader=FileSystemReader(model_dir),
                planner=model_planner,
                process_group=process_group,
            )

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            state["extra_state"] = torch.load(
                extra_state_path,
                weights_only=False,
            )

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state
