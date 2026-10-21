"""Training-only PICA action alignment and sharded FP32 EMA utilities."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_pica_config(config, num_layers):
    required = {"student_layer", "teacher_layer", "weight", "ema_decay", "projector_hidden_dim"}
    if set(config) != required:
        raise ValueError(f"pica_params must contain exactly {sorted(required)}")
    if not 0 <= config["student_layer"] < config["teacher_layer"] < num_layers:
        raise ValueError(f"PICA requires 0 <= student_layer < teacher_layer < {num_layers}")
    if not math.isfinite(config["weight"]) or config["weight"] <= 0:
        raise ValueError("PICA weight must be positive and finite")
    if not 0 <= config["ema_decay"] < 1 or config["projector_hidden_dim"] <= 0:
        raise ValueError("Invalid PICA EMA decay or projector width")


def attach_pica_projector(model, config):
    layers = model.model.qwenvl_with_expert.qwen_expert.model.layers
    validate_pica_config(config, len(layers))
    model.config.pica_params = dict(config)
    width = model.config.proj_width
    ref = next(model.parameters())
    model.pica_projector = nn.Sequential(
        nn.Linear(width, config["projector_hidden_dim"]), nn.GELU(),
        nn.Linear(config["projector_hidden_dim"], width),
    ).to(device=ref.device, dtype=ref.dtype)


def action_alignment_loss(projected, target, action_is_pad=None, joint_mask=None):
    """Negative cosine on valid action tokens only; teacher is always detached."""
    if projected.shape != target.shape or projected.ndim != 3:
        raise ValueError("PICA expects matching [batch, action_steps, hidden] features")
    valid = torch.ones(projected.shape[:2], device=projected.device, dtype=torch.bool)
    if action_is_pad is not None:
        valid &= ~action_is_pad.bool()
    if joint_mask is not None:
        valid &= joint_mask.bool().any(dim=-1)
    similarity = F.cosine_similarity(projected.float(), target.detach().float(), dim=-1)
    # No valid tokens gives a differentiable zero.
    return -(similarity * valid).sum() / valid.sum().clamp_min(1)


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


@torch.no_grad()
def update_pica_teacher(teacher, student, decay):
    """Update identical FSDP2 local shards, without materializing full weights.

    Teacher master weights stay FP32; FSDP may cast gathered forward weights to
    BF16. Call after optimizer.step, once per optimizer update (not microbatch).
    Non-parameter buffers, including MoE routing bias, are copied from student.
    """
    source = dict(student.named_parameters())
    for name, target in teacher.named_parameters():
        value = source[name]
        if hasattr(target, "placements"):
            if target.placements != value.placements or target.device_mesh != value.device_mesh:
                raise ValueError(f"PICA EMA shard placement mismatch: {name}")
        target, value = _local(target), _local(value.detach())
        if target.dtype != torch.float32 or target.shape != value.shape:
            raise ValueError(f"PICA EMA requires matching FP32 teacher shards: {name}")
        target.lerp_(value.float(), 1.0 - decay)
    buffers = dict(student.named_buffers())
    for name, target in teacher.named_buffers():
        if name in buffers:
            _local(target).copy_(_local(buffers[name].detach()))
