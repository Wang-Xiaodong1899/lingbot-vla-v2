"""Lingbot visual dynamics and velocity-only online gradient guidance.

All coordinates are the base policy's normalized absolute action coordinates.
The base is frozen; OGG differentiates only a detached velocity tensor.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from .vendor.MLP import SiglipResidualMLP
from .vendor.config import SiglipMLPConfig, ModelScale


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_metadata(base_model, norm_stats, train_list=None, qwen_path=None):
    base = Path(base_model).resolve()
    result = dict(base_model=str(base), base_config_sha256=sha256(base / 'config.json'),
                  base_index_sha256=sha256(base / 'model.safetensors.index.json'),
                  norm_stats=str(Path(norm_stats).resolve()), norm_stats_sha256=sha256(norm_stats),
                  visual_dtype='float32', action_representation='native_normalized_absolute_55_masked',
                  feature_representation='qwen3vl_merged_spatial_tokens_all_views',
                  robot_config_sha256=sha256(Path(__file__).resolve().parents[2] / 'configs/robot_configs/robotwin.yaml'))
    if train_list:
        result.update(train_list=str(Path(train_list).resolve()), train_list_sha256=sha256(train_list))
    if qwen_path:
        result['qwen_path'] = str(Path(qwen_path).resolve())
    return result


def check_base_metadata(metadata, base_model, norm_stats):
    current = base_metadata(base_model, norm_stats)
    for key, value in current.items():
        if metadata.get(key) != value:
            raise ValueError(f'Corrector/base metadata mismatch: {key}')


@torch.no_grad()
def visual_tokens(core, batch):
    """Exactly the merged image embeddings used by the frozen Lingbot prefix."""
    images, grid = batch['images'], batch['image_grid_thw']
    if images.ndim != 4 or grid.ndim != 3:
        raise ValueError(f'Expected batched Qwen pixels/grid, got {images.shape}, {grid.shape}')
    b, views = images.shape[:2]
    encoder = core.qwenvl_with_expert
    dtype = next(encoder.qwenvl.visual.parameters()).dtype
    tokens, _ = encoder.embed_image(images.flatten(0, 1).to(dtype), grid.flatten(0, 1))
    if tokens.shape[0] != b * views:
        raise ValueError('Unexpected number of camera token blocks')
    return tokens.reshape(b, -1, tokens.shape[-1]).float()


def quantize_tokens(z):
    z = np.asarray(z, dtype=np.float32)
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite visual token cache input')
    scale = np.maximum(np.max(np.abs(z), axis=-1) / 127, 1e-6).astype(np.float16)
    quantized = np.clip(np.rint(z / scale.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    return quantized, scale


def dequantize_tokens(z, scale):
    return np.asarray(z, dtype=np.float32) * np.asarray(scale, dtype=np.float32)[..., None]


def build_mlp(config):
    config = dict(config)
    if 'scale' in config:
        config['scale'] = ModelScale(config['scale'])
    return SiglipResidualMLP(SiglipMLPConfig(**config))


def load_corrector(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'lingbot-vla-corrector-v1':
        raise ValueError('Unsupported corrector checkpoint')
    model = build_mlp(payload['model_config'])
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).requires_grad_(False).eval(), payload


def cosine_error(prediction, actual, epsilon=1e-4):
    # Motionless intervals contain no identifiable residual direction.
    if actual.norm() < epsilon or prediction.norm() < epsilon:
        return None
    error = 1 - F.cosine_similarity(prediction.flatten()[None], actual.flatten()[None]).item()
    if not np.isfinite(error):
        raise FloatingPointError('Nonfinite dynamics error')
    return error


def guide_velocity(velocity, x_t, time, z, target, model, joint_mask, eta=0.1, max_grad_norm=1.0):
    """Match reference OGG: x0_hat=x_t-t*v; attract dynamics to pred-real."""
    with torch.enable_grad():
        v = velocity.detach().float().requires_grad_(True)
        x0 = x_t.detach().float() - time.detach().float()[:, None, None] * v
        action = x0[:, 0] * joint_mask.float()
        prediction = model(z.detach().float(), action)
        loss = (1 - F.cosine_similarity(prediction.flatten(1), target.detach().flatten(1), dim=-1)).mean()
        gradient, = torch.autograd.grad(loss, v)
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite OGG loss or velocity gradient')
        norm = gradient.norm()
        gradient = gradient * min(1.0, max_grad_norm / max(float(norm), 1e-12))
        result = velocity.detach().float() - eta * gradient
    return result.to(velocity.dtype), dict(loss=float(loss.detach()), gradient_norm=float(norm),
                                          delta_velocity_norm=float((eta * gradient).norm()))


@contextmanager
def online_guidance(core, model, z, target, joint_mask, eta=0.1, every=3):
    """Patch one sampling call; restore even on errors, with no base gradients."""
    if eta <= 0 or every < 1:
        raise ValueError('OGG needs positive eta and application interval')
    original = core.predict_velocity
    had_instance_value = 'predict_velocity' in core.__dict__
    instance_value = core.__dict__.get('predict_velocity')
    stats = []
    calls = 0

    def predict(self, state, prefix_pad_masks, past_key_values, x_t, timestep, **kwargs):
        nonlocal calls
        with torch.no_grad():
            velocity = original(state, prefix_pad_masks, past_key_values, x_t, timestep, **kwargs)
        if calls % every == 0:
            velocity, record = guide_velocity(velocity, x_t, timestep, z, target, model, joint_mask, eta)
            stats.append(record)
        calls += 1
        return velocity

    core.predict_velocity = types.MethodType(predict, core)
    try:
        yield stats
    finally:
        if had_instance_value:
            core.predict_velocity = instance_value
        else:
            del core.predict_velocity
