"""Online Z4 sensor/image corruptions; no rendering, generated content or asset pasting.

One clean/augmented decision per observation; parameters and noise fields are
shared over time within each camera. Occlusion uses the union of protected
regions over ALL frames, after applying the same affine transform as the RGB.
The default detector is conservative classical image analysis, not semantic
segmentation. Uncertain pixels are protected; no safe box means no occlusion.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
from io import BytesIO
import math
import os
from pathlib import Path
import zlib

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

# DataLoader already supplies process parallelism; avoid nested OpenCV pools.
cv2.setNumThreads(1)


@dataclass(frozen=True)
class Z4Config:
    clean_probability: float = 0.10
    color_probability: float = 0.80
    brightness: tuple = (0.25, 1.10)
    contrast: tuple = (0.50, 1.50)
    saturation: tuple = (0.40, 1.60)
    temperature_shift: float = 0.06
    affine_probability: float = 0.50
    scale: tuple = (0.90, 1.10)
    translation: float = 0.06
    erase_probability: float = 0.95
    erase_area: tuple = (0.01, 0.06)
    erase_aspect: tuple = (0.40, 2.50)
    erase_attempts: int = 64
    mosaic_block: int = 8
    foreground_margin: int = 5
    protected_center: float = 0.60
    gaussian_noise_probability: float = 0.50
    salt_pepper_probability: float = 0.15
    gaussian_noise_std: tuple = (0.005, 0.020)
    salt_pepper_amount: tuple = (0.001, 0.010)
    gaussian_blur_probability: float = 0.25
    motion_blur_probability: float = 0.10
    gaussian_blur_sigma: tuple = (0.30, 1.00)
    motion_blur_sizes: tuple = (3, 5, 7)
    jpeg_probability: float = 0.30
    jpeg_quality: tuple = (35, 90)

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith('probability') and not 0 <= value <= 1:
                raise ValueError(f'{field.name} must be in [0,1]')
            if isinstance(value, (list, tuple)) and field.name != 'motion_blur_sizes':
                if len(value) != 2 or not 0 <= value[0] <= value[1]:
                    raise ValueError(f'Invalid range: {field.name}')
        if self.color_probability + self.erase_probability < 1:
            raise ValueError('Color + erase probabilities must cover the augmented branch')
        if self.gaussian_noise_probability + self.salt_pepper_probability > 1:
            raise ValueError('Gaussian / salt-pepper probabilities must sum to <= 1')
        if self.gaussian_blur_probability + self.motion_blur_probability > 1:
            raise ValueError('Gaussian / motion blur probabilities must sum to <= 1')
        if not (0 < self.scale[0] and 0 < self.erase_aspect[0] and 0 < self.erase_area[0] <= self.erase_area[1] <= 1):
            raise ValueError('Invalid scale, aspect or erase area')
        if not (0 <= self.translation <= 1 and 0 <= self.temperature_shift < 1 and 0 <= self.protected_center < 1):
            raise ValueError('Invalid translation, temperature or center guard')
        if self.salt_pepper_amount[1] > 1 or not 1 <= self.jpeg_quality[0] <= self.jpeg_quality[1] <= 100:
            raise ValueError('Invalid salt/pepper amount or JPEG quality')
        if self.foreground_margin < 0 or self.erase_attempts < 1 or self.mosaic_block < 1:
            raise ValueError('Invalid foreground/occlusion settings')
        if not self.motion_blur_sizes or any(k < 3 or k % 2 == 0 for k in self.motion_blur_sizes):
            raise ValueError('Motion blur kernel sizes must be odd and >= 3')


@lru_cache(maxsize=8)
def load_config(path=None):
    if path is None:
        return Z4Config()
    values = yaml.safe_load(Path(path).read_text())
    return Z4Config(**values)


def sample_plan(rng, config):
    """Exact marginal probabilities, with at least one requested operation.

    Color and erase share one uniform draw: color if u<.8, erase if u>=.05.
    Both marginals stay exact and their union covers every augmented sample.
    Other families use independent draws; variants within noise/blur are exclusive.
    Erase can still be skipped by the foreground safety check.
    """
    u = rng.random()
    noise, blur = rng.random(2)
    return dict(
        color=bool(u < config.color_probability),
        affine=bool(rng.random() < config.affine_probability),
        erase=bool(u >= 1 - config.erase_probability),
        noise=('gaussian' if noise < config.gaussian_noise_probability else
               'salt_pepper' if noise < config.gaussian_noise_probability + config.salt_pepper_probability else None),
        blur=('gaussian' if blur < config.gaussian_blur_probability else
              'motion' if blur < config.gaussian_blur_probability + config.motion_blur_probability else None),
        jpeg=bool(rng.random() < config.jpeg_probability),
    )


def detect_protected_regions(rgb, config):
    """Only flat, low-saturation, border-connected areas are eligible background.

    Edges and filled contours, colored regions, texture, and a central workspace
    guard are protected. This deliberately favors missed occlusions over masking
    a detected object. It cannot guarantee recognition of every task object.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 35, 90)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    protected = edges.copy()
    cv2.drawContours(protected, contours, -1, 255, thickness=cv2.FILLED)
    gray_f = gray.astype(np.float32)
    variance = np.maximum(0, cv2.blur(gray_f ** 2, (7, 7)) - cv2.blur(gray_f, (7, 7)) ** 2)
    protected[(hsv[..., 1] > 45) | (variance > 16)] = 255
    if config.protected_center:
        ch, cw = max(1, round(h * config.protected_center)), max(1, round(w * config.protected_center))
        y, x = (h - ch) // 2, (w - cw) // 2
        protected[y:y + ch, x:x + cw] = 255
    if config.foreground_margin:
        k = 2 * config.foreground_margin + 1
        protected = cv2.dilate(protected, np.ones((k, k), np.uint8))
    # Isolated flat object interiors are NOT assumed to be background.
    _, labels = cv2.connectedComponents((protected == 0).astype(np.uint8), connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]
    safe = (labels != 0) & np.isin(labels, border_labels)
    return ~safe


def affine_matrix(height, width, scale, dx, dy):
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 0, scale)
    matrix[:, 2] += (dx * width, dy * height)
    return matrix


def choose_box(protected, rng, config):
    """Reject every rectangle that touches a protected pixel."""
    h, w = protected.shape
    integral = cv2.integral(protected.astype(np.uint8))
    for _ in range(config.erase_attempts):
        area = rng.uniform(*config.erase_area) * h * w
        aspect = math.exp(rng.uniform(*np.log(config.erase_aspect)))
        bh, bw = max(1, round(math.sqrt(area / aspect))), max(1, round(math.sqrt(area * aspect)))
        if bh > h or bw > w:
            continue
        y, x = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        count = integral[y + bh, x + bw] - integral[y, x + bw] - integral[y + bh, x] + integral[y, x]
        if count == 0:
            return x, y, bw, bh
    return None


def augment_camera_clip(frames, rng, config, foreground_masks=None, plan=None):
    """uint8 CPU [T,3,H,W] -> same shape/dtype, without mutating the input."""
    if frames.device.type != 'cpu' or frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError('Z4 expects CPU uint8 RGB [T,3,H,W]')
    images = frames.permute(0, 2, 3, 1).numpy().copy()
    n, h, w, _ = images.shape
    plan = sample_plan(rng, config) if plan is None else dict(plan)
    trace = dict(plan, erase_applied=False, box=None)
    protected = None
    if plan['erase']:
        if foreground_masks is None:
            masks = np.stack([detect_protected_regions(frame, config) for frame in images])
        else:
            masks = np.asarray(foreground_masks, dtype=bool)
            if masks.shape != (n, h, w):
                raise ValueError('Foreground masks must have shape [T,H,W]')
            if config.foreground_margin:
                k = 2 * config.foreground_margin + 1
                masks = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0 for m in masks])
        protected = np.any(masks, axis=0)
    if plan['color']:
        brightness, contrast, saturation = (rng.uniform(*getattr(config, name)) for name in ('brightness', 'contrast', 'saturation'))
        temperature = rng.uniform(-config.temperature_shift, config.temperature_shift)
        x = images.astype(np.float32) / 255
        x *= brightness
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        x = (x - mean) * contrast + mean
        gray = (x * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1, keepdims=True)
        x = gray + (x - gray) * saturation
        x *= np.array([1 + temperature, 1, 1 - temperature], dtype=np.float32)
        images = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
        trace['color_params'] = dict(brightness=brightness, contrast=contrast, saturation=saturation, temperature=temperature)
    valid = np.ones((h, w), dtype=bool)
    if plan['affine']:
        scale = rng.uniform(*config.scale)
        dx, dy = rng.uniform(-config.translation, config.translation, 2)
        matrix = affine_matrix(h, w, scale, dx, dy)
        images = np.stack([cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) for frame in images])
        valid = cv2.warpAffine(np.ones((h, w), np.float32), matrix, (w, h), flags=cv2.INTER_LINEAR) >= 1 - 1e-6
        if protected is not None:
            protected = cv2.warpAffine(protected.astype(np.float32), matrix, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0
            protected |= ~valid
        trace['affine_params'] = dict(scale=scale, dx=dx, dy=dy)
    if plan['erase']:
        box = choose_box(protected, rng, config)
        trace['box'] = box
        if box is not None:
            x, y, bw, bh = box
            mode = str(rng.choice(['solid', 'mosaic', 'pixel_noise']))
            patch = images[:, y:y + bh, x:x + bw]
            if mode == 'solid':
                patch[:] = rng.integers(0, 256, size=(1, 1, 1, 3), dtype=np.uint8)
            elif mode == 'pixel_noise':
                patch[:] = rng.integers(0, 256, size=(1, bh, bw, 3), dtype=np.uint8)
            else:
                for frame in patch:
                    small = cv2.resize(frame, (max(1, bw // config.mosaic_block), max(1, bh // config.mosaic_block)), interpolation=cv2.INTER_AREA)
                    frame[:] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
            trace.update(erase_applied=True, erase_mode=mode)
    if plan['noise'] == 'gaussian':
        sigma = rng.uniform(*config.gaussian_noise_std)
        noise = rng.normal(0, sigma * 255, size=(1, h, w, 3))
        images = np.rint(np.clip(images.astype(np.float32) + noise, 0, 255)).astype(np.uint8)
        trace['noise_strength'] = sigma
    elif plan['noise'] == 'salt_pepper':
        amount = rng.uniform(*config.salt_pepper_amount)
        draws = rng.random((h, w))
        images[:, draws < amount / 2, :] = 0
        images[:, (draws >= amount / 2) & (draws < amount), :] = 255
        trace['noise_strength'] = amount
    if plan['blur'] == 'gaussian':
        sigma = rng.uniform(*config.gaussian_blur_sigma)
        k = max(3, int(math.ceil(3 * sigma)) * 2 + 1)
        images = np.stack([cv2.GaussianBlur(frame, (k, k), sigmaX=sigma) for frame in images])
        trace['blur_sigma'] = sigma
    elif plan['blur'] == 'motion':
        k = int(rng.choice(config.motion_blur_sizes))
        angle = rng.uniform(0, 180)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), angle, 1), (k, k))
        kernel /= kernel.sum()
        images = np.stack([cv2.filter2D(frame, -1, kernel) for frame in images])
        trace.update(motion_size=k, motion_angle=angle)
    if plan['jpeg']:
        quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
        decoded = []
        for frame in images:
            with BytesIO() as stream:
                Image.fromarray(frame).save(stream, format='JPEG', quality=quality, subsampling=2)
                stream.seek(0)
                with Image.open(stream) as image:
                    decoded.append(np.array(image.convert('RGB')))
        images = np.stack(decoded)
        trace['jpeg_quality'] = quality
    # Keep exposed borders black even after noise, blur and JPEG compression.
    images[:, ~valid, :] = 0
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))), trace


def augment_observation(observation, *, seed=None, config=None, foreground_masks=None, return_trace=False):
    """Choose the 10% original branch ONCE for all cameras/current/future frames."""
    config = config or load_config(os.environ.get('LINGBOT_Z4_CONFIG'))
    seed = int(torch.randint(0, 2**31 - 1, ()).item()) if seed is None else int(seed)
    clean = bool(np.random.default_rng(seed).random() < config.clean_probability)
    trace = {'policy': 'z4', 'clean': clean, 'cameras': {}}
    if clean:
        return (observation, trace) if return_trace else observation
    result = dict(observation)
    result['image'] = dict(observation['image'])
    result['future_image'] = dict(observation.get('future_image', {}))
    for camera, current in observation['image'].items():
        future = observation.get('future_image', {}).get(camera)
        frames = torch.stack([current] if future is None else [current, future])
        # Stable per-view streams; no dependence on dictionary order or hash randomization.
        rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(str(camera).encode())]))
        masks = None if foreground_masks is None else foreground_masks[camera]
        augmented, trace['cameras'][camera] = augment_camera_clip(frames, rng, config, masks)
        result['image'][camera] = augmented[0]
        if future is not None:
            result['future_image'][camera] = augmented[1]
    return (result, trace) if return_trace else result
