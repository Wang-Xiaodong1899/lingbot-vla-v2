"""PICA LingBot-VA / RoboTwin augmentation, applied before image processing.

Each call samples one camera's parameters and one noise field for all frames.
The photometric order below is an implementation choice, not a paper claim.
"""
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def augment_camera_clip(frames):
    """Transform uint8 RGB [T,C,H,W] frames with temporally shared randomness."""
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("PICA augmentation expects uint8 [T,3,H,W]")
    x = frames.float() / 255

    def uniform(low, high):
        return float(torch.empty(()).uniform_(low, high))

    mode = float(torch.rand(()))
    if mode < 0.25 or mode >= 0.5:  # photometric only / photometric + noise
        x = TF.adjust_brightness(x, uniform(0.7, 1.3))
        x = TF.adjust_contrast(x, uniform(0.7, 1.3))
        x = TF.adjust_saturation(x, uniform(0.75, 1.25))
        x = TF.adjust_hue(x, uniform(-0.04, 0.04))
        x = TF.adjust_gamma(x, uniform(0.8, 1.25))
        x = (x * (2 ** uniform(-0.25, 0.25))).clamp(0, 1)
    if mode >= 0.25:  # noise only / photometric + noise
        x = (x + torch.randn_like(x[:1]) * 0.015).clamp(0, 1)
    if float(torch.rand(())) < 0.25:
        height, width = x.shape[-2:]
        scale = uniform(0.92, 1.0)  # side length, not area
        crop_h, crop_w = round(height * scale), round(width * scale)
        top = int(torch.randint(height - crop_h + 1, ()))
        left = int(torch.randint(width - crop_w + 1, ()))
        x = TF.resized_crop(x, top, left, crop_h, crop_w, [height, width],
                            interpolation=InterpolationMode.BILINEAR, antialias=True)
        x = TF.rotate(x, uniform(-8, 8), interpolation=InterpolationMode.BILINEAR)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def augment_observation_pair(observation):
    """Copy image mappings; keep actions, state and original images untouched."""
    student = dict(observation)
    student["image"] = {}
    student["future_image"] = dict(observation.get("future_image", {}))
    for camera, current in observation["image"].items():
        future = observation.get("future_image", {}).get(camera)
        clip = torch.stack([current] if future is None else [current, future])
        augmented = augment_camera_clip(clip)
        student["image"][camera] = augmented[0]
        if future is not None:
            student["future_image"][camera] = augmented[1]
    return student
