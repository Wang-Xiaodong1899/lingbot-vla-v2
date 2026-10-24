"""Z6b level-5 pixel whitelist, ported from the reference lingbot-vla recipe.

Source revision and hashes: experiment/robotwin/records/z6b_source.json.
Probabilities, operation order, table heuristic, shared per-view geometry and
fresh global-RNG sensor noise intentionally match the reference implementation.
The table-colour mask is a heuristic, not guaranteed semantic segmentation.
"""
import math
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

def _view_generator(params: dict, view_key) -> torch.Generator:
    g = torch.Generator()
    base = int(params["seed"])
    g.manual_seed((base + zlib.crc32(str(view_key).encode("utf-8"))) & 0x7FFFFFFF)
    return g

def _urand(g: torch.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * torch.rand((), generator=g))



def _gaussian_blur(image, sigma):
    radius = max(1, int(math.ceil(2.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=image.dtype)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k = (k / k.sum()).to(image.device)
    ch = image.shape[1]
    kh = k.view(1, 1, 1, -1).expand(ch, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(ch, 1, -1, 1)
    out = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"), kh, groups=ch)
    return F.conv2d(F.pad(out, (0, 0, radius, radius), mode="reflect"), kv, groups=ch)

def _table_mask(frame: Tensor, thr: float, dilate: int = 11) -> Tensor:
    """frame (C,H,W) in [0,1] -> (1,1,H,W) float mask, 1 = tabletop, 0 = foreground (objects, arms, background).
    The table colour is the per-channel median of the lower 40 %% of the image. A pixel is "table" when its chromaticity
    (RGB normalised by brightness) is close to the table's AND it is not much darker than the table (so the table's own
    shadows stay "table" while dark objects do not). Foreground is dilated so overlays never touch the objects."""
    _, h, w = frame.shape
    low = frame[:, int(h * 0.6):, :].reshape(3, -1)
    colour = low.median(dim=1).values.view(3, 1, 1)
    eps = 1e-3
    chroma = frame / (frame.sum(0, keepdim=True) + eps)
    chroma_t = colour / (colour.sum() + eps)
    chroma_dist = (chroma - chroma_t).abs().sum(0, keepdim=True)
    lum, lum_t = frame.mean(0, keepdim=True), colour.mean()
    table = ((chroma_dist < thr) & (lum > 0.45 * lum_t) & (lum < 1.25 * lum_t + 0.05)).to(frame.dtype).unsqueeze(0)
    fg = F.max_pool2d(1.0 - table, kernel_size=dilate, stride=1, padding=dilate // 2)
    return 1.0 - fg

def _affine_grid4(g: torch.Generator, amp, n_t: int, shape, dtype, device):
    s_amp, t_amp, r_deg = amp
    zoom = 1.0 + _urand(g, -s_amp, s_amp)
    tx, ty = _urand(g, -t_amp, t_amp) * 2.0, _urand(g, -t_amp, t_amp) * 2.0
    rot = math.radians(_urand(g, -r_deg, r_deg)) if r_deg > 0 else 0.0
    inv, c, s = 1.0 / zoom, math.cos(rot), math.sin(rot)
    theta = torch.tensor([[inv * c, -inv * s, tx], [inv * s, inv * c, ty]], dtype=dtype).unsqueeze(0).expand(n_t, 2, 3).to(device)
    return F.affine_grid(theta, list(shape), align_corners=False)

def _jpeg_roundtrip(image: Tensor, quality: int) -> Tensor:
    """(T,C,H,W) float [0,1] -> JPEG encode/decode at `quality` (OpenCV); returns the input unchanged if cv2 is missing."""
    try:
        import cv2
    except Exception:
        return image
    out = []
    for frame in image:
        arr = (frame.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return image
        dec = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out.append(torch.from_numpy(dec).to(image.device, image.dtype).permute(2, 0, 1) / 255.0)
    return torch.stack(out)

# ---------------------------------------------------------------------------
# Augmentation level 5: the "pixel-level whitelist" recipe (LINGBOT_AUG_LEVEL=5), as published by another participant:
#   10 % of the samples untouched; for the other 90 % every operation below triggers independently with probability p:
#   colour / colour-temperature jitter  p=0.80  brightness 0.25-1.10, contrast 0.5-1.5, saturation 0.4-1.6 (+ warm/cold shift)
#   small scale / translation           p=0.50  scale 0.90-1.10, shift <= 6 %, exposed border filled with black
#   random rectangle occlusion          p=0.95  avoids the foreground; solid colour / mosaic / pixel noise
#   gaussian noise / salt-and-pepper    p=0.50 / 0.15
#   gaussian blur / motion blur         p=0.25 / 0.10 (mutually exclusive)
#   JPEG compression                    p=0.30  quality 35-90
# Magnitudes the recipe does not specify (noise sigma, salt-and-pepper density, blur size, occluder size, temperature shift)
# are set here and documented in _AUG5. Foreground = pixels that do not match the table colour (see _table_mask); occluders are
# blended only over table pixels, so objects and arms are never covered. Pixel-level only: scene and task semantics untouched.
# ---------------------------------------------------------------------------
_AUG5 = dict(
    keep_p=0.10,
    color_p=0.80, bright=(0.25, 1.10), contrast=(0.5, 1.5), sat=(0.4, 1.6), temp=0.10,
    geo_p=0.50, scale=0.10, shift=0.06,
    occ_p=0.95, occ_area=(0.02, 0.10), occ_tries=12, occ_min_table=0.8, table_thr=0.12,
    gauss_p=0.50, gauss_sigma=(0.01, 0.05), sp_p=0.15, sp_density=(0.002, 0.01),
    gblur_p=0.25, gblur_sigma=(0.5, 1.5), mblur_p=0.10, mblur_len=(3, 9),
    jpeg_p=0.30, jpeg_q=(35, 90),
)


def _motion_blur(image: Tensor, length: int, direction: int) -> Tensor:
    """(T,C,H,W) -> motion blur with a line kernel of `length` px in one of 4 directions (0, 45, 90, 135 degrees)."""
    length = max(3, length | 1)
    k = torch.zeros(length, length, dtype=image.dtype, device=image.device)
    c = length // 2
    if direction == 0:
        k[c, :] = 1
    elif direction == 1:
        k[torch.arange(length), torch.arange(length)] = 1
    elif direction == 2:
        k[:, c] = 1
    else:
        k[torch.arange(length), torch.arange(length - 1, -1, -1)] = 1
    k = (k / k.sum()).view(1, 1, length, length).repeat(image.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(image, (c, c, c, c), mode="replicate"), k, groups=image.shape[1])


def whitelist_visual_augmentation(image: Tensor, params: dict, view_key, clean=None):
    """Level 5. image: (T,C,H,W) float32 in [0,1]. With `clean` given, the clean copy receives the same geometric warp only."""
    cfg = _AUG5
    if "seed" not in params:
        return image if clean is None else (image, clean)
    gs = _view_generator(params, "__sample__")
    if _urand(gs, 0.0, 1.0) < cfg["keep_p"]:
        return image if clean is None else (image, clean)
    g = _view_generator(params, view_key)
    n_t, _, height, width = image.shape
    dtype, device = image.dtype, image.device
    table = _table_mask(image[0], cfg["table_thr"]).to(dtype)        # from the original frame

    # small scale / translation, black border (same warp for all frames of this view, the clean copy and the mask)
    if _urand(g, 0.0, 1.0) < cfg["geo_p"]:
        grid = _affine_grid4(g, (cfg["scale"], cfg["shift"], 0.0), n_t, image.shape, dtype, device)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if clean is not None:
            clean = F.grid_sample(clean, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        table = F.grid_sample(table, grid[:1], mode="nearest", padding_mode="zeros", align_corners=False)

    # random rectangle occlusion on non-foreground pixels only
    if _urand(g, 0.0, 1.0) < cfg["occ_p"]:
        for _try in range(cfg["occ_tries"]):
            area = _urand(g, *cfg["occ_area"]) * height * width
            aspect = math.exp(_urand(g, math.log(0.4), math.log(2.5)))
            bh = int(min(height, max(2, round(math.sqrt(area / aspect)))))
            bw = int(min(width, max(2, round(math.sqrt(area * aspect)))))
            y0 = int(_urand(g, 0.0, float(height - bh)))
            x0 = int(_urand(g, 0.0, float(width - bw)))
            region_mask = table[0, 0, y0:y0 + bh, x0:x0 + bw]
            if region_mask.mean().item() < cfg["occ_min_table"]:
                continue
            colour = torch.tensor([_urand(g, 0.0, 1.0) for _ in range(3)], dtype=dtype).view(1, 3, 1, 1).to(device)
            kind = _urand(g, 0.0, 1.0)
            if kind < 1.0 / 3.0:
                patch = colour.expand(1, 3, bh, bw)
            elif kind < 2.0 / 3.0:
                m = torch.rand((1, 3, max(1, bh // 6), max(1, bw // 6)), generator=g).to(dtype=dtype, device=device)
                patch = F.interpolate(m, size=(bh, bw), mode="nearest")
            else:
                patch = torch.rand((1, 3, bh, bw), generator=g).to(dtype=dtype, device=device)
            rm = region_mask.view(1, 1, bh, bw)
            image[:, :, y0:y0 + bh, x0:x0 + bw] = image[:, :, y0:y0 + bh, x0:x0 + bw] * (1 - rm) + patch * rm
            break

    # colour / colour-temperature jitter
    if _urand(g, 0.0, 1.0) < cfg["color_p"]:
        image = image * _urand(g, *cfg["bright"])
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * _urand(g, *cfg["contrast"]) + mean
        gray = image.mean(dim=1, keepdim=True)
        image = gray + (image - gray) * _urand(g, *cfg["sat"])
        t = _urand(g, -cfg["temp"], cfg["temp"])                  # warm (+) / cold (-): red up, blue down
        image = image * torch.tensor([1.0 + t, 1.0, 1.0 - t], dtype=dtype, device=device).view(1, 3, 1, 1)
        image = image.clamp(0.0, 1.0)

    # sensor noise
    if _urand(g, 0.0, 1.0) < cfg["gauss_p"]:
        image = image + torch.randn_like(image) * _urand(g, *cfg["gauss_sigma"])
    if _urand(g, 0.0, 1.0) < cfg["sp_p"]:
        density = _urand(g, *cfg["sp_density"])
        r = torch.rand((n_t, 1, height, width), device=device)
        image = torch.where(r < density / 2, torch.zeros_like(image), image)
        image = torch.where((r >= density / 2) & (r < density), torch.ones_like(image), image)
    image = image.clamp(0.0, 1.0)

    # blur: gaussian or motion (mutually exclusive)
    u = _urand(g, 0.0, 1.0)
    if u < cfg["gblur_p"]:
        image = _gaussian_blur(image, _urand(g, *cfg["gblur_sigma"]))
    elif u < cfg["gblur_p"] + cfg["mblur_p"]:
        image = _motion_blur(image, int(_urand(g, *cfg["mblur_len"])), int(_urand(g, 0.0, 3.999)))

    # JPEG compression
    if _urand(g, 0.0, 1.0) < cfg["jpeg_p"]:
        image = _jpeg_roundtrip(image.clamp(0.0, 1.0), int(_urand(g, *cfg["jpeg_q"])))
    image = image.clamp(0.0, 1.0)
    return image if clean is None else (image, clean)
