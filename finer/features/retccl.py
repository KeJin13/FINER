import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.preprocessing import normalize as l2_normalize
from sklearn.utils.extmath import randomized_svd
from skimage.io import imread
from tqdm.auto import tqdm

from ..models.resnet import resnet50


Image.MAX_IMAGE_PIXELS = 1_000_000_000

_IMAGENET_NORMALIZE = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def _resample_image_and_coords(
    img,
    coords_yx,
    pixel_size_um,
    target_pixel_size_um,
):
    resize_factor = float(pixel_size_um) / float(target_pixel_size_um)

    _, h, w = img.shape
    new_h = max(1, int(round(h * resize_factor)))
    new_w = max(1, int(round(w * resize_factor)))

    img = F.interpolate(
        img[None],
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )[0]

    coords_yx = np.asarray(coords_yx, dtype=np.float32) * resize_factor
    return img, coords_yx


def _load_resampled_image(
    img_path,
    coords_yx,
    pixel_size_um,
    target_pixel_size_um,
):
    img = imread(img_path)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {img.shape}")

    img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
    if img.max() > 1:
        img /= 255.0

    return _resample_image_and_coords(
        img,
        coords_yx,
        pixel_size_um,
        target_pixel_size_um,
    )


def _pad_to_divisible(img, div=256, value=1.0):
    h, w = img.shape[-2:]
    pad_h = (-h) % div
    pad_w = (-w) % div

    return F.pad(
        img[None],
        (0, pad_w, 0, pad_h),
        mode="constant",
        value=value,
    )[0]


def _to_uint8(img):
    img = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255.0).round().astype(np.uint8)


def _build_backbone(checkpoint_path, device):
    model = resnet50(
        num_classes=128,
        mlp=False,
        two_branch=False,
        normlinear=True,
    ).to(device)

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))

    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, _ = model.load_state_dict(state, strict=False)

    allowed = {"fc.weight", "fc.bias"}
    if not set(missing).issubset(allowed):
        raise RuntimeError(
            f"Missing RetCCL parameters: {set(missing) - allowed}"
        )

    feat_dim = getattr(model.fc, "in_features", 2048)
    model.fc = nn.Identity()
    model.eval()

    return model, feat_dim


@torch.no_grad()
def _extract_raw_features(
    image_uint8,
    coords_yx,
    model,
    raw_path,
    mpp=0.5,
    scales_um=(20, 35, 55),
    out_size=224,
    batch_size=128,
    min_px=32,
    max_upscale=6,
    device="cpu",
):
    device = torch.device(device)

    img = (
        torch.from_numpy(image_uint8)
        .to(device)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    sizes_px = []
    for um in scales_um:
        px = max(min_px, int(round(um / mpp)))
        if out_size / px > max_upscale:
            px = int(round(out_size / max_upscale))
        sizes_px.append((px // 2) * 2)

    half_max = max(sizes_px) // 2 + 2
    img = F.pad(
        img[None],
        (half_max, half_max, half_max, half_max),
        mode="reflect",
    )

    coords_yx = np.asarray(coords_yx, dtype=np.float32)
    ys = torch.from_numpy(coords_yx[:, 0]).to(device).float() + half_max
    xs = torch.from_numpy(coords_yx[:, 1]).to(device).float() + half_max
    n_cells = len(coords_yx)

    base_grid = F.affine_grid(
        torch.eye(2, 3, device=device)[None],
        size=(1, 3, out_size, out_size),
        align_corners=False,
    )[0]

    probe = _IMAGENET_NORMALIZE(
        torch.zeros(1, 3, out_size, out_size, device=device)
    )
    feat_dim = model(probe).shape[1]

    required_bytes = n_cells * len(sizes_px) * feat_dim * np.dtype(np.float32).itemsize
    free_bytes = shutil.disk_usage(Path(raw_path).parent).free
    if required_bytes > free_bytes:
        raise OSError(
            "Insufficient disk space for temporary RetCCL features: "
            f"need {required_bytes / 1024**3:.1f} GB, "
            f"available {free_bytes / 1024**3:.1f} GB."
        )

    raw = np.memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_cells, len(sizes_px) * feat_dim),
    )

    total_batches = len(sizes_px) * int(np.ceil(n_cells / batch_size))

    with tqdm(
        total=total_batches,
        desc="RetCCL extraction",
        unit="batch",
    ) as pbar:
        col0 = 0

        for px in sizes_px:
            col1 = col0 + feat_dim
            half = px / 2.0

            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)

                y1 = ys[start:end] - half
                x1 = xs[start:end] - half
                y2 = ys[start:end] + half
                x2 = xs[start:end] + half

                h = y2 - y1
                w = x2 - x1
                cy = (y1 + y2) * 0.5
                cx = (x1 + x2) * 0.5
                batch = end - start

                grid = base_grid[None].repeat(batch, 1, 1, 1)

                gx = grid[..., 0] * (w[:, None, None] / 2.0) + cx[:, None, None]
                gy = grid[..., 1] * (h[:, None, None] / 2.0) + cy[:, None, None]

                hp, wp = img.shape[-2:]

                grid = torch.stack(
                    [
                        (2.0 * gx + 1.0) / wp - 1.0,
                        (2.0 * gy + 1.0) / hp - 1.0,
                    ],
                    dim=-1,
                )

                crops = F.grid_sample(
                    img.expand(batch, -1, -1, -1),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )

                crops = _IMAGENET_NORMALIZE(crops)
                z = model(crops).float().cpu().numpy()

                raw[start:end, col0:col1] = z
                pbar.update(1)

            col0 = col1

    raw.flush()
    del raw

    return feat_dim, sizes_px


def _reduce_multiscale_features(
    raw_path,
    n_cells,
    n_scales,
    feat_dim,
    out_dim=256,
):
    x = np.memmap(
        raw_path,
        mode="r+",
        dtype=np.float32,
        shape=(n_cells, n_scales * feat_dim),
    )

    x = l2_normalize(
        x,
        norm="l2",
        axis=1,
        copy=False,
    )

    reduced = []

    for k in tqdm(
        range(n_scales),
        desc="RetCCL reduction",
        unit="scale",
    ):
        start = k * feat_dim
        end = (k + 1) * feat_dim

        xk = x[:, start:end].astype(np.float64, copy=False)
        mean = xk.mean(axis=0)

        _, _, vt = randomized_svd(
            xk - mean,
            n_components=out_dim,
            n_iter=5,
            random_state=42,
        )

        w = vt.T.astype(np.float32)

        reduced.append(
            (
                (xk.astype(np.float32) - mean.astype(np.float32))
                @ w
            ).astype(np.float32)
        )

    return np.concatenate(reduced, axis=1)


def retccl_features(
    cell_df,
    img_path,
    pixel_size_um,
    target_pixel_size_um,
    checkpoint_path,
    device="cpu",
    scales_um=(20, 35, 55),
    out_dim=256,
    batch_size=128,
    temp_dir=None,
):
    """Extract multi-scale cell-level RetCCL features."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best_ckpt.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    coords_yx = cell_df[
        ["pixel_y", "pixel_x"]
    ].to_numpy(dtype=np.float32)

    img, coords_yx = _load_resampled_image(
        img_path,
        coords_yx,
        pixel_size_um,
        target_pixel_size_um,
    )

    img = _pad_to_divisible(
        img,
        div=256,
        value=1.0,
    )
    image_uint8 = _to_uint8(img)
    del img

    model, _ = _build_backbone(
        checkpoint_path,
        device,
    )

    temp_dir = Path(temp_dir) if temp_dir is not None else None
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(
        prefix=".retccl_raw_",
        suffix=".dat",
        dir=temp_dir,
        delete=False,
    )
    raw_path = Path(tmp.name)
    tmp.close()

    try:
        feat_dim, _ = _extract_raw_features(
            image_uint8=image_uint8,
            coords_yx=coords_yx,
            model=model,
            raw_path=raw_path,
            mpp=target_pixel_size_um,
            scales_um=scales_um,
            batch_size=batch_size,
            device=device,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        features = _reduce_multiscale_features(
            raw_path=raw_path,
            n_cells=len(cell_df),
            n_scales=len(scales_um),
            feat_dim=feat_dim,
            out_dim=out_dim,
        )

    finally:
        if raw_path.exists():
            raw_path.unlink()

    return pd.DataFrame(
        features,
        index=cell_df.index,
        columns=[f"retccl_{i}" for i in range(features.shape[1])],
    )
