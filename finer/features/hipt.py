import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from sklearn.neighbors import KDTree
from skimage.transform import rescale
from tqdm.auto import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from ..models._hipt import HIPT4K, eval_transform


def _rescale_image(img, scale):
    scale = [scale, scale, 1] if img.ndim == 3 else [scale, scale]
    return rescale(img, scale, preserve_range=True)


def _pad_image(img, pad=256):
    shape = np.array(img.shape[:2])
    extra = (pad - shape % pad) % pad
    return np.pad(img, ((0, extra[0]), (0, extra[1]), (0, 0)), mode="constant", constant_values=255)


def _patchify(img, patch_size=4096):
    shape = np.array(img.shape[:2])
    padded = (shape + patch_size - 1) // patch_size * patch_size
    x = np.pad(
        img,
        ((0, padded[0] - shape[0]), (0, padded[1] - shape[1]), (0, 0)),
        mode="edge",
    )

    tile_shape = np.array(x.shape[:2]) // patch_size
    tiles = []
    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            r0, c0 = i * patch_size, j * patch_size
            tiles.append(x[r0:r0 + patch_size, c0:c0 + patch_size])

    return tiles, {"original": shape, "tiles": tile_shape}


def _embed_256(model, tile):
    x = tile.astype(np.float32) / 255.0
    x = eval_transform()(x)
    x_cls, x_sub = model.forward_all256(x[None])

    x_cls = x_cls.cpu().detach().numpy()[0].transpose(1, 2, 0)
    x_sub = x_sub.cpu().detach().numpy()[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def _embed_4k(model, x):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        _, x = model.forward_all4k(x[None])
    return x.cpu().detach().numpy()[0].transpose(1, 2, 0)


def _extract_embeddings(img, checkpoint_dir, device="cuda"):
    tiles, shapes = _patchify(img, patch_size=4096)

    model = HIPT4K(
        model256_path=str(Path(checkpoint_dir) / "vit256_small_dino.pth"),
        model4k_path=str(Path(checkpoint_dir) / "vit4k_xs_dino.pth"),
        device=device,
    )
    model.eval()

    emb_mid = []
    emb_sub = []

    for tile in tiles:
        x_mid, x_sub = _embed_256(model, tile)
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)

    del tiles
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    emb_mid = rearrange(
        emb_mid,
        "(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k",
        h1=shapes["tiles"][0],
        w1=shapes["tiles"][1],
    )

    emb_cls = _embed_4k(model, emb_mid)

    del emb_mid, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shape = np.array(shapes["original"]) // np.array((16, 16))

    sub = []
    for i in range(emb_sub[0].shape[-1]):
        x = rearrange(
            np.array([e[..., i] for e in emb_sub]),
            "(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)",
            h1=shapes["tiles"][0],
            w1=shapes["tiles"][1],
        )
        sub.append(x[:shape[0], :shape[1]])

    cls = []
    for i in range(emb_cls[0].shape[-1]):
        x = repeat(
            np.array([e[..., i] for e in emb_cls]),
            "h12 w12 -> (h12 h3) (w12 w3)",
            h3=16,
            w3=16,
        )
        cls.append(x[:shape[0], :shape[1]])

    return cls, sub


def _extract_shifted_embeddings(
    img,
    checkpoint_dir,
    margin=256,
    stride=64,
    device="cuda",
):
    factor = 16
    shape = np.array(img.shape[:2]) // factor

    cls = [np.zeros(shape, dtype=np.float32) for _ in range(192)]
    sub = [np.zeros(shape, dtype=np.float32) for _ in range(384)]

    shifts = list(range(0, margin, stride))
    shift_pairs = [(r, c) for r in shifts for c in shifts]

    for start0, start1 in tqdm(
        shift_pairs,
        desc="HIPT embedding",
        unit="shift",
    ):
        stop0, stop1 = -margin + start0, -margin + start1

        x_cls, x_sub = _extract_embeddings(
            img[start0:stop0, start1:stop1],
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        r0, c0 = start0 // factor, start1 // factor
        r1, c1 = stop0 // factor, stop1 // factor

        for i in range(192):
            cls[i][r0:r1, c0:c1] += x_cls[i]
        for i in range(384):
            sub[i][r0:r1, c0:c1] += x_sub[i]

    n_reps = len(shift_pairs)
    margin_grid = margin // factor

    for channels in (cls, sub):
        for x in channels:
            x /= n_reps
            x[-margin_grid:] = 0.0
            x[:, -margin_grid:] = 0.0

    return cls, sub


def _smooth_channels(channels, size, desc):
    kernel = np.ones((size, size), np.float32) / size**2
    out = []

    for x in tqdm(channels, desc=desc, unit="channel", leave=False):
        y = cv2.filter2D(
            x[..., None],
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT,
        )
        if y.ndim == 2:
            y = y[..., None]
        out.append(y[..., 0])

    return out


def _flatten_channels(channels):
    x = np.asarray(channels)
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2]).T


def _match_cells(channels, cell_df, image_shape, resize_factor):
    features = _flatten_channels(channels)

    rows = list(range(0, image_shape[0], 16))
    cols = list(range(0, image_shape[1], 16))

    patch_xy = np.column_stack(
        [np.repeat(rows, len(cols)), cols * len(rows)]
    ).astype(float)

    # Map patch centers back to raw-image pixel coordinates.
    patch_xy = (patch_xy + 8.0) / resize_factor

    # Keep the row/column ordering used in the original FINER implementation.
    cell_xy = cell_df[["pixel_y", "pixel_x"]].to_numpy()

    _, idx = KDTree(patch_xy).query(cell_xy, k=1)
    cell_features = features[idx[:, 0]]

    return pd.DataFrame(
        cell_features,
        index=cell_df.index,
        columns=[f"feat_{i}" for i in range(cell_features.shape[1])],
    )


def hipt_features(
    cell_df,
    img_path,
    pixel_size_um,
    checkpoint_path,
    target_pixel_size_um=0.5,
    device="cuda",
):
    """
    Extract cell-level HIPT features.

    pixel_size_um:
        Physical resolution of the input image in µm/pixel.

    target_pixel_size_um:
        Physical resolution used for HIPT extraction.
        The HIPT/iStar pipeline uses 0.5 µm/pixel.
    """
    np.random.seed(0)
    torch.manual_seed(0)

    resize_factor = pixel_size_um / target_pixel_size_um

    with Image.open(str(img_path)) as im:
        img = np.array(im.convert("RGB"))
        
    if img is None:
        raise FileNotFoundError(img_path)

    img = _rescale_image(img.astype(np.float32), resize_factor).astype(np.uint8)
    img = _pad_image(img, pad=256)

    cls, sub = _extract_shifted_embeddings(
        img,
        checkpoint_dir=checkpoint_path,
        margin=256,
        stride=64,
        device=device,
    )

    cls = _smooth_channels(cls, size=16, desc="HIPT CLS smoothing")
    sub = _smooth_channels(sub, size=4, desc="HIPT SUB smoothing")

    cls_df = _match_cells(cls, cell_df, img.shape, resize_factor)
    sub_df = _match_cells(sub, cell_df, img.shape, resize_factor)

    return {"cls": cls_df, "sub": sub_df}
