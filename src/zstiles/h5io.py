from pathlib import Path
from typing import Optional, Tuple, Sequence
import h5py
import numpy as np
from PIL import Image

def fix_orientation(pil_img: Image.Image) -> Image.Image:
    return pil_img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

def pick_tile_dataset(f: h5py.File, preferred_key: Optional[str]) -> h5py.Dataset:
    if preferred_key is not None:
        assert preferred_key in f, f"Preferred dataset {preferred_key} not found."
        return f[preferred_key]
    cand = [k for k in f.keys() if k.startswith("tile_")]
    assert cand, "No tile_* dataset found in HDF5."
    cand = sorted(cand, key=lambda k: int(k.split("_")[1]))
    return f[cand[0]]

def read_grid_from_h5(h5_path: str, preferred_key: Optional[str]):
    with h5py.File(h5_path, "r") as f:
        if "tilerow" in f and "tilecol" in f:
            tilerow = f["tilerow"][()].astype(int)
            tilecol = f["tilecol"][()].astype(int)
        else:
            for rk, ck in [("tile_row","tile_col"), ("row","col"), ("rows","cols")]:
                if rk in f and ck in f:
                    tilerow = f[rk][()].astype(int)
                    tilecol = f[ck][()].astype(int)
                    break
            else:
                raise KeyError("No tilerow/tilecol (or equivalents) found.")

        ds = pick_tile_dataset(f, preferred_key)
        size   = int(ds.attrs["size"])
        stride = int(ds.attrs["stride"])
        out_sz = int(ds.attrs.get("output_size", size))

        H = int(tilerow.max()) + 1
        W = int(tilecol.max()) + 1

        x0 = tilecol * stride
        y0 = tilerow * stride
        x1 = x0 + size
        y1 = y0 + size
        rects = np.stack([x0, y0, x1, y1], axis=1).astype(int)

        if "background" in f:
            bg_mask = f["background"][()].astype(bool)
            if bg_mask.shape[0] != rects.shape[0]:
                bg_mask = np.zeros(rects.shape[0], dtype=bool)
        else:
            bg_mask = np.zeros(rects.shape[0], dtype=bool)

        return rects, tilerow, tilecol, stride, size, out_sz, (H, W), bg_mask

def stitch_mosaic_scaled(h5_path: str,
                         rects_px: np.ndarray,
                         preferred_key: Optional[str],
                         target_max_dim: int = 2000,
                         bg_color=(245,245,245)) -> Image.Image:
    if rects_px.size == 0:
        from PIL import Image
        return Image.new("RGB", (1024, 1024), bg_color)

    w_full = int(rects_px[:, 2].max())
    h_full = int(rects_px[:, 3].max())
    cur_max = max(w_full, h_full)
    scale = 1.0 if cur_max <= target_max_dim else target_max_dim / float(cur_max)
    W = max(1, int(round(w_full * scale)))
    H = max(1, int(round(h_full * scale)))

    canvas = Image.new("RGB", (W, H), bg_color)
    sx = scale; sy = scale

    with h5py.File(h5_path, "r") as f:
        ds = pick_tile_dataset(f, preferred_key)
        for i in range(len(ds)):
            arr = ds[i]
            if arr.ndim == 3 and arr.shape[0] == 3:
                import numpy as np
                arr = np.moveaxis(arr, 0, -1)
            pil = Image.fromarray(arr, "RGB")
            x0, y0, x1, y1 = rects_px[i]
            X0 = int(round(x0 * sx)); Y0 = int(round(y0 * sy))
            X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
            tw = max(1, X1 - X0); th = max(1, Y1 - Y0)
            if pil.size != (tw, th):
                pil = pil.resize((tw, th), Image.BILINEAR)
            canvas.paste(pil, (X0, Y0))
    return canvas

def write_binary_dataset(h5_path: str, key: str, mask_1d: np.ndarray):
    assert mask_1d.ndim == 1, "mask must be 1-D"
    with h5py.File(h5_path, "r+") as f:
        if key in f:
            del f[key]
        f.create_dataset(
            key, data=mask_1d.astype("uint8"),
            compression="gzip", compression_opts=4, shuffle=True
        )
