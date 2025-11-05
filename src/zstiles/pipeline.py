from typing import Optional, Sequence, List, Dict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .h5io import pick_tile_dataset, read_grid_from_h5, stitch_mosaic_scaled, write_binary_dataset, fix_orientation
from .visualize import draw_overlay_on_base, legend_for_classes, show_grid as _grid

@torch.no_grad()
def classify_slide(h5_path: str,
                   model,
                   preprocess,
                   text_feats: torch.Tensor,
                   class_order: List[str],
                   preferred_key: Optional[str] = None,
                   batch_size: int = 512):
    rects_px, tilerow, tilecol, stride, size, out_sz, grid_hw, bg_mask = read_grid_from_h5(h5_path, preferred_key)
    fg_idx = np.where(~bg_mask)[0]
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(next(model.parameters()).dtype==torch.float16)):
            with torch.no_grad():
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                    Z = []
                    import h5py
                    with h5py.File(h5_path, "r") as f:
                        ds = pick_tile_dataset(f, preferred_key)
                        idxs = np.arange(len(ds), dtype=int)
                        if fg_idx.size > 0:
                            idxs = fg_idx
                        for s in range(0, len(idxs), batch_size):
                            batch_ids = idxs[s:s+batch_size]
                            batch = []
                            for i in batch_ids:
                                arr = ds[i]
                                if arr.ndim == 3 and arr.shape[0] == 3:
                                    import numpy as np
                                    arr = np.moveaxis(arr, 0, -1)
                                pil = Image.fromarray(arr, "RGB")
                                pil = fix_orientation(pil)
                                batch.append(preprocess(pil).unsqueeze(0))
                            imgs = torch.cat(batch, 0).to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
                            z = model.encode_image(imgs)
                            z = z / z.norm(dim=-1, keepdim=True)
                            Z.append(z)
                    Z = torch.cat(Z, 0)

                    logit_scale = model.logit_scale.exp() if hasattr(model.logit_scale, "exp") else model.logit_scale
                    logits = (Z @ text_feats.T) * logit_scale
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                    winners = probs.argmax(1)
                    conf = probs.max(1)

    return probs, winners, conf, fg_idx, rects_px, tilerow, tilecol, bg_mask

def heatmap_for_slide(h5_path: str,
                      rects_px: np.ndarray,
                      winners: np.ndarray,
                      conf: np.ndarray,
                      class_order: List[str],
                      target_max_dim: int = 2000,
                      alpha_base: float = 0.35,
                      alpha_gamma: float = 0.0,
                      preferred_key: Optional[str] = None,
                      class_colors: Optional[Dict[str, tuple]] = None):
    base = stitch_mosaic_scaled(h5_path, rects_px, preferred_key, target_max_dim)
    slide_dims = (int(rects_px[:,2].max()), int(rects_px[:,3].max()))
    blended = draw_overlay_on_base(
        base, rects_px, winners, conf, class_order,
        class_colors=class_colors, alpha_base=alpha_base, alpha_gamma=alpha_gamma,
        slide_dims=slide_dims, scale_rects=rects_px
    )
    return blended

def grid_for_classes(h5_path: str,
                     class_order: List[str],
                     probs: np.ndarray,
                     winners: np.ndarray,
                     preferred_key: Optional[str] = None,
                     per_class: int = 25):
    # Collect a random subset per class and load original tile images
    import h5py, numpy as np
    figs = {}
    with h5py.File(h5_path, "r") as f:
        ds = pick_tile_dataset(f, preferred_key)
        by_class = {cls: [] for cls in class_order}
        all_idx = np.arange(len(winners))
        for ci, cls in enumerate(class_order):
            idxs = all_idx[winners == ci]
            if idxs.size == 0:
                continue
            idxs = idxs[:min(per_class, idxs.size)]
            for i in idxs:
                arr = ds[i]
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = np.moveaxis(arr, 0, -1)
                from PIL import Image
                pil = Image.fromarray(arr, "RGB")
                by_class[cls].append((Path(h5_path).stem, int(i), pil, probs[i]))
    for cls in class_order:
        fig = _grid(by_class[cls], class_order, title=f"Predicted: {cls}")
        figs[cls] = fig
    return figs

def maybe_write_tumor_mask(h5_path: str,
                           class_order: List[str],
                           winners: np.ndarray,
                           fg_idx: np.ndarray,
                           bg_mask: np.ndarray,
                           tumor_class_name: str = "tumor"):
    if tumor_class_name not in class_order:
        return False
    tumor_idx = class_order.index(tumor_class_name)
    full = np.zeros(len(bg_mask), dtype=np.uint8)
    if fg_idx.size > 0:
        full[fg_idx] = (winners == tumor_idx).astype(np.uint8)
    write_binary_dataset(h5_path, "tumor_region", full)
    return True
