from typing import Dict, List, Optional, Sequence
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEFAULT_CLASS_COLORS = {
    # RGB
    "tumor":  (220, 50,  47),
    "normal": (38,  139, 210),
    "other":  (128, 0, 128),
}

def draw_overlay_on_base(base: Image.Image,
                         rects_px: np.ndarray,
                         winners: np.ndarray,
                         conf: np.ndarray,
                         class_order: List[str],
                         class_colors: Optional[Dict[str, tuple]] = None,
                         alpha_base: float = 0.35,
                         alpha_gamma: float = 0.0,
                         slide_dims: Optional[tuple] = None,
                         scale_rects: Optional[np.ndarray] = None) -> Image.Image:
    class_colors = class_colors or DEFAULT_CLASS_COLORS
    base_w, base_h = base.size
    if slide_dims and all(slide_dims):
        w_full, h_full = slide_dims
    else:
        src = scale_rects if scale_rects is not None else rects_px
        w_full = int(src[:, 2].max()) if src.size else base_w
        h_full = int(src[:, 3].max()) if src.size else base_h

    sx = base_w / max(1.0, float(w_full))
    sy = base_h / max(1.0, float(h_full))

    overlay = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(overlay, "RGBA")
    for (x0, y0, x1, y1), k, p in zip(rects_px, winners, conf):
        X0 = int(round(x0 * sx)); Y0 = int(round(y0 * sy))
        X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
        cls = class_order[int(k)]
        r, g, b = class_colors.get(cls, (0,0,0))
        a = int(255 * (alpha_base * (float(p) ** alpha_gamma)))
        dr.rectangle([X0, Y0, X1, Y1], fill=(r, g, b, a))
    return Image.alpha_composite(base.convert("RGBA"), overlay)

def legend_for_classes(class_order: List[str],
                       class_colors: Optional[Dict[str, tuple]] = None):
    class_colors = class_colors or DEFAULT_CLASS_COLORS
    return [Patch(facecolor=np.array(class_colors[c])/255.0, edgecolor='none', label=c)
            for c in class_order]

def show_grid(items: List[tuple],
              class_order: List[str],
              title: str,
              columns: int = 5,
              tile_inches: float = 3.5,
              dpi: int = 120):
    if len(items) == 0:
        print(f"[{title}] No tiles.")
        return None

    n = len(items)
    n_cols = max(1, min(columns, n))
    n_rows = int(np.ceil(n / n_cols))

    figsize = (tile_inches * n_cols, tile_inches * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, data in zip(axes, items):
        slide, idx, pil_img, p = data
        ax.imshow(pil_img)
        pred_idx = int(np.argmax(p))
        pred = class_order[pred_idx]
        score_text = " | ".join([f"{cls}: {p_i:.2f}" for cls, p_i in zip(class_order, p)])
        ax.set_title(f"{slide} • tile {idx}\nPred: {pred}", fontsize=11, pad=2)
        ax.text(0.5, -0.08, score_text, transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(len(items), len(axes)):
        axes[j].axis('off')

    sup = fig.suptitle(title, fontsize=16)
    sup.set_in_layout(True)
    plt.subplots_adjust(top=0.90, wspace=0.06, hspace=0.15)
    return fig
