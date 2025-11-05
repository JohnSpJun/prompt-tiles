#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path

import matplotlib.pyplot as plt

from zstiles.model import load_conch, build_text_features_from_yaml
from zstiles.pipeline import classify_slide, heatmap_for_slide, grid_for_classes, maybe_write_tumor_mask

def parse_args():
    ap = argparse.ArgumentParser(description="Zero-shot tumor tile classification with CONCH")
    ap.add_argument("--h5", nargs="+", required=True, help="HDF5 path(s) or glob(s)")
    ap.add_argument("--prompts", required=True, help="YAML with classes+prompts")
    ap.add_argument("--tile-key", default=None, help="Preferred tile dataset (e.g., tile_224)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="heatmaps", help="Output dir for heatmaps")
    ap.add_argument("--target-max-dim", type=int, default=2000)
    ap.add_argument("--alpha-base", type=float, default=0.35)
    ap.add_argument("--alpha-gamma", type=float, default=0.0)
    ap.add_argument("--show-heatmap", action="store_true")
    ap.add_argument("--show-grids", action="store_true")
    ap.add_argument("--write-tumor-dataset", action="store_true",
                    help="Write 'tumor_region' dataset if class 'tumor' exists")
    return ap.parse_args()

def expand_paths(patterns):
    out = []
    for p in patterns:
        if any(ch in p for ch in "*?[]"):
            out.extend(glob.glob(p))
        else:
            out.append(p)
    return sorted(set(out))

def main():
    args = parse_args()
    paths = expand_paths(args.h5)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    device, model, preprocess, tokenizer, model_dtype = load_conch()
    class_order, text_feats = build_text_features_from_yaml(args.prompts, tokenizer, model, device)

    for h5 in paths:
        slide = Path(h5).stem
        print(f"[{slide}] classifying…")
        probs, winners, conf, fg_idx, rects_px, tilerow, tilecol, bg_mask = classify_slide(
            h5, model, preprocess, text_feats, class_order,
            preferred_key=args.tile_key, batch_size=args.batch_size
        )

        # Heatmap
        heat = heatmap_for_slide(
            h5, rects_px, winners, conf, class_order,
            target_max_dim=args.target_max_dim,
            alpha_base=args.alpha_base, alpha_gamma=args.alpha_gamma,
            preferred_key=args.tile_key
        )
        out_path = Path(args.out) / f"{slide}_overlay.png"
        heat.save(out_path)
        print(f"[{slide}] saved heatmap: {out_path}")

        if args.show_heatmap:
            plt.figure(figsize=(10,10), dpi=110)
            plt.imshow(heat); plt.xticks([]); plt.yticks([])
            from zstiles.visualize import legend_for_classes
            lg = legend_for_classes(class_order)
            plt.legend(handles=lg, loc='lower right', frameon=True, title="Predicted class")
            plt.title(f"{slide} • overlay (alpha ∝ confidence)")
            plt.tight_layout()
            plt.show()

        if args.show_grids:
            figs = grid_for_classes(h5, class_order, probs, winners, preferred_key=args.tile_key)
            for cls, fig in figs.items():
                if fig is None: continue
                fig.show()

        if args.write_tumor_dataset:
            if maybe_write_tumor_mask(h5, class_order, winners, fg_idx, bg_mask, tumor_class_name="tumor"):
                print(f"[{slide}] wrote HDF5 dataset: tumor_region")

if __name__ == "__main__":
    main()
