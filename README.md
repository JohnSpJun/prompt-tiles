# prompt-tiles
## Zero-Shot Tumor Tiles (CONCH -- CLIP-style zero-shot)

Classify histopathology tiles **without training** using [CONCH](https://github.com/mahmoodlab/CONCH) text-image embeddings and user-defined prompts.
Supports:
- Custom **classes + prompts** via YAML
- Per-slide **heatmap overlay** (mosaic + colored rectangles)
- **Tile grids** grouped by predicted class
- Writes a `tumor_region` dataset (optional) back into the `.hdf5` (1 = tumor, 0 = other)

> Works with HDF5 files that contain:  
> `tile_*` dataset(s) with RGB tiles, `tilerow`, `tilecol`, and optional `background` (0=foreground, 1=background).  
> If you have different row/col keys (e.g., `tile_row`, `tile_col`) we handle common variants.

---

## Quickstart

### 1) Install

```bash
git clone https://github.com/johnspjun/prompt-tiles.git
cd prompt-tiles

# Option A: editable install
pip install -e .

# Option B: requirements only
pip install -r requirements.txt
pip install "git+https://github.com/Mahmoodlab/CONCH.git"
```
### 2) Run a slide
```
python scripts/zs_classify.py \
  --h5 /path/to/slide.hdf5 \
  --prompts prompts/pancreas.yaml \
  --out heatmaps \
  --tile-key tile_224 \
  --write-tumor-dataset \
  --show-heatmap \
  --show-grids
```
* --prompts points to a YAML defining classes + text prompts (see prompts/).
* --tile-key is optional. If omitted, the first dataset starting with tile_ is used.

### 3) Batch over many slides
```
python scripts/zs_classify.py \
  --h5 /data/tiles/*.hdf5 \
  --prompts prompts/pancreas.yaml \
  --out heatmaps \
  --show-heatmap
```










