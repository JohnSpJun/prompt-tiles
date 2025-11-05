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
