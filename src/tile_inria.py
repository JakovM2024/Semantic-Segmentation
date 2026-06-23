# src/tile_inria.py
import os, glob
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow the big 5000x5000 tiles

SRC = "data/train"  # adjust to your extracted path
OUT  = "data/tiles"
TILE, STRIDE = 512, 512                   # use stride 256 later for overlap/more data
VAL_IDX = {"1", "2", "3", "4", "5"}       # official: first 5 per city = val

def tiles(arr):
    h, w = arr.shape[:2]
    for y in range(0, h - TILE + 1, STRIDE):
        for x in range(0, w - TILE + 1, STRIDE):
            yield y, x, arr[y:y+TILE, x:x+TILE]

def main():
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(OUT, split, sub), exist_ok=True)

    for ip in sorted(glob.glob(os.path.join(SRC, "images", "*.tif"))):
        name = os.path.splitext(os.path.basename(ip))[0]      # e.g. "austin1"
        idx  = "".join(c for c in name if c.isdigit())
        split = "val" if idx in VAL_IDX else "train"
        mp = os.path.join(SRC, "gt", os.path.basename(ip))

        img  = np.array(Image.open(ip).convert("RGB"))
        mask = (np.array(Image.open(mp).convert("L")) > 127).astype(np.uint8) * 255

        for (y, x, it), (_, _, mt) in zip(tiles(img), tiles(mask)):
            tag = f"{name}_{y}_{x}"
            Image.fromarray(it).save(os.path.join(OUT, split, "images", tag + ".png"))
            Image.fromarray(mt).save(os.path.join(OUT, split, "masks",  tag + ".png"))
        print("tiled", name, "->", split)

if __name__ == "__main__":
    main()