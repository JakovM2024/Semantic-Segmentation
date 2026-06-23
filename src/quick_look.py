import glob
import matplotlib.pyplot as plt
from PIL import Image

img_paths = sorted(glob.glob("data/tiles/train/images/*.png"))
msk_paths = sorted(glob.glob("data/tiles/train/masks/*.png"))
print(len(img_paths), "tiles found")

fig, ax = plt.subplots(3, 2, figsize=(8, 12))
for r in range(3):
    i = r * 300                                    # spread out so we see different areas
    ax[r, 0].imshow(Image.open(img_paths[i]))
    ax[r, 0].set_title("image"); ax[r, 0].axis("off")
    ax[r, 1].imshow(Image.open(msk_paths[i]), cmap="gray")
    ax[r, 1].set_title("mask");  ax[r, 1].axis("off")
plt.tight_layout()
plt.savefig("sample_check.png", dpi=100)
print("saved sample_check.png")