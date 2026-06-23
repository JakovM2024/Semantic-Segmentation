# src/dataset.py
import glob, os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_tf(train):
    aug = [A.HorizontalFlip(0.5), A.VerticalFlip(0.5),
           A.RandomRotate90(0.5), A.RandomBrightnessContrast(p=0.2)] if train else []
    return A.Compose(aug + [A.Normalize(), ToTensorV2()])

class InriaRoofDataset(Dataset):
    def __init__(self, root, train=True):
        self.imgs = sorted(glob.glob(os.path.join(root, "images", "*.png")))
        self.msks = sorted(glob.glob(os.path.join(root, "masks",  "*.png")))
        assert self.imgs and len(self.imgs) == len(self.msks), f"no tiles in {root}"
        self.tf = build_tf(train)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = np.array(Image.open(self.imgs[i]).convert("RGB"))
        msk = (np.array(Image.open(self.msks[i]).convert("L")) > 127).astype("uint8")
        out = self.tf(image=img, mask=msk)
        return out["image"], out["mask"].long()