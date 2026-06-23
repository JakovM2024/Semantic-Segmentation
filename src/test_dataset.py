from src.dataset import InriaRoofDataset
ds = InriaRoofDataset("data/tiles/train", train=True)
print(len(ds), "samples")
img, msk = ds[0]
print("image:", img.shape, img.dtype)     # expect torch.Size([3, 512, 512]) float32
print("mask: ", msk.shape, msk.dtype)      # expect torch.Size([512, 512]) int64
print("mask values:", msk.unique())        # expect tensor([0, 1]) or just [0] if empty tile