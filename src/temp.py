import torch
import torch.nn as nn

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size =3, padding =1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.network(x)
    
'''
neauron = DoubleConv2d(3,64)
x = torch.randn(1,3,512,512)
out = neauron(x)
print(out.shape)
'''

class Encoder(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        for i in features:
            self.downs.append(DoubleConv2d(in_channels, i))
            in_channels = i
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x= self.pool(x)
        return x, skips
    
enc = Encoder()
x = torch.randn(1, 3, 512, 512)
out, skips = enc(x)
print("final:", out.shape)                       # expect [1, 512, 32, 32]
for i, s in enumerate(skips):
    print(f"skip {i}:", s.shape)
# expect: [1,64,512,512] [1,128,256,256] [1,256,128,128] [1,512,64,64]


    

        