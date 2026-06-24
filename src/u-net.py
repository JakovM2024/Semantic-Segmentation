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



class Decoder (nn.Module):
    def __init__(self, out_channels = 2, features = [64,128,256,512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature *2, feature, kernel_size=2, stride = 2))
            self.up_convs.append(DoubleConv2d(feature*2, feature))
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x, skips):
        skips = skips[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            x = torch.cat([skips[i], x], dim = 1)
            x = self.up_convs[i](x)
        return self.final(x)
    


class UNet (nn.Module):
    def __init__(self, in_channels = 3, out_channels = 2, features = [64, 128, 256, 512]):
        super().__init__()
        self.encoder = Encoder(in_channels, features)
        self.bottleneck = DoubleConv2d(features[-1], features[-1]*2)
        self.decoder = Decoder(out_channels, features)
    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        return self.decoder(x, skips)


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 512, 512)
    print(model(x).shape)                  # expect [1, 2, 512, 512]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters")     

        