import model_layers
import torch
from torch import nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.convblock1 = model_layers.convblock(3,12)
        self.downsamp1 = model_layers.downsamp(12, 16)

        self.convblock2 = model_layers.convblock(12,24)
        self.downsamp2 = model_layers.downsamp(24, 8)

        self.bottleneck = nn.Sequential(nn.Flatten(),
                                        nn.Linear(24 * 8 * 8, 1000)
                                        )
    def forward(self, x):
        x = self.convblock1(x)
        x = self.downsamp1(x)

        x = self.convblock2(x)
        x = self.downsamp2(x)

        x = self.bottleneck(x)

        return x

class decoder(nn.Module):
    def __init__(self, scales):
        super(decoder, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(1000, 24 * 8 * 8),
                                        model_layers.reshape([-1,24,8,8])
                                        )

        self.up1 = model_layers.Upsample(24,12, scales[0])
        self.up2 = model_layers.Upsample(12,3, scales[1])

    def forward(self,x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        return x

class autoencoder(nn.Module):
    def __init__(self, scales):
        super(autoencoder, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder(scales)

    def encode(self, x): return self.encoder(x)
    def decode(self, x): return torch.clamp(self.decoder(x), min = 0, max = 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.clamp(x, min = 0, max = 1)
