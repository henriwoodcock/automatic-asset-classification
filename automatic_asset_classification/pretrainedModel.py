import model_layers
import torch
from torch import nn


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained = True)
        resnet = nn.Sequential(*(list(resnet.children())[0:8]))
        if model_weights:
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            for param in resnet.parameters():
                param.requires_grad = True


        self.encoder = nn.Sequential(resnet, model_layers.AdaptiveConcatPool2d(), nn.Flatten())

    def encode(self, x): return self.encoder(x)

    def forward(self, x):
      encoded = self.encoder(x)
      return encoded

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(1024, 24 * 8 * 8),
                                        reshape([-1,24,8,8])
                                        )

        self.up1 = Upsample(24,12,scale=7)
        self.up2 = Upsample(12,6,scale=2)
        self.up3 = Upsample(6,3)

    def forward(self,x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x

class autoencoder(nn.Module):
    def __init__(self, model_weights):
        super(autoencoder, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

    def encode(self, x): return self.encoder(x)
    def decode(self, x): return torch.clamp(self.decoder(x), min = 0, max = 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.clamp(x, min = 0, max = 1)
