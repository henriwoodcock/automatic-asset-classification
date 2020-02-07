from torchsummary import summary
import torch
from torch import nn
import imageio
import torch
import glob
from fastai.vision import *
import os
from torch import nn
import torch.nn.functional as F
loc = os.getcwd() + "/"
image_loc = "data/final_dataset/final/"

size = 224
batchsize = 32
tfms = get_transforms(do_flip = False)
src = (ImageImageList.from_folder(image_loc).split_by_rand_pct(seed=2).label_from_func(lambda x: x))
data = (src.transform(size=size, tfm_y=True)
        .databunch(bs=batchsize)
        .normalize(imagenet_stats))#, do_y = False))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained = True)
        resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        for param in resnet.parameters():
            param.requires_grad = False

        feature_extract = resnet



        bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Flatten(),
            nn.Linear(in_features = 8192, out_features = 2048, bias = True),
        )

        self.encoder = nn.Sequential(
        feature_extract,
        bottleneck
        )

        self.decoder = nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Upsample(scale_factor = 1.75, mode = 'bilinear'),
        nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.ReLU(inplace = True),
        nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        midpoint = torch.reshape(encoded, (-1, 2048, 1, 1))
        decoded = self.decoder(midpoint)
        return decoded

autoencoder = AutoEncoder();
#learn = Learner(data, autoencoder);
summary(autoencoder, (3,224,224))
learn = Learner(data, autoencoder, loss_func=F.mse_loss);
