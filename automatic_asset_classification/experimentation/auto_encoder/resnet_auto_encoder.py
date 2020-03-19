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
folder_names = ["embankment", "flood_gate", "flood_wall", "outfall",
                "reservoir", "weir"]

size = 224
batchsize = 32
tfms = get_transforms(do_flip = False)
src = (ImageImageList.from_folder(image_loc).split_by_rand_pct(seed=2).label_from_func(lambda x: x))
data = (src.transform(size=size, tfm_y=True)
        .databunch(bs=batchsize)
        .normalize(imagenet_stats)#, do_y = False))
#data = (src.transform(get_transforms(), size=size, tfm_y=True)
#        .databunch(bs=batchsize)
#        .normalize(imagenet_stats, do_y = False))

#from fast ai
class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

#===================
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained = True)
        resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        for param in resnet.parameters():
            param.requires_grad = False

        feature_extract = resnet

        ConcatPool2D = AdaptiveConcatPool2d()
        bottleneck = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        #nn.Linear(in_features=512, out_features=3, bias=True),
        nn.Linear(in_features=512, out_features=3, bias=True)
        )

        self.encoder = nn.Sequential(
        feature_extract,
        ConcatPool2D,
        bottleneck)

        self.decoder = nn.Sequential(
            nn.Linear(in_features = 3, out_features = 512, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 512, out_features = 2048, bias = True),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(inplace = True)
            #nn.ConvTranspose2d(16, 128, kernel_size=8, stride=4),
            #nn.ReLU(inplace = True),
            #nn.ConvTranspose2d(128, 256, kernel_size=8, stride=4),
            #nn.ReLU(inplace = True),
            #nn.ConvTranspose2d(256, 1024, kernel_size=4, stride=1),
            #out features to twice the size of final image
            #nn.Linear(in_features = 1024, out_features = 448, bias = True),
            #nn.ReLU(inplace = True),
            #nn.Linear(in_features = 448, out_features = 64, bias = True),
            #nn.ConvTranspose2d(64, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )

        self.upsampling = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.Upsample(scale_factor = 3.5, mode = 'bilinear'),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            #nn.ReLU(inplace = True)
            nn.Sigmoid()
        )

    def forward(self, x):
        #extracted = self.feature_extract(x)
        #encoded = self.encoder(extracted)
        #decoded = self.decoder(encoded)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, (-1, 512, 2, 2))
        #decoded = torch.reshape(decoded, (1, 512, 512))
        decoded = self.upsampling(decoded)
        return decoded
        #return encoded, decoded



autoencoder = AutoEncoder();
#learn = Learner(data, autoencoder);
summary(autoencoder, (3,224,224))
learn = Learner(data, autoencoder, loss_func=F.mse_loss);
