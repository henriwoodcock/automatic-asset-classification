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

size = 28
batchsize = 32
tfms = get_transforms(do_flip = False)
src = (ImageImageList.from_folder(image_loc).split_by_rand_pct(seed=2).label_from_func(lambda x: x))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=batchsize)
        .normalize(imagenet_stats, do_y = True))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Linear(64, 12),
            nn.ReLU(inplace = True),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(inplace = True),
            nn.Linear(12, 64),
            nn.ReLU(inplace = True),
            nn.Linear(64, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder();
print(autoencoder);

learn = Learner(data, autoencoder);
