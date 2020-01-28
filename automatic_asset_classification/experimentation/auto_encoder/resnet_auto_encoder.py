from torchsummary import summary
import torch
from torch import nn
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
            nn.Linear(in_features = 512, out_features = 1024, bias = True),
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
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Upsample(scale_factor = 3.5, mode = 'bilinear'),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            #nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            #nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.ReLU(inplace = True),
        )

    def forward(self, x):
        #extracted = self.feature_extract(x)
        #encoded = self.encoder(extracted)
        #decoded = self.decoder(encoded)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, (1, 512, 2, 2))
        #decoded = torch.reshape(decoded, (1, 512, 512))
        decoded = self.upsampling(decoded)
        return encoded, decoded



autoencoder = AutoEncoder();
#learn = Learner(data, autoencoder);
summary(autoencoder, (3,224,224))

#==============================
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
            nn.Linear(in_features = 512, out_features = 1024, bias = True),
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
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Upsample(scale_factor = 3.5, mode = 'bilinear'),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(512, 512, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=(3,3), padding = (1,1), bias = False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 128, kernel_size=(1,1), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 128, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(32, 32, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(16, 16, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(8, 8, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(8, 4, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(4, 4, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(4, 2, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(2, 2, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(2, 1, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),

        )

    def forward(self, x):
        #extracted = self.feature_extract(x)
        #encoded = self.encoder(extracted)
        #decoded = self.decoder(encoded)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, (1, 512, 2, 2))
        #decoded = torch.reshape(decoded, (1, 512, 512))
        decoded = self.upsampling(decoded)
        return encoded, decoded



autoencoder = AutoEncoder();
#learn = Learner(data, autoencoder);

'''
            nn.ConvTranspose2d(512, 512, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 256, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 128, kernel_size=(1,1), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 128, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(32, 32, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(16, 16, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(8, 8, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(8, 4, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(4, 4, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(4, 2, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(2, 2, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(2, 1, kernel_size=(3,3), stride=(2,2), padding = (1,1), bias = False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3), stride=(1,1), padding = (1,1), bias = False),
'''
