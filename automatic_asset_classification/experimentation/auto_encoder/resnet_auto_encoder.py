#from fast ai
class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        resnet = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained = True)
        resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        for param in resnet.parameters():
            param.requires_grad = False

        feature_extract = resnet

        ConcatPool2D = AdaptiveConcatPool2d()
        bottleneck = nn.Sequential(nn.Flatten(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5, inplace=False),
        #nn.Linear(in_features=512, out_features=3, bias=True),
        nn.Linear(in_features=512, out_features=224, bias=True)
        )

        self.encoder = nn.Sequential(feature_extract, ConcatPool2D, bottleneck)

        self.decoder = nn.Sequential(
            nn.Linear(in_features = 3, out_features = 16, bias = True),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(16, 128, kernel_size=8, stride=4),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 256, kernel_size=8, stride=4),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 1024, kernel_size=4, stride=1),
            #out features to twice the size of final image
            nn.Linear(in_features = 1024, out_features = 448, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 448, out_features = 64, bias = True),
            nn.ConvTranspose2d(64, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )

        #self.model = nn.Sequential(feature_extract, encoder)
    def forward(self, x):
        #extracted = self.feature_extract(x)
        #encoded = self.encoder(extracted)
#        decoded = self.decoder(encoded)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

autoencoder = AutoEncoder();
#learn = Learner(data, autoencoder);
