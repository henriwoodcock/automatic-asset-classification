import image io
import torch
import glob
image_loc = "data/final_dataset/final/"
folder_names = ["embankment", "flood_gate", "flood_wall", "outfall",
                "reservoir", "weir"]
X = np.array([])
y = np.array([])

class AutoEncoder(nn.module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Relu(),
            nn.Linear(128, 64),
            nn.Relu(),
            nn.Linear(64, 12),
            nn.Relu(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Relu(),
            nn.Linear(12, 64),
            nn.Relu(),
            nn.Linear(64, 128),
            nn.Relu(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
