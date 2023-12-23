# PyTorch module
import torch
import torch.nn as nn
import torch.nn.functional as F


class vGdVAEb(nn.Module):
    def __init__(self):
        super(vGdVAEb, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 180, (10, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-10+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 180, (20, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-20+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 180, (30, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-30+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1, 180, (40, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-40+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1, 180, (50, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-50+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1, 180, (60, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-60+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(1, 180, (70, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-70+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(1, 180, (80, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-80+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(1, 180, (90, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-90+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(1, 180, (100, 5)),
            nn.ReLU(),
            nn.MaxPool2d((100-100+1, 1)),
            nn.Flatten(),
            nn.Linear(180, int(180/2)),
            nn.ReLU(),
            nn.Linear(int(180/2), 3)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=5, nhead=5), num_layers=6)
        self.linear1 = nn.Linear(in_features=3*10, out_features=int(3*10/2))
        self.linear2 = nn.Linear(in_features=int(3*10/2), out_features=5)
        self.linear3 = nn.Linear(in_features=int(3*10/2), out_features=5)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=5, out_features=int(180*10/2)),
            nn.ReLU(),
            nn.Linear(in_features=int(180*10/2), out_features=180*10),
            nn.Unflatten(1, (10, 180, 1)),
            nn.Conv2d(in_channels=10, out_channels=1,
                      kernel_size=(180-100+1, 5), padding=(0, 4))
        )

    def forward(self, x):
        padding_mask = torch.sum(x.squeeze(dim=1), dim=2) == 0
        x = x.permute(2, 1, 0, 3)
        x = torch.squeeze(x, dim=1)
        x = self.transformer_encoder(src=x, src_key_padding_mask=padding_mask)
        x = torch.unsqueeze(x, dim=1)
        x = x.permute(2, 1, 0, 3)
        f1 = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(x)
        f5 = self.layer5(x)
        f6 = self.layer6(x)
        f7 = self.layer7(x)
        f8 = self.layer8(x)
        f9 = self.layer9(x)
        f10 = self.layer10(x)
        h1 = F.relu(self.linear1(
            torch.cat((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10), 1)))
        mu = self.linear2(h1)
        logvar = self.linear3(h1)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        encoded = mu + eps*std
        decoded = self.decoder(encoded)
        return encoded, decoded, mu, logvar
