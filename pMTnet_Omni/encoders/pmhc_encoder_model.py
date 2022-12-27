# PyTorch module
import torch
import torch.nn as nn
import torch.nn.functional as F


class pMHC(nn.Module):
    def __init__(self):
        super(pMHC, self).__init__()
        self.layerP1 = nn.Sequential(
            nn.Conv2d(1, 200, (2, 5)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (2, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (2, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((30-3*2+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerP2 = nn.Sequential(
            nn.Conv2d(1, 200, (4, 5)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (4, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (4, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((30-3*4+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerP3 = nn.Sequential(
            nn.Conv2d(1, 200, (6, 5)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (6, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (6, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((30-3*6+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerA1 = nn.Sequential(
            nn.Conv2d(1, 200, (10, 1280)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (10, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (10, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((380-3*10+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerA2 = nn.Sequential(
            nn.Conv2d(1, 200, (20, 1280)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (20, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (20, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((380-3*20+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerA3 = nn.Sequential(
            nn.Conv2d(1, 200, (30, 1280)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (30, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (30, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((380-3*30+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerB1 = nn.Sequential(
            nn.Conv2d(1, 200, (10, 1280)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (10, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (10, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((380-3*10+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerB2 = nn.Sequential(
            nn.Conv2d(1, 200, (20, 1280)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (20, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (20, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((380-3*20+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.layerB3 = nn.Sequential(
            nn.Conv2d(1, 200, (30, 1280)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (30, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, (30, 1)),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d((380-3*30+3*1, 1)),
            nn.Flatten(),
            nn.Linear(200, int(200/2)),
            nn.ReLU(),
            nn.Linear(int(200/2), 3)
        )
        self.fc1 = nn.Linear(3 * 9, 30)
        self.fc2 = nn.Linear(30, 3)

    def forward(self, x_p, x_a, x_b):
        f1_p = self.layerP1(x_p)
        f2_p = self.layerP2(x_p)
        f3_p = self.layerP3(x_p)

        f1_a = self.layerA1(x_a)
        f2_a = self.layerA2(x_a)
        f3_a = self.layerA3(x_a)

        f1_b = self.layerB1(x_b)
        f2_b = self.layerB2(x_b)
        f3_b = self.layerB3(x_b)
        encoded = self.fc1(
            torch.cat((f1_p, f2_p, f3_p, f1_a, f2_a, f3_a, f1_b, f2_b, f3_b), dim=1))
        encoded_act = F.relu(encoded)
        return encoded_act, self.fc2(encoded_act)
