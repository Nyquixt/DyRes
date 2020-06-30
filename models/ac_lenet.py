import torch.nn as nn
from .acnet import *

__all__ = ['AC_LeNet']

class AC_LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AC_LeNet, self).__init__()
        self.conv = nn.Sequential(
            AC_Conv2dBN(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            AC_Conv2dBN(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out