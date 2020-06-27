import torch
import torch.nn as nn

from .dyconv import *

__all__ = ['Dy_AlexNet']

class Dy_AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(Dy_AlexNet, self).__init__()
        self.features = nn.Sequential(
            DyConv(3, 64, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2),
            DyConv(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            DyConv(192, 384, kernel_size=3),
            DyConv(384, 256, kernel_size=3),
            DyConv(256, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x