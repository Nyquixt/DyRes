import torch
import torch.nn as nn

from .dyconv_bn import *

__all__ = ['DyBN_AlexNet', 'DyBN2_AlexNet']

class DyBN_AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(DyBN_AlexNet, self).__init__()
        self.features = nn.Sequential(
            DyConvBN(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DyConvBN(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DyConvBN(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DyConvBN(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DyConvBN(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024 if num_classes == 10 else 4096),
            nn.ReLU(inplace=True),
            nn.Linear(1024 if num_classes == 10 else 4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DyBN2_AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(DyBN2_AlexNet, self).__init__()
        self.features = nn.Sequential(
            DyConvBN2(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DyConvBN2(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DyConvBN2(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DyConvBN2(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DyConvBN2(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024 if num_classes == 10 else 4096),
            nn.ReLU(inplace=True),
            nn.Linear(1024 if num_classes == 10 else 4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

