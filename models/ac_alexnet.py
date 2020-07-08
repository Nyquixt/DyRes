import torch
import torch.nn as nn

from .acnet import *

__all__ = ['AC_AlexNet']

class AC_AlexNet(nn.Module):

    def __init__(self, num_classes=10, input_size=32):
        super(AC_AlexNet, self).__init__()
        self.features = nn.Sequential(
            AC_Conv2dBN(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            AC_Conv2dBN(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            AC_Conv2dBN(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            AC_Conv2dBN(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            AC_Conv2dBN(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2 if input_size==32 else 256 * 4 * 4, 4096),
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