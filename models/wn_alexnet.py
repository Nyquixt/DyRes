import torch
import torch.nn as nn

from .weightnet import *

__all__ = ['WN_AlexNet']

class WN_AlexNet(nn.Module):

    def __init__(self, num_classes=10, input_size=32):
        super(WN_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            WeightNet(64, 192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            WeightNet(192, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            WeightNet(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            WeightNet(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
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

def test():
    x = torch.randn(256, 3, 32, 32)
    net = WN_AlexNet(input_size=32)
    y = net(x)
    print(y.shape)

# test()