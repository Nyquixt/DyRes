import torch
import torch.nn as nn

from convs.dysep_conv import *

__all__ = ['DySep_AlexNet']

class DySep_AlexNet(nn.Module):

    def __init__(self, num_classes=1000, mode='A'):
        super(DySep_AlexNet, self).__init__()
        self.features = nn.Sequential(
            DySepConv(3, 64, kernel_size=3, stride=2, padding=1, mode=mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DySepConv(64, 192, kernel_size=3, padding=1, mode=mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DySepConv(192, 384, kernel_size=3, padding=1, mode=mode),
            nn.ReLU(inplace=True),
            DySepConv(384, 256, kernel_size=3, padding=1, mode=mode),
            nn.ReLU(inplace=True),
            DySepConv(256, 256, kernel_size=3, padding=1, mode=mode),
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

def test():
    x = torch.randn(256, 3, 32, 32)
    net = DySep_AlexNet(input_size=32)
    y = net(x)
    print(y.shape)

# test()