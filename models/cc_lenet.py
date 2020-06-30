import torch.nn as nn
from .condconv import *

__all__ = ['CC_LeNet']

class CC_LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CC_LeNet, self).__init__()
        self.conv = nn.Sequential(
            CondConv(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            CondConv(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
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