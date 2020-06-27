import torch
import torch.nn as nn

from .condconv import *

__all__ = ['CC_AlexNet']

class CC_AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(CC_AlexNet, self).__init__()
        self.features = nn.Sequential(
            CondConv(3, 64, num_experts=3, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            CondConv(64, 192, num_experts=3, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            CondConv(192, 384, num_experts=3, kernel_size=3, padding=1),
            CondConv(384, 256, num_experts=3, kernel_size=3, padding=1),
            CondConv(256, 256, num_experts=3, kernel_size=3, padding=1),
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