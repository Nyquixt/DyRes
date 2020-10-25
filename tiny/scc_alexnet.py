import torch
import torch.nn as nn

from convs.scc_conv import *

__all__ = ['SCC_AlexNet']

class SCC_AlexNet(nn.Module):

    def __init__(self, num_classes=200, num_experts=3):
        super().__init__()
        self.features = nn.Sequential(
            SCConv(3, 64, kernel_size=3, stride=2, padding=1, num_experts=num_experts, spatial=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            SCConv(64, 192, kernel_size=3, padding=1, num_experts=num_experts, spatial=5),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            SCConv(192, 384, kernel_size=3, padding=1, num_experts=num_experts, spatial=5),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            SCConv(384, 256, kernel_size=3, padding=1, num_experts=num_experts, spatial=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SCConv(256, 256, kernel_size=3, padding=1, num_experts=num_experts, spatial=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x