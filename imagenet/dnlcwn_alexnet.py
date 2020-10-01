import torch
import torch.nn as nn

from convs.dnlcwn import DNLCWN

__all__ = ['DNLCWN_AlexNet']

class DNLCWN_AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            DNLCWN(3, 64, kernel_size=11, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            DNLCWN(64, 192, kernel_size=5),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            DNLCWN(192, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            DNLCWN(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DNLCWN(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x