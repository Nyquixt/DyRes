import torch
import torch.nn as nn

from .ms_conv import MS2Conv

__all__ = ['MS2_AlexNet']

class MS2_AlexNet(nn.Module):

    def __init__(self, num_classes=10, input_size=32):
        super(MS2_AlexNet, self).__init__()
        self.features = nn.Sequential(
            MS2Conv(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            MS2Conv(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            MS2Conv(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            MS2Conv(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            MS2Conv(256, 256, kernel_size=3, padding=1),
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