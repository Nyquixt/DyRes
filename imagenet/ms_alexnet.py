import torch
import torch.nn as nn

from .ms_conv import MSConv

__all__ = ['MS_AlexNet']

class MS_AlexNet(nn.Module):

    def __init__(self, num_classes=1000, input_size=32):
        super(MS_AlexNet, self).__init__()
        self.features = nn.Sequential(
            MSConv(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MSConv(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MSConv(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MSConv(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MSConv(256, 256, kernel_size=3, padding=1),
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