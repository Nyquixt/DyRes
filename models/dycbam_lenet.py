import torch.nn as nn
from .dycbam_conv import *

__all__ = ['DyCBAM2_LeNet', 'DyCBAM4_LeNet', 'DyCBAM5_LeNet']

class DyCBAM2_LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DyCBAM2_LeNet, self).__init__()
        self.conv = nn.Sequential(
            DyCBAMConv_2(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DyCBAMConv_2(6, 16, kernel_size=5),
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

class DyCBAM4_LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DyCBAM4_LeNet, self).__init__()
        self.conv = nn.Sequential(
            DyCBAMConv_4(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DyCBAMConv_4(6, 16, kernel_size=5),
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

class DyCBAM5_LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DyCBAM5_LeNet, self).__init__()
        self.conv = nn.Sequential(
            DyCBAMConv_5(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DyCBAMConv_5(6, 16, kernel_size=5),
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