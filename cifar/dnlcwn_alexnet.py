import torch
import torch.nn as nn

from convs.dnlcwn import DNLCWN

__all__ = ['DNLCWN_AlexNet']

class DNLCWN_AlexNet(nn.Module):

    def __init__(self, num_classes=100, bn=False, gap_mode='prior'):
        super().__init__()
        self.features = nn.Sequential(
            DNLCWN(3, 64, kernel_size=3, stride=2, bn=bn, gap_mode=gap_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DNLCWN(64, 192, kernel_size=3, bn=bn, gap_mode=gap_mode),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DNLCWN(192, 384, kernel_size=3, bn=bn, gap_mode=gap_mode),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            DNLCWN(384, 256, kernel_size=3, bn=bn, gap_mode=gap_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DNLCWN(256, 256, kernel_size=3, bn=bn, gap_mode=gap_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
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

def test():
    x = torch.randn(64, 3, 32, 32)
    nlcwn = DNLCWN_AlexNet()
    z = nlcwn(x)
    print(z.size())

# test()