import torch
import torch.nn as nn

from convs.gc_weightnet import GC_WeightNet

__all__ = ['GCWN_AlexNet']

class GCWN_AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(GCWN_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            GC_WeightNet(64, 192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            GC_WeightNet(192, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            GC_WeightNet(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            GC_WeightNet(256, 256, kernel_size=3),
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
    x = torch.randn(256, 3, 32, 32)
    net = GCWN_AlexNet()
    y = net(x)
    print(y.shape)

# test()