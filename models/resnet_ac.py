import torch
import torch.nn as nn
import torch.nn.functional as F
from .acnet import *

__all__ = ['ResNet_AC20', 'ResNet_AC56', 'ResNet_AC110', 'ResNet_AC164']

class Basic_AC_Block(nn.Module):

    def __init__(self, in_channels, channels, stride=1):
        super(Basic_AC_Block, self).__init__()
        self.conv1 = AC_Conv2dBN(in_channels, channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = AC_Conv2dBN(channels, channels, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                AC_Conv2dBN(in_channels, channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # Addition
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_AC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_AC, self).__init__()
        self.in_channels = 16

        self.conv1 = AC_Conv2dBN(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet_AC20(num_classes):
    return ResNet_AC(Basic_AC_Block, [6, 6, 6], num_classes)

def ResNet_AC56(num_classes):
    return ResNet_AC(Basic_AC_Block, [9, 9, 9], num_classes)

def ResNet_AC110(num_classes):
    return ResNet_AC(Basic_AC_Block, [18, 18, 18], num_classes)

def ResNet_AC164(num_classes):
    return ResNet_AC(Basic_AC_Block, [27, 27, 27], num_classes)

def test():
    x = torch.randn(128, 3, 32, 32)
    net1 = ResNet_AC20(10)
    net2 = ResNet_AC56(100)

    y1 = net1(x); print(y1.size())
    y2 = net2(x); print(y2.size())
    

# test()