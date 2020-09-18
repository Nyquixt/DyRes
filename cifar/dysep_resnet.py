import torch
import torch.nn as nn
import torch.nn.functional as F

from .dysep_conv import *

__all__ = ['DySep_ResNet18']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, mode='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = DySepConv(in_channels, channels, 
                        kernel_size=3, stride=stride, padding=1, mode=mode)
        self.conv2 = DySepConv(channels, channels, 
                        kernel_size=3, stride=1, padding=1, mode=mode)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # Addition
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, mode='A'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = DySepConv(channels, channels, kernel_size=3, stride=stride, padding=1, mode=mode)
        self.conv3 = nn.Conv2d(channels, self.expansion*channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DySep_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, mode='A'):
        super(DySep_ResNet, self).__init__()
        self.mode = mode
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride, mode=self.mode))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DySep_ResNet18(mode):
    return DySep_ResNet(BasicBlock, [2, 2, 2, 2], mode=mode)