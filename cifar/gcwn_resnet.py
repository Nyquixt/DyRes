import torch
import torch.nn as nn
import torch.nn.functional as F

from convs.gc_weightnet import *

__all__ = ['GCWN_ResNet18','GCWNNL_ResNet18']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, normalized=False):
        super(BasicBlock, self).__init__()
        self.conv1 = GC_WeightNet(in_channels, channels, kernel_size=3, stride=stride, normalized=normalized)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = GC_WeightNet(channels, channels, kernel_size=3, stride=1, normalized=normalized)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Addition
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class GCWN_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, normalized=False):
        super(GCWN_ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, normalized=normalized)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, normalized=normalized)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, normalized=normalized)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, normalized=normalized)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride, normalized):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride, normalized=normalized))
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

def GCWN_ResNet18():
    return GCWN_ResNet(BasicBlock, [2, 2, 2, 2])

def GCWNNL_ResNet18():
    return GCWN_ResNet(BasicBlock, [2, 2, 2, 2], normalized=True)