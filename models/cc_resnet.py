import torch
import torch.nn as nn
import torch.nn.functional as F
from .condconv import *

__all__ = ['CC_ResNet18', 'CC_ResNet34', 'CC_ResNet50', 'CC_ResNet101', 'CC_ResNet152']

class CC_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(CC_BasicBlock, self).__init__()
        self.conv1 = CondConv(in_channels, channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = CondConv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Addition
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CC_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super(CC_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.cc = CondConv(channels, channels, kernel_size=3, stride=stride, padding=1)
        self.cc_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, self.expansion*channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.cc_bn(self.cc(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CC_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CC_ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
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
            layers.append(block(self.in_channels, channels, stride))
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

def CC_ResNet18(num_classes):
    return CC_ResNet(CC_BasicBlock, [2, 2, 2, 2], num_classes)

def CC_ResNet34(num_classes):
    return CC_ResNet(CC_BasicBlock, [3, 4, 6, 3], num_classes)

def CC_ResNet50(num_classes):
    return CC_ResNet(CC_Bottleneck, [3, 4, 6, 3], num_classes)

def CC_ResNet101(num_classes):
    return CC_ResNet(CC_Bottleneck, [3, 4, 23, 3], num_classes)

def CC_ResNet152(num_classes):
    return CC_ResNet(CC_Bottleneck, [3, 8, 36, 3], num_classes)

def test():
    x = torch.randn(128, 3, 32, 32)
    net1 = CC_ResNet18(10)
    net2 = CC_ResNet50(100)

    y1 = net1(x); print(y1.size())
    y2 = net2(x); print(y2.size())
    

# test()