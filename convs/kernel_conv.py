import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['Kernel_Conv']

class Kernel_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels // groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.avgpool = nn.AdaptiveAvgPool2d(kernel_size + 2)
        self.maxpool = nn.AdaptiveMaxPool2d(kernel_size + 2)
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, groups=self.in_channels)
        self.conv2 = nn.Conv2d(self.in_channels, out_channels * self.in_channels, kernel_size=1, groups=self.in_channels)
        self.conv_att  = nn.Conv2d(in_channels, out_channels * self.in_channels, kernel_size=3, groups=self.in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        maxi = self.maxpool(x)
        att = self.conv_att(maxi)
        att = self.sigmoid(att) # N x C_out*C_in x kH x kW
        att = att.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        avg = self.avgpool(x)
        weight = F.relu(self.conv1(avg))
        weight = self.conv2(weight) # N x C_out*C_in x kH x kW
        weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        b, _, h, w = x.size()
        x = x.view(1, -1, h, w)
        output = F.conv2d(x, weight=weight * att, bias=None,
                            stride=self.stride, padding=self.padding, groups=self.groups * b)
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

def test():
    x = torch.randn(4, 16 , 32, 32)
    conv = Kernel_Conv(x.size(1), 64, kernel_size=3, padding=1, groups=8)
    y = conv(x)
    print(y.size())

# test()