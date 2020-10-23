import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['Kernel_Conv']

class Kernel_Conv(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.avgpool = nn.AvgPool2d(kernel_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels * in_channels, kernel_size=1)
        self.conv_att  = nn.Conv2d(in_channels, out_channels * in_channels, kernel_size=1)
        
    def forward(self, x):
        avg = self.avgpool(x)
        att = self.conv_att(avg)
        att = F.sigmoid(att) # N x C_out*C_in x kH x kW
        att = att.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        weight = F.relu(self.conv1(x))
        weight = self.conv2(weight) # N x C_out*C_in x kH x kW
        weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        b, _, h, w = x.size()
        x = x.view(1, -1, h, w)
        output = F.conv2d(x, weight=weight, bias=None,
                            stride=self.stride, padding=self.padding, groups=self.groups * b)
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

def test():
    x = torch.randn(4, 16 , 32, 32)
    conv = Kernel_Conv(x.size(1), 64, 3, padding=1)
    y = conv(x)
    print(y.size())