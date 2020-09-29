import torch
import torch.nn as nn
import torch.nn.functional as F
import math
__all__ = ['NLCNet']

class NLC_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size=1) #K
        self.conv_value = nn.Conv2d(in_channels, in_channels // groups, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.key_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, c_in, h, w = x.size()
        # Context Modeling
        value = self.conv_value(x)
        value = value.view(b, c_in // self.groups, h * w).permute(0, 2, 1) # N x H*W x C_in//groups
        key = self.conv_mask(x) # N x C_out x H x W
        key = key.view(b, self.out_channels, h * w) # N x C_out x H*W
        key = self.key_bn(key)
        key = self.softmax(key)
        context = torch.bmm(key, value) # N x C_out x C_in
        context = context.mean(dim=0)
        return context

class NLCNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()

        self.padding = padding
        self.stride = stride
        self.groups = groups

        self.gc_att = NLC_Attention(in_channels, out_channels, groups)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        att = self.gc_att(x)
        att = self.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
        weight = self.weight * att
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return out

def test():
    x = torch.randn(64, 128, 32, 32)
    nlc = NLCNet(128, 256, 3, groups=2)
    y = nlc(x)
    print(y.size())

# test()