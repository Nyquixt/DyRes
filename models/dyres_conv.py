import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DyResConv'] # Dynamic "Squeeze?" Conv

class DyResConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction=16):
        super(DyResConv, self).__init__()

        # Global Average Pool
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap3 = nn.AdaptiveAvgPool2d(3)
        self.gap5 = nn.AdaptiveAvgPool2d(5)

        squeeze_channels = in_channels // reduction if in_channels > reduction else 1

        self.pointwise1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1,
                                groups=squeeze_channels, bias=False)
        self.pointwise2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1, stride=1,
                                groups=squeeze_channels, bias=False)

        self.channelwise1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=3, stride=1,
                                groups=squeeze_channels, bias=False)
        self.channelwise2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=3, stride=1,
                                groups=squeeze_channels, bias=False)               
        
        self.softmax = nn.Softmax(1)

        # 3x3 Convs
        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.one_bn = nn.BatchNorm2d(out_channels)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.two_bn = nn.BatchNorm2d(out_channels)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.three_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        a1 = self.gap1(x)
        a3 = F.interpolate(self.gap3(x), 5, mode='bicubic', align_corners=False)
        a5 = self.gap5(x)
        a1 = a1.expand_as(a5)
        attention = torch.cat([a1, a3, a5], dim=0)
        attention = F.relu(self.pointwise1(attention))
        attention = F.relu(self.pointwise2(attention))
        attention = F.relu(self.channelwise1(attention))
        attention = self.channelwise2(attention)
        attention = self.softmax(attention.squeeze(dim=-1).squeeze(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x1 = x * attention[0:b].expand_as(x)
        y1 = self.one_bn(self.one_conv(x1))
        x2 = x * attention[b:2*b].expand_as(x)
        y2 = self.two_bn(self.two_conv(x2))
        x3 = x * attention[2*b:3*b].expand_as(x)
        y3 = self.three_bn(self.three_conv(x3))
        return y1 + y2 + y3

def test():
    x = torch.randn(64, 128, 32, 32)
    conv = DyResConv(128, 256, 3, padding=1)
    y = conv(x)
    print(y.shape)

# test()