import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DySepConv'] # Dynamic "Squeeze?" Conv

class DySepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, reduction=16, mode='A'):
        super(DySepConv, self).__init__()
        assert mode == 'A' or mode == 'B' or mode == 'C' or mode == 'D'
        self.mode = mode

        # Number of experts k = 3
        self.k = 3
        # Average Pool
        self.gap5 = nn.AdaptiveAvgPool2d(5)

        squeeze_channels = max(in_channels // reduction, reduction)
        
        if self.mode == 'A': # 1-1-3-3
            self.dwise_separable = nn.Sequential(
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, self.k * in_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.k * in_channels, self.k * in_channels, kernel_size=3, stride=1, groups=self.k * in_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.k * in_channels, self.k * in_channels, kernel_size=3, stride=1, groups=self.k * in_channels, bias=False)
            )
        elif self.mode == 'B': # 3-3-1-1
            self.dwise_separable = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, self.k * in_channels, kernel_size=1, stride=1, groups=1, bias=False)
            )
        elif self.mode == 'C': # 1-3-3-1
            self.dwise_separable = nn.Sequential(
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, self.k * in_channels, kernel_size=1, stride=1, groups=1, bias=False)                      
            )
        elif self.mode == 'D': # 3-1-1-3
            self.dwise_separable = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, self.k * in_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.k * in_channels, self.k * in_channels, kernel_size=3, stride=1, groups=self.k * in_channels, bias=False)
            )
            
        self.sigmoid = nn.Sigmoid()

        # 3x3 Convs
        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding, bias=bias)
        self.one_bn = nn.BatchNorm2d(out_channels)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding, bias=bias)
        self.two_bn = nn.BatchNorm2d(out_channels)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding, bias=bias)
        self.three_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        attention = self.gap5(x) # N x C x 5 x 5
        attention = self.dwise_separable(attention) # N x k*C x 1 x 1
        attention = self.sigmoid(attention)
        x1 = x * attention[:, 0:c].expand_as(x)
        y1 = self.one_bn(self.one_conv(x1))
        x2 = x * attention[:, c:2*c].expand_as(x)
        y2 = self.two_bn(self.two_conv(x2))
        x3 = x * attention[:, 2*c:3*c].expand_as(x)
        y3 = self.three_bn(self.three_conv(x3))
        return y1 + y2 + y3

def test():
    x = torch.randn(64, 128, 32, 32)
    conv = DySepConv(128, 256, 3, padding=1, mode='D')
    y = conv(x)
    print(y.shape)

# test()