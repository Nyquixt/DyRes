import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DySQConv'] # Dynamic "Squeeze?" Conv

class DySQConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction=16):
        super(DySQConv, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1) # Global Average Pool

        squeeze_channels = in_channels // reduction if in_channels > reduction else 1
        groups = squeeze_channels // 4 if squeeze_channels > 4 else 1
        # groups = 1

        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=stride,
                                groups=groups, bias=bias)
        self.excite = nn.Conv2d(squeeze_channels, 3 * in_channels, kernel_size=1, stride=stride,
                                groups=groups, bias=bias)
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
        attention = self.gap(x)
        attention = F.relu(self.squeeze(attention))
        attention = self.excite(attention)
        attention = self.softmax(attention.squeeze(dim=-1).squeeze(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        attention = attention.view(b, 3, -1, 1, 1).permute(1, 0, 2, 3, 4)
        x1 = x * attention[0].expand_as(x)
        y1 = self.one_bn(self.one_conv(x1))
        x2 = x * attention[1].expand_as(x)
        y2 = self.two_bn(self.two_conv(x2))
        x3 = x * attention[2].expand_as(x)
        y3 = self.three_bn(self.three_conv(x3))
        return y1 + y2 + y3

def test():
    x = torch.randn(64, 128, 32, 32)
    conv = DySQConv(128, 256, 3, padding=1)
    y = conv(x)
    print(y.shape)

# test()