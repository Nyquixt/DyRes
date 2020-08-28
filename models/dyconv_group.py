import torch
import torch.nn as nn 

__all__ = ['DyConvGroup']

class DyConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, reduction=16):
        super(DyConvGroup, self).__init__()
        reduction_channels = max(in_channels // reduction, reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size=1, groups=reduction_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1) # N x 3 x N x H_out x W_out
        gap = self.gap(x)
        weights = self.attention(gap)
        out = weights.unsqueeze(dim=-1) * all_out
        out = out.sum(dim=1, keepdim=False)
        return out

def test():
    x = torch.randn(64, 128 , 32, 32)
    conv = DyConvGroup(x.size(1), 64, 3, padding=1)
    y = conv(x)
    print(y.size())

# test()