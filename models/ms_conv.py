import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MSConv']

class MSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(MSConv, self).__init__()
        reduction = max(in_channels // 16, 16)

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        
        self.scale_1 = nn.AdaptiveAvgPool2d(1)

        self.scale_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(in_channels, in_channels, 3, stride=1),
        )

        self.scale_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            nn.Conv2d(in_channels, reduction, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, in_channels, 3, stride=1),
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(3 * in_channels, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)

        scale_1 = self.scale_1(x)
        scale_3 = self.scale_3(x)
        scale_5 = self.scale_5(x)
        scale = F.relu(torch.cat([scale_1, scale_3, scale_5], dim=1))

        attention = self.attention(scale).unsqueeze(dim=-1)
        
        out = attention * all_out
        out = out.sum(dim=1, keepdim=False)
        return out

def test():
    x = torch.randn(4, 3 , 32, 32)
    conv1 = MSConv(x.size(1), 64, 3, padding=1)
    y1 = conv1(x)
    print(y1.size())

# test()