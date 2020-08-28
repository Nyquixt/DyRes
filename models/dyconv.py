import torch
import torch.nn as nn 

__all__ = ['DyConv']

class DyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, reduction=16, temp=30):
        super(DyConv, self).__init__()
        self.temp = temp
        reduction_channels = max(in_channels // reduction, reduction)
        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, reduction_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, 3)
        )

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        gap = x.mean(dim=-1).mean(dim=-1)
        weights = torch.softmax(self.attention(gap) / self.temp, 1).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        return out

def test():
    x = torch.randn(4, 3 , 32, 32)
    conv = DyConv(x.size(1), 64, 3)
    y = conv(x)
    print(y.size())

# test()