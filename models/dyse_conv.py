import torch
import torch.nn as nn 

__all__ = ['DySE_Conv']

class DySEConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction=16):
        super(DySEConv, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.one_bn = nn.BatchNorm2d(out_channels)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_bn = nn.BatchNorm2d(out_channels)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_bn = nn.BatchNorm2d(out_channels)
        self.kernel_attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(1)
        )
        if in_channels < reduction:
            self.channel_attention = nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.ReLU(inplace=True),
                nn.Linear(1, in_channels),
                nn.Sigmoid()
            )
        else:
            self.channel_attention = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        gap = x.mean(dim=-1).mean(dim=-1)
        channel_attention = self.channel_attention(gap)
        # apply channel attention to input x
        out = x * channel_attention.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(x)
        one_out = self.one_bn(self.one_conv(out)).unsqueeze(dim=1)
        two_out = self.two_bn(self.two_conv(out)).unsqueeze(dim=1)
        three_out = self.three_bn(self.three_conv(out)).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)

        weights = self.kernel_attention(gap).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # apply kernel attention
        kernels = weights * all_out
        kernels = kernels.sum(dim=1, keepdim=False)

        return kernels

def test():
    x = torch.randn(4, 64 , 32, 32)
    conv = DySEConv(x.size(1), 128, 3, padding=1)
    y = conv(x)
    print(y.size())

test()