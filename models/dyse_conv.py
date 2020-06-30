import torch
import torch.nn as nn 

__all__ = ['DySE_Conv', 'DySE_Conv_2B']

class DySE_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction=16):
        super(DySE_Conv, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(out_channels // reduction, 3)
        self.softmax = nn.Softmax(1)

        self.fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        gap = x.mean(dim=-1).mean(dim=-1)
        attention = self.attention(gap)
        weights = self.softmax(self.fc1(attention)).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        kernels = weights * all_out
        kernels = kernels.sum(dim=1, keepdim=False)
        channels = self.sigmoid(self.fc2(attention))
        return kernels * channels.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(kernels)

class DySE_Conv_2B(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction=16):
        super(DySE_Conv_2B, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.kernel_attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(1)
        )

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        gap = x.mean(dim=-1).mean(dim=-1)

        weights = self.kernel_attention(gap).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        kernels = weights * all_out
        kernels = kernels.sum(dim=1, keepdim=False)

        channels = self.channel_attention(gap)
        return kernels * channels.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(kernels)

def test():
    x = torch.randn(4, 64 , 32, 32)
    conv = DySE_Conv_2B(x.size(1), 128, 3, padding=1)
    y = conv(x)
    print(y.size())

# test()