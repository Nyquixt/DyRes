import torch
import torch.nn as nn
import torch.nn.functional as F
from convs.gcnet import GC_Attention

__all__ = ['GC_WeightNet']

class GC_WeightNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=16):
        super().__init__()

        self.padding = kernel_size // 2
        reduction_channels = max(reduction, in_channels // reduction)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.gc_att = GC_Attention(in_channels=in_channels, reduction=reduction)
        self.fc = nn.Conv2d(in_channels, out_channels * in_channels * kernel_size * kernel_size, 1, 1, 0, groups=in_channels)

    def forward(self, x):
        b, _, _, _ = x.size()

        x_w = self.gc_att(x) # N x C_in x 1 x 1
        x_w = self.fc(x_w) # N x (C_out)(C_in)(kH)(kW) x 1 x 1

        x = x.view(1, -1, x.size(2), x.size(3)) # 1 x N(C_in) x H x W
        x_w = x_w.view(-1, self.in_channels, self.kernel_size, self.kernel_size) # (C_out)(N) x C_in x kH, kW
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b)
        x = x.view(-1, self.out_channels, x.size(2), x.size(3)) # N x C_out x H x W
        return x

def test():
    x = torch.randn(64, 128, 32, 32)
    gcwn = GC_WeightNet(128, 128, 3)
    y = gcwn(x)
    print(y.size())

# test()