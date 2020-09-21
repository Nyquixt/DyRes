import torch
import torch.nn as nn
import torch.nn.functional as F

''' 
https://github.com/megvii-model/WeightNet/blob/master/weightnet.py
'''

__all__ = ['WeightNet', 'WeightNet_DW', 'WeightNet_Tanh', 'WeightNet_DW_Tanh']

class WeightNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction_ratio=16, M=2, G=2):
        super().__init__()

        self.M = M
        self.G = G

        self.padding = kernel_size // 2
        input_gap = max(reduction_ratio, in_channels // reduction_ratio)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.fc1 = nn.Conv2d(input_gap, self.M * out_channels, 1, 1, 0, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Conv2d(self.M * out_channels, out_channels * in_channels * kernel_size * kernel_size, 1, 1, 0, groups=self.G * out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, input_gap, 1, 1, 0, bias=True)

    def forward(self, x):
        b, _, _, _ = x.size()
        x_gap = self.avg_pool(x) # N x C_in x 1 x 1
        x_gap = self.reduce(x_gap) # N x C_in / r x 1 x 1

        x_w = self.fc1(x_gap) # N x M(C_out) x 1 x 1
        x_w = self.sigmoid(x_w)
        x_w = self.fc2(x_w) # N x (C_out)(C_in)(kH)(kW) x 1 x 1

        x = x.view(1, -1, x.size(2), x.size(3)) # 1 x N(C_in) x H x W
        x_w = x_w.view(-1, self.in_channels, self.kernel_size, self.kernel_size) # (C_out)(N) x C_in x kH, kW
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b)
        x = x.view(-1, self.out_channels, x.size(2), x.size(3)) # N x C_out x H x W
        return x

class WeightNet_DW(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, reduction_ratio=16, M=2, G=2):
        super().__init__()

        self.M = M
        self.G = G # lambda = M // G

        self.padding = kernel_size // 2
        input_gap = max(reduction_ratio, channels // reduction_ratio)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.fc1 = nn.Conv2d(input_gap, self.M // self.G * channels, 1, 1, 0, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Conv2d(self.M // self.G * channels, channels * kernel_size * kernel_size, 1, 1, 0, groups=channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, input_gap, 1, 1, 0, bias=True)

    def forward(self, x):
        b, _, _, _ = x.size()
        x_gap = self.avg_pool(x) # N x C_in x 1 x 1
        x_gap = self.reduce(x_gap) # N x C_in / r x 1 x 1

        x_w = self.fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.fc2(x_w)

        x = x.view(1, -1, x.size(2), x.size(3))
        x_w = x_w.view(-1, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b * self.channels)
        x = x.view(-1, self.channels, x.size(2), x.size(3))
        return x

class WeightNet_Tanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction_ratio=16, M=2, G=2):
        super().__init__()

        self.M = M
        self.G = G

        self.padding = kernel_size // 2
        input_gap = max(reduction_ratio, in_channels // reduction_ratio)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.fc1 = nn.Conv2d(input_gap, self.M * out_channels, 1, 1, 0, groups=1, bias=False)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Conv2d(self.M * out_channels, out_channels * in_channels * kernel_size * kernel_size, 1, 1, 0, groups=self.G * out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, input_gap, 1, 1, 0, bias=True)

    def forward(self, x):
        b, _, _, _ = x.size()
        x_gap = self.avg_pool(x) # N x C_in x 1 x 1
        x_gap = self.reduce(x_gap) # N x C_in / r x 1 x 1

        x_w = self.fc1(x_gap) # N x M(C_out) x 1 x 1
        x_w = self.tanh(x_w)
        x_w = self.fc2(x_w) # N x (C_out)(C_in)(kH)(kW) x 1 x 1

        x = x.view(1, -1, x.size(2), x.size(3)) # 1 x N(C_in) x H x W
        x_w = x_w.view(-1, self.in_channels, self.kernel_size, self.kernel_size) # (C_out)(N) x C_in x kH, kW
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b)
        x = x.view(-1, self.out_channels, x.size(2), x.size(3)) # N x C_out x H x W
        return x

class WeightNet_DW_Tanh(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, reduction_ratio=16, M=2, G=2):
        super().__init__()

        self.M = M
        self.G = G # lambda = M // G

        self.padding = kernel_size // 2
        input_gap = max(reduction_ratio, channels // reduction_ratio)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.fc1 = nn.Conv2d(input_gap, self.M // self.G * channels, 1, 1, 0, groups=1, bias=False)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Conv2d(self.M // self.G * channels, channels * kernel_size * kernel_size, 1, 1, 0, groups=channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, input_gap, 1, 1, 0, bias=True)

    def forward(self, x):
        b, _, _, _ = x.size()
        x_gap = self.avg_pool(x) # N x C_in x 1 x 1
        x_gap = self.reduce(x_gap) # N x C_in / r x 1 x 1

        x_w = self.fc1(x_gap)
        x_w = self.tanh(x_w)
        x_w = self.fc2(x_w)

        x = x.view(1, -1, x.size(2), x.size(3))
        x_w = x_w.view(-1, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b * self.channels)
        x = x.view(-1, self.channels, x.size(2), x.size(3))
        return x

def test():
    x = torch.randn(64, 128, 32, 32)
    wn = WeightNet(128, 256, 3)
    wn_dw = WeightNet_DW(128, 3)
    y = wn(x)
    y_dw = wn_dw(x) 
    print(y.size(), y_dw.size())

# test()
