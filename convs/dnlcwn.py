import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DNLCWN', 'DNLCWN_DW']

class DNLCLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=16, bn=False, stride=2):
        """
        Non-Local along channel dimension.
        :param in_channels: in-channel number
        :param kernel_size: Convolution embedding/projection kernel size.
        :param reduction: reduce intermidiate channel number for computational efficiency.
        :param stride: reduce spatial size by convolutional stride.
        """
        super().__init__()
        reduction_channels = in_channels // reduction if in_channels > reduction else 1
        padding = (kernel_size - 1) // 2
        self.theta = nn.Conv2d(in_channels, reduction_channels, kernel_size=kernel_size, padding=padding, groups=reduction_channels, stride=stride)
        self.phi = nn.Conv2d(in_channels, reduction_channels, kernel_size=kernel_size, padding=padding, groups=reduction_channels, stride=stride)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, reduction_channels, kernel_size=1, groups=reduction_channels, stride=stride)
        self.W = nn.Conv2d(reduction_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=reduction_channels)

        self.bn = bn
        if bn:
            self.theta_bn = nn.BatchNorm1d(reduction_channels)
            self.phi_bn = nn.BatchNorm1d(reduction_channels)

    def forward(self, x):
        b, _, _, _ = x.size()
        theta = self.theta(x)
        _, c, w, h = theta.size()
        theta = theta.view(b, c, -1)  # [b,c,w*h]
        phi = self.phi(x).view(b, c, -1) # [b,c,w*h]

        if self.bn:
            theta = self.theta_bn(theta)
            phi = self.phi_bn(phi)

        phi = phi.permute(0, 2, 1) # [n,w*h,c]
        gap = self.gap(x)
        g = self.fc(gap).squeeze(dim=-1)
        Mat = torch.matmul(theta, phi) # [b, c, c] c=c_in/r
        Mat = F.softmax(Mat, dim=-1)
        out = torch.matmul(Mat, g).unsqueeze(dim=-1)
        return out

class DNLCWN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, reduction=16):
        super().__init__()

        self.padding = kernel_size // 2
        reduction_channels = in_channels // reduction if in_channels > reduction else 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.dnlc = DNLCLayer(in_channels=in_channels, reduction=reduction, bn=bn)
        self.fc = nn.Conv2d(reduction_channels, out_channels * in_channels * kernel_size * kernel_size, 1, 1, 0, groups=reduction_channels)

    def forward(self, x):
        b, _, _, _ = x.size()

        x_w = self.dnlc(x) # N x c_in/r x 1 x 1
        x_w = self.fc(x_w) # N x (C_out)(C_in)(kH)(kW) x 1 x 1

        x = x.view(1, -1, x.size(2), x.size(3)) # 1 x N(C_in) x H x W
        x_w = x_w.view(-1, self.in_channels, self.kernel_size, self.kernel_size) # (C_out)(N) x C_in x kH, kW
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b)
        x = x.view(-1, self.out_channels, x.size(2), x.size(3)) # N x C_out x H x W
        return x

class DNLCWN_DW(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, bn=False, reduction=16):
        super().__init__()

        self.padding = kernel_size // 2
        reduction_channels = max(reduction, channels // reduction)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.dnlc = DNLCLayer(in_channels=channels, reduction=reduction, bn=bn)
        self.fc = nn.Conv2d(reduction_channels, channels * kernel_size * kernel_size, 1, 1, 0, groups=reduction_channels)

    def forward(self, x):
        b, _, _, _ = x.size()

        x_w = self.dnlc(x) # N x c_in/r x 1 x 1
        x_w = self.fc(x_w) # N x (C_out)(C_in)(kH)(kW) x 1 x 1

        x = x.view(1, -1, x.size(2), x.size(3)) # 1 x N(C_in) x H x W
        x_w = x_w.view(-1, 1, self.kernel_size, self.kernel_size) # (C_out)(N) x C_in x kH, kW
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=self.channels * b)
        x = x.view(-1, self.channels, x.size(2), x.size(3)) # N x C_out x H x W
        return x

def test():
    x = torch.randn(64, 128, 32, 32)
    dnlcwn = DNLCWN(in_channels=128, out_channels=256, kernel_size=3, bn=True)
    y = dnlcwn(x)
    print(y.size())

    dnlcwndw = DNLCWN_DW(channels=128, kernel_size=3, bn=True)
    y = dnlcwndw(x)
    print(y.size())

# test()