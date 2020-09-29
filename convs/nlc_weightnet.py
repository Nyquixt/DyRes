import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NLC_WeightNet']

class NLC_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, normalized=False):
        super().__init__()
        self.out_channels = out_channels
        reduction_channels = max(in_channels // reduction, reduction)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size=1) #K
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c_in, h, w = x.size()
        # Context Modeling
        input_x = x
        input_x = input_x.view(b, c_in, h * w).permute(0, 2, 1) # N x x H*W
        context_mask = self.conv_mask(x) # N x C_out x H x W
        context_mask = context_mask.view(b, self.out_channels, h * w) # N x C_out x H*W
        # context_mask = context_mask.permute(0, 2, 1) # N x H*W x C_out
        context_mask = self.softmax(context_mask)
        context = torch.bmm(context_mask, input_x) # N x C_in x C_in
        context = context.mean(0)
        # Weight Learning
        # out = context
        return context

class NLC_WeightNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=16, M=2, G=2, normalized=False):
        super().__init__()

        self.M = M
        self.G = G

        self.padding = kernel_size // 2
        reduction_channels = max(reduction, in_channels // reduction)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.gc_att = NLC_Attention(in_channels=in_channels, out_channels=self.M * out_channels, reduction=reduction, normalized=normalized)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Conv2d(self.M * out_channels, out_channels * in_channels * kernel_size * kernel_size, 1, 1, 0, groups=self.G * out_channels)

    def forward(self, x):
        b, _, _, _ = x.size()

        x_w = self.gc_att(x) # N x M(C_out) x 1 x 1
        x_w = self.sigmoid(x_w)
        x_w = self.fc(x_w) # N x (C_out)(C_in)(kH)(kW) x 1 x 1

        x = x.view(1, -1, x.size(2), x.size(3)) # 1 x N(C_in) x H x W
        x_w = x_w.view(-1, self.in_channels, self.kernel_size, self.kernel_size) # (C_out)(N) x C_in x kH, kW
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.padding, groups=b)
        x = x.view(-1, self.out_channels, x.size(2), x.size(3)) # N x C_out x H x W
        return x

def test():
    x = torch.randn(64, 128, 32, 32)
    nlcwn = NLC_Attention(128, 256, 3)
    y = nlcwn(x)
    print(y.size())

test()