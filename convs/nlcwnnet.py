import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NLCWNNet', 'NLCWNNet_DW']

class NLC_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size=1) #K
        self.conv_value = nn.Conv2d(in_channels, in_channels // groups, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.key_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, c_in, h, w = x.size()
        # Context Modeling
        value = self.conv_value(x)
        value = value.view(b, c_in // self.groups, h * w).permute(0, 2, 1) # N x H*W x C_in//groups
        key = self.conv_mask(x) # N x C_out x H x W
        key = key.view(b, self.out_channels, h * w) # N x C_out x H*W
        key = self.key_bn(key)
        key = self.softmax(key)
        context = torch.bmm(key, value) # N x C_out x C_in
        # context = context.mean(dim=0)
        return context

class NLCWNNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups

        self.gc_att = NLC_Attention(in_channels, out_channels, groups)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Conv2d(out_channels * in_channels, out_channels * in_channels * kernel_size * kernel_size, 
                        kernel_size=1, groups=out_channels * in_channels)
        
    def forward(self, x):
        b, c_in, h, w = x.size()
        att = self.gc_att(x)
        att = self.sigmoid(att).view(b, -1, 1).unsqueeze(-1)
        weight = self.fc(att).view(-1, c_in, self.kernel_size, self.kernel_size)
        x = x.view(1, -1, h, w)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=b)
        out = out.view(-1, self.out_channels, out.size(2), out.size(3))
        return out

# Depthwise Model for MobileNetV2 in our experiments
class NLCWNNet_DW(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups

        self.gc_att = NLC_Attention(channels, channels, groups)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Conv2d(channels * channels, channels * kernel_size * kernel_size, 
                        kernel_size=1, groups=channels)
        

    def forward(self, x):
        b, c_in, h, w = x.size()
        att = self.gc_att(x)
        att = self.sigmoid(att).view(b, -1, 1).unsqueeze(-1)
        weight = self.fc(att).view(-1, 1, self.kernel_size, self.kernel_size)
        x = x.view(1, -1, h, w)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.channels * b)
        out = out.view(-1, self.channels, out.size(2), out.size(3))
        return out

def test():
    x = torch.randn(64, 128, 32, 32)
    nlcwn = NLCWNNet(128, 256, 3, stride=2, padding=1, groups=1)
    z = nlcwn(x)
    print(z.size())
    nlcwndw = NLCWNNet_DW(128, 3, padding=1, groups=1)
    y = nlcwndw(x)
    print(y.size())

# test()