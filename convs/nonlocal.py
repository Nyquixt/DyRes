import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['NonLocal']
'''
    Vanilla Non-Local Block for 2D only
'''
class NonLocal(nn.Module):
    def __init__(self, in_channels, channels, downsample=False, temperature=1.0):
        super().__init__()
        
        self.conv_query = nn.Conv2d(in_channels, channels, kernel_size=1) #Q
        self.conv_key = nn.Conv2d(in_channels, channels, kernel_size=1) #K
        self.conv_value = nn.Conv2d(in_channels, channels, kernel_size=1) #V

        self.conv_out = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(2)
        self.scale = math.sqrt(channels)
        self.temperature = temperature

        if downsample:
            max_pool = nn.MaxPool2d(2, 2)
        else:
            max_pool = None

        self.downsample = maxpool

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        query = self.conv_query(x) # N x C' x H x W
        key = self.conv_key(input_x) # N x C' x H x W
        value = self.conv_value(input_x) # N x C' x H x W

        query = query.view(query.size(0), query.size(1), -1) # N x C' x H*W
        key = key.view(key.size(0), key.size(1), -1) # N x C' x H * W
        value = value.view(value.size(0), value.size(1), -1) # N x C' x H*W
        
        # torch.bmm() is batch mm
        similiarity_map = torch.bmm(query.transpose(1, 2), key) # N x H*W x H*W
        similiarity_map = similiarity_map / self.scale
        similiarity_map = similiarity_map / self.temperature
        similiarity_map = self.softmax(similiarity_map)

        out = torch.bmm(similiarity_map, value.transpose(1, 2)) # N x H*W x C'
        out = out.transpose(1, 2) # N x C' x H*W
        out = out.view(out.size(0), out.size(1), *x.size()[2:]) # N x C' x H x W
        out = residual + out # N x C' x H x W

        return out

def test():
    x = torch.randn(64, 128, 32, 32)
    nl = NonLocal(128, 128)
    y = nl(x)
    print(y.size())

# test()