import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['Distangled_NL']
'''
    Vanilla Non-Local Block for 2D only
'''
class Distangled_NL(nn.Module):
    def __init__(self, in_channels, channels, with_unary=True, whiten_type=['channel'], downsample=False, temperature=1.0):
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

        self.downsample = max_pool
        self.with_unary = with_unary
        self.whiten_type = whiten_type

        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)

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
        
        # whitening
        if 'channel' in self.whiten_type:
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)
        if 'ln_nostd' in self.whiten_type :
            key_mean = key.mean(1).mean(1).view(key.size(0), 1, 1)
            query_mean = query.mean(1).mean(1).view(query.size(0), 1, 1)
            key -= key_mean
            query -= query_mean

        # torch.bmm() is batch mm
        similiarity_map = torch.bmm(query.transpose(1, 2), key) # N x H*W x H*W
        similiarity_map = similiarity_map / self.scale
        similiarity_map = similiarity_map / self.temperature
        similiarity_map = self.softmax(similiarity_map)

        out = torch.bmm(similiarity_map, value.transpose(1, 2)) # N x H*W x C'
        out = out.transpose(1, 2) # N x C' x H*W
        out = out.view(out.size(0), out.size(1), *x.size()[2:]) # N x C' x H x W

        if self.with_unary:
            if query_mean.shape[1] ==1:
                query_mean = query_mean.expand(-1, key.shape[1], -1)
            unary = torch.bmm(query_mean.transpose(1,2),key)
            unary = self.softmax(unary)
            out_unary = torch.bmm(value, unary.permute(0,2,1)).unsqueeze(-1)
            out = out + out_unary

        out = residual + out # N x C' x H x W

        return out

def test():
    x = torch.randn(64, 128, 32, 32)
    dnl = Distangled_NL(128, 128)
    y = dnl(x)
    print(y.size())

test()