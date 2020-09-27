import torch
import torch.nn as nn

__all__ = ['DGCNet', 'DGC_Attention']

class DGCNet(nn.Module):
    def __init__(self, in_channels, whiten_type=['spatial'], reduction=16, temperature=1.0):
        super().__init__()
        reduction_channels = max(in_channels // reduction, reduction)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1) #K
        
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size=1),
            nn.LayerNorm([reduction_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, in_channels, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=2)
        self.whiten_type = whiten_type
        if 'bn_affine' in whiten_type:
            self.bn_affine = nn.BatchNorm1d(1)
        if 'bn' in whiten_type:
            self.bn = nn.BatchNorm1d(1, affine=False)

    def forward(self, x):
        b, c_in, h, w = x.size()
        residual = x
        # Context Modeling
        input_x = x
        input_x = input_x.view(b, c_in, h * w).unsqueeze(1) # N x 1 x C_in x H*W
        context_mask = self.conv_mask(x) # N x 1 x H x W
        context_mask = context_mask.view(b, 1, h * w) # N x 1 x H*W
        unary = self.softmax(context_mask)
        # whitening
        if 'channel' in self.whiten_type:
            context_mean = context_mask.mean(2).unsqueeze(2)
            context_mask -= context_mean
        if 'spatial' in self.whiten_type:
            context_mean = context_mask.mean(1).unsqueeze(1)
            context_mask -= context_mean
        if 'bn_affine' in self.whiten_type:
            context_mask = self.bn_affine(context_mask)
        if 'bn' in self.whiten_type:
            context_mask = self.bn(context_mask)
        if 'ln_nostd' in self.whiten_type :
            context_mean = context_mask.mean(1).mean(1).view(context_mask.size(0), 1, 1)
            context_mask -= context_mean

        context_mask = self.softmax(context_mask)
        context_mask = context_mask + unary
        context_mask = context_mask.unsqueeze(-1) # N x 1 x H*W x 1
        context = torch.matmul(input_x, context_mask) # N x 1 x C_in x 1
        context = context.view(b, c_in, 1, 1)
        # Transform
        out = self.transform(context)
        out = out + residual

        return out

class DGC_Attention(nn.Module):
    def __init__(self, in_channels, whiten_type=['spatial'], reduction=16, temperature=1.0):
        super().__init__()
        reduction_channels = max(in_channels // reduction, reduction)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1) #K
        
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size=1),
            nn.LayerNorm([reduction_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, in_channels, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=2)
        self.whiten_type = whiten_type
        if 'bn_affine' in whiten_type:
            self.bn_affine = nn.BatchNorm1d(1)
        if 'bn' in whiten_type:
            self.bn = nn.BatchNorm1d(1, affine=False)

    def forward(self, x):
        b, c_in, h, w = x.size()
        # Context Modeling
        input_x = x
        input_x = input_x.view(b, c_in, h * w).unsqueeze(1) # N x 1 x C_in x H*W
        context_mask = self.conv_mask(x) # N x 1 x H x W
        context_mask = context_mask.view(b, 1, h * w) # N x 1 x H*W
        unary = self.softmax(context_mask)
        # whitening
        if 'channel' in self.whiten_type:
            context_mean = context_mask.mean(2).unsqueeze(2)
            context_mask -= context_mean
        if 'spatial' in self.whiten_type:
            context_mean = context_mask.mean(1).unsqueeze(1)
            context_mask -= context_mean
        if 'bn_affine' in self.whiten_type:
            context_mask = self.bn_affine(context_mask)
        if 'bn' in self.whiten_type:
            context_mask = self.bn(context_mask)
        if 'ln_nostd' in self.whiten_type :
            context_mean = context_mask.mean(1).mean(1).view(context_mask.size(0), 1, 1)
            context_mask -= context_mean

        context_mask = self.softmax(context_mask)
        context_mask = context_mask + unary
        context_mask = context_mask.unsqueeze(-1) # N x 1 x H*W x 1
        context = torch.matmul(input_x, context_mask) # N x 1 x C_in x 1
        context = context.view(b, c_in, 1, 1)
        # Transform
        out = self.transform(context)

        return out

def test():
    x = torch.randn(64, 128, 32, 32)
    nl = DGCNet(128, 128)
    y = nl(x)
    print(y.size())

# test()