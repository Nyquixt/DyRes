import torch
import torch.nn as nn

__all__ = ['GCNet']

class GCNet(nn.Module):
    def __init__(self, in_channels, reduction=16):
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

    def forward(self, x):
        b, c_in, h, w = x.size()
        residual = x
        # Context Modeling
        input_x = x
        input_x = input_x.view(b, c_in, h * w).unsqueeze(1) # N x 1 x C_in x H*W
        context_mask = self.conv_mask(x) # N x 1 x H x W
        context_mask = context_mask.view(b, 1, h * w) # N x 1 x H*W
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(-1) # N x 1 x H*W x 1
        context = torch.matmul(input_x, context_mask)
        context = context.view(b, c_in, 1, 1)
        # Transform
        out = self.transform(context)
        out = out + residual

        return out

def test():
    x = torch.randn(64, 128, 32, 32)
    gc = GCNet(128, 128)
    y = gc(x)
    print(y.size())

# test()