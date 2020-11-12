import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DyResConv_Inf']

class route_func(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=16, mode='A'):
        super().__init__()
        assert mode == 'A' or mode == 'B'
        # Global Average Pool
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap3 = nn.AdaptiveAvgPool2d(3)
        self.gap5 = nn.AdaptiveAvgPool2d(5)

        squeeze_channels = max(in_channels // reduction, reduction)
        
        if mode == 'A': # 1-3-3-1
            self.dwise_separable = nn.Sequential(
                nn.Conv2d(3 * in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, num_experts * in_channels, kernel_size=1, stride=1, groups=1, bias=False)
            )
        elif mode == 'B': # 3-1-1-3
            self.dwise_separable = nn.Sequential(
                nn.Conv2d(3 * in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(squeeze_channels, in_channels, kernel_size=1, stride=1, groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, num_experts * in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
            )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, _, _ = x.size()
        a1 = self.gap1(x)
        a3 = F.interpolate(self.gap3(x), 5, mode='bicubic', align_corners=False)
        a5 = self.gap5(x)
        a1 = a1.expand_as(a5)
        attention = torch.cat([a1, a3, a5], dim=1)
        attention = self.sigmoid(self.dwise_separable(attention))
        return attention

class DyResConv_Inf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1, reduction=16, mode='A'):
        super().__init__()
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, num_experts, reduction, mode)
        # convs
        self.convs = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels, kernel_size, kernel_size))
    
    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k*C
        routing_weight = routing_weight.view(-1, self.num_experts, self.out_channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b, c_in, h, w = x.size()
        x = x.view(1, -1, h, w)
        weight = self.convs.unsqueeze(0)
        combined_weight = (weight * routing_weight).view(b*self.num_experts, self.out_channels, c_in, self.kernel_size, self.kernel_size)
        combined_weight = torch.sum(combined_weight, dim=0)
        output = F.conv2d(x, weight=combined_weight,
                            stride=self.stride, padding=self.padding, groups=self.groups * b)
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

def test():
    x = torch.randn(1, 16, 32, 32)
    conv = DyResConv_Inf(16, 64, 3, padding=1, mode='A')
    y = conv(x)
    print(y.shape)
    conv = DyResConv_Inf(16, 64, 3, padding=1, mode='B')
    y = conv(x)
    print(y.shape)

# test()