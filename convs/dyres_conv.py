import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DyResConv']

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

class DyResConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1, reduction=16, mode='A', deploy=False):
        super().__init__()
        assert mode == 'A' or mode == 'B'
        self.deploy = deploy
        self.num_experts = num_experts

        # routing function
        self.routing_func = route_func(in_channels, num_experts, reduction, mode)
        # convs
        if deploy:
            self.convs = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups) for i in range(num_experts)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_experts)])
        
    def forward(self, x):
        _, c_in, _, _ = x.size()
        routing_weight = self.routing_func(x) # N x k x C
        if self.deploy:
            inp = x * routing_weight[:, :c_in].expand_as(x)
            for i in range(1, self.num_experts):
                route = routing_weight[:, i*c_in:(i+1)*c_in]
                inp += x * route
            output = self.convs(inp)
        else:
            outputs = []
            for i in range(self.num_experts):
                route = routing_weight[:, i*c_in:(i+1)*c_in]
                attention = x * route.expand_as(x)
                out = self.convs[i](attention)
                out = self.bns[i](out)
                outputs.append(out)
            output = sum(outputs)
        return output

def test():
    x = torch.randn(1, 16, 32, 32)
    conv = DyResConv(16, 64, 3, padding=1, mode='A')
    y = conv(x)
    print(y.shape)
    conv = DyResConv(16, 64, 3, padding=1, mode='B')
    y = conv(x)
    print(y.shape)

# test()