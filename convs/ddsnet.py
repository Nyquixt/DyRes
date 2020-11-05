import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DDSConv']

class route_func(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, reduction=16):
        super().__init__()
        # Global Average Pool
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap3 = nn.AdaptiveAvgPool2d(3)

        squeeze_channels = max(in_channels // reduction, reduction)
        
        self.dwise_separable = nn.Sequential(
            nn.Conv2d(2 * in_channels, squeeze_channels, kernel_size=1, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=1, groups=squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, num_experts * out_channels, kernel_size=1, stride=1, groups=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, _, _ = x.size()
        a1 = self.gap1(x)
        a3 = self.gap3(x)
        a1 = a1.expand_as(a3)
        attention = torch.cat([a1, a3], dim=1)
        attention = self.sigmoid(self.dwise_separable(attention))
        return attention

class DDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1, reduction=16, deploy=True):
        super().__init__()
        self.deploy = deploy
        self.num_experts = num_experts
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, out_channels, num_experts, reduction)
        # convs
        if deploy:
            self.convs = [nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) for i in range(num_experts)]
        else:
            self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups) for i in range(num_experts)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_experts)])
        
    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k*C
        if self.deploy:
            convs = []
            for i in range(self.num_experts):
                route = routing_weight[:, i * self.out_channels : (i+1) * self.out_channels].squeeze(0).unsqueeze(-1)
                weight = self.convs[i]
                weight = weight * route
                convs.append(weight)
            conv = sum(convs)
            output = F.conv2d(x, weight=conv, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            outputs = []
            for i in range(self.num_experts):
                route = routing_weight[:, i * self.out_channels : (i+1) * self.out_channels]
                # X * W
                out = self.convs[i](x)
                out = self.bns[i](out)
                out = out * route.expand_as(out)
                outputs.append(out)
            output = sum(outputs)
        return output

def test():
    x = torch.randn(1, 16, 32, 32)
    conv = DDSConv(16, 64, 3, padding=1)
    y = conv(x)
    print(y.shape)
    conv = DDSConv(16, 64, 3, padding=1)
    y = conv(x)
    print(y.shape)

# test()