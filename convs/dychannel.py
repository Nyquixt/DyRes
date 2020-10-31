import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['DyChannel']

# TODO: if use bias, out_channels is used in route_func instead

class route_func(nn.Module):
    def __init__(self, in_channels, num_experts, reduction=16, activation='sigmoid'):
        super().__init__()

        reduction_channels = max(in_channels // reduction, reduction)
        self.num_experts = num_experts

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, reduction_channels)
        self.fc2 = nn.Linear(reduction_channels, num_experts * in_channels)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(2)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.avgpool(x).view(b, c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.activation(x)
        return x.unsqueeze(-1).unsqueeze(-1)

class DyChannel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, num_experts=3, reduction=16, activation='sigmoid'):
        super().__init__()

        self.num_experts = num_experts

        # routing function
        self.routing_func = route_func(in_channels, num_experts, reduction)
        # convs
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups) for i in range(num_experts)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(num_experts)])

    def forward(self, x):
        outputs = []
        
        _, c_in, _, _ = x.size()
        routing_weight = self.routing_func(x) # N x k x C

        for i in range(self.num_experts):
            route = routing_weight[:, i*c_in:(i+1)*c_in]
            attention = x * route.expand_as(x)
            out = self.convs[i](attention)
            out = self.bns[i](out)
            outputs.append(out)
        
        return sum(outputs)

def test():
    x = torch.randn(4, 16 , 32, 32)
    conv = DyChannel(x.size(1), 64, 3, padding=1, activation='softmax', num_experts=5)
    y = conv(x)
    print(y.size())

# test()