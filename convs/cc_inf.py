import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CondConv_Inf']

class route_func(nn.Module):

    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class CondConv_Inf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts=3, stride=1, padding=0, groups=1, reduction=16, mode='A'):
        super().__init__()
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, num_experts)
        # convs
        self.convs = [nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) for i in range(num_experts)]
    
    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k
        convs = []
        for i in range(self.num_experts):
            route = routing_weight[:, i]
            weight = self.convs[i]
            weight = weight * route
            convs.append(weight)
        conv = sum(convs)
        output = F.conv2d(x, weight=conv, stride=self.stride, padding=self.padding, groups=self.groups)
        return None

def test():
    x = torch.randn(1, 16, 32, 32)
    conv = CondConv_Inf(16, 64, 3, padding=1)
    y = conv(x)
    print(y.shape)
    conv = CondConv_Inf(16, 64, 3, padding=1)
    y = conv(x)
    print(y.shape)

# test()