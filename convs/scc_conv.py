import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['SCConv']

class route_func(nn.Module):

    def __init__(self, in_channels, num_experts, spatial=4, reduction=16, activation='sigmoid'):
        super().__init__()

        self.reduction_channels = max(in_channels // reduction, reduction)
        self.num_experts = num_experts
        self.spatial = spatial

        self.avgpool = nn.AdaptiveAvgPool2d(spatial)
        self.conv = nn.Conv2d(in_channels, self.reduction_channels, 1)
        self.fc = nn.Linear(self.reduction_channels * spatial * spatial, num_experts)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(2)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.avgpool(x)
        x = F.relu(self.conv(x).view(b, self.reduction_channels * self.spatial * self.spatial))
        x = F.relu(self.fc(x))
        x = x.view(b, self.num_experts) # N x k
        x = self.activation(x)
        return x

class SCConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, num_experts=3, spatial=4, reduction=16, activation='sigmoid'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_experts = num_experts

        # routing function
        self.routing_func = route_func(in_channels, num_experts, spatial, reduction, activation)

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        routing_weight = self.routing_func(x)
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size() # k is num_experts
        x = x.view(1, -1, h, w) # 1 x N*C_in x H x W
        weight = self.weight.view(k, -1) # k x C_out*C_in*kH*hW
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(x, weight=combined_weight, bias=combined_bias, 
                            stride=self.stride, padding=self.padding, groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=combined_weight, bias=None, 
                            stride=self.stride, padding=self.padding, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output

def test():
    x = torch.randn(4, 16 , 32, 32)
    conv = SCConv(x.size(1), 64, 3)
    y = conv(x)
    print(y.size())

# test()