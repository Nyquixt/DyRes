import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['SDYCConv', 'SDYCConv_DW']

# TODO: if use bias, out_channels is used in route_func instead

class route_func(nn.Module):
    def __init__(self, in_channels, num_experts, spatial=4, reduction=16, activation='sigmoid'):
        super().__init__()

        self.reduction_channels = max(in_channels // reduction, reduction)
        self.num_experts = num_experts
        self.spatial = spatial

        self.avgpool = nn.AdaptiveAvgPool2d(spatial)
        self.conv = nn.Conv2d(in_channels, self.reduction_channels)
        self.fc = nn.Linear(self.reduction_channels * spatial * spatial, num_experts * in_channels)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(2)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.avgpool(x)
        x = F.relu(self.conv(x).view(b, self.reduction_channels * self.spatial * self.spatial))
        x = F.relu(self.fc(x))
        x = x.view(b, self.num_experts, -1) # N x k x C_in
        x = self.activation(x)
        return x

class DyChannel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, num_experts=3, spatial=4, reduction=16, activation='sigmoid'):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, num_experts, spatial, reduction, activation)

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        
        self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k x C_in
        routing_weight = routing_weight.unsqueeze(dim=2).unsqueeze(dim=-1).unsqueeze(dim=-1) # N x k x 1 x C_in x 1 x 1

        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.view(1, -1, h, w) # 1 x N*C_in x H x W
        weight = self.weight.unsqueeze(dim=0) # 1 x k x C_out x C_in x kH x hW 

        combined_weight = (routing_weight * weight).sum(1).view(-1, c_in, kh, kw)
        output = F.conv2d(x, weight=combined_weight, bias=None, 
                            stride=self.stride, padding=self.padding, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output

'''
    Depthwise Dynamic Convolution
'''

class route_func_dw(nn.Module):
        def __init__(self, channels, num_experts, groups, spatial=4, reduction=16, activation='sigmoid'):
            super().__init__()

            self.reduction_channels = max(channels // reduction, reduction)
            self.num_experts = num_experts
            self.spatial = spatial

            self.avgpool = nn.AdaptiveAvgPool2d(spatial)
            self.conv = nn.Conv2d(channels, self.reduction_channels)
            self.fc = nn.Linear(self.reduction_channels * spatial * spatial, num_experts * channels)
            if activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            else:
                self.activation = nn.Softmax(2)

        def forward(self, x):
            b, c, _, _ = x.size()
            x = self.avgpool(x)
            x = F.relu(self.conv(x).view(b, self.reduction_channels * self.spatial * self.spatial))
            x = F.relu(self.fc(x))
            x = x.view(b, self.num_experts, -1) # N x k x C_in
            x = self.activation(x)
            return x

class DyChannel_DW(nn.Module): # depthwise, use for MobileNetV2

    def __init__(self, channels, kernel_size, stride=1, padding=0, groups=1, num_experts=3, spatial=4, reduction=16, activation='sigmoid'):
        super().__init__()
    
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func_dw(channels, num_experts, spatial, groups, reduction, activation)

        self.weight = nn.Parameter(torch.Tensor(num_experts, channels, channels // groups, kernel_size, kernel_size))
        
        self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        routing_weight = self.routing_func(x) # N x k x C
        routing_weight = routing_weight.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) # N x k x C x 1 x 1 x 1

        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.view(1, -1, h, w) # 1 x N*C_in x H x W
        weight = self.weight.unsqueeze(dim=0) # 1 x k x C x C x kH x hW 
        combined_weight = (routing_weight * weight).sum(1).view(-1, c_in, kh, kw)

        output = F.conv2d(x, weight=combined_weight, bias=None, 
                            stride=self.stride, padding=self.padding, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output

def test():
    x = torch.randn(4, 16 , 32, 32)
    conv = DyChannel(x.size(1), 64, 3, padding=1, activation='softmax', num_experts=5)
    y = conv(x)
    print(y.size())

    conv = DyChannel_DW(x.size(1), 3, padding=1, activation='softmax', num_experts=5)
    y = conv(x)
    print(y.size())

# test()