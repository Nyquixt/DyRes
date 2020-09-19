import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['MSConv']

class route_func(nn.Module):

    def __init__(self, in_channels, num_experts, reduction=16):
        super().__init__()
        reduction_channels = max (in_channels // reduction, reduction)
        self.scale_1 = nn.AdaptiveAvgPool2d(1)

        self.scale_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(in_channels, in_channels, 3, stride=1),
        )

        self.scale_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            nn.Conv2d(in_channels, reduction_channels, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, in_channels, 3, stride=1),
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(3 * in_channels, num_experts, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale_1 = self.scale_1(x)
        scale_3 = self.scale_3(x)
        scale_5 = self.scale_5(x)
        scale = F.relu(torch.cat([scale_1, scale_3, scale_5], dim=1))
        attention = self.attention(scale).squeeze(dim=-1).squeeze(dim=-1)
        return attention

class MSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, num_experts=3):
        super(MSConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_experts = num_experts

        # routing function
        self.routing_func = route_func(in_channels, num_experts)

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
        k, c_out, c_in, kh, kw = self.weight.size()
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
    x = torch.randn(4, 3 , 32, 32)
    conv1 = MSConv(x.size(1), 64, 3, padding=1)
    y1 = conv1(x)
    print(y1.size())

# test()