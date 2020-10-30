import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['RouterConv']

class route_func(nn.Module):
    def __init__(self, in_channels, num_experts, reduction=16, pool_size=4):
        super().__init__()
        reduction_channels = max(in_channels // reduction, reduction)
        self.channel_extractor = nn.Sequential(
            nn.Linear(in_channels,reduction_channels,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels,in_channels,bias=False)
        )
        self.adp_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.glo_pool = nn.AdaptiveAvgPool2d(1)
        self.expert_extractor = nn.Linear(pool_size*pool_size, num_experts,bias=False)

    def forward(self, x):
        gap = self.glo_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        # [n, in_channels]
        channel_weight = self.channel_extractor(gap)
        # [n, pool_size*pool_size]
        channel_pool = (self.adp_pool(x)).mean(dim=1,keepdim=False).view(x.shape[0],-1)
        expert_weight = self.expert_extractor(channel_pool)
        # [n, in_channel, experts]
        general = channel_weight.unsqueeze(-1)*expert_weight.unsqueeze(1)
        general = torch.sigmoid(general)

        return general

class RouterConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, num_experts=3, reduction=16, spatial=4):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.groups = groups

        # routing function
        self.routing_func = route_func(in_channels, num_experts, reduction, spatial)

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

class route_func_dw(nn.Module):
    def __init__(self, in_channels, num_experts, groups, reduction=16, pool_size=4):
        super().__init__()
        reduction_channels = max(in_channels // reduction, reduction)
        self.channel_extractor = nn.Sequential(
            nn.Linear(in_channels,reduction_channels,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels,in_channels,bias=False)
        )
        self.adp_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.glo_pool = nn.AdaptiveAvgPool2d(1)
        self.expert_extractor = nn.Linear(pool_size*pool_size, num_experts,bias=False)

    def forward(self, x):
        gap = self.glo_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        # [n, in_channels]
        channel_weight = self.channel_extractor(gap)
        # [n, pool_size*pool_size]
        channel_pool = (self.adp_pool(x)).mean(dim=1,keepdim=False).view(x.shape[0],-1)
        expert_weight = self.expert_extractor(channel_pool)
        # [n, in_channel, experts]
        general = channel_weight.unsqueeze(-1)*expert_weight.unsqueeze(1)
        general = torch.sigmoid(general)

        return general

def demo():
    net = RouterConv(3, 16, 3)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.shape)
# demo()
