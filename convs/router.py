import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Router(nn.Module):
    def __init__(self, in_channels, num_experts, reduction=16, pool_size =4):
        super(Router, self).__init__()
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
    net = Router(3,3)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.shape)
demo()
