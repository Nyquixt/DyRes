import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CondConv']

class CondConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, num_experts=3):
        super(CondConv, self).__init__()
        self.num_experts = num_experts
        self.convs = nn.Parameter(torch.rand(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))

        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.attention = nn.Sequential(
            nn.Linear(in_channels, num_experts),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = []
        for sample in x:
            sample = sample.unsqueeze(dim=0)
            gap = sample.mean(dim=-1).mean(dim=-1)
            routing = self.attention(gap).view(self.num_experts) # 1 x 3
            convs = torch.sum(routing.view(self.num_experts, 1, 1, 1, 1) * self.convs, dim=0)
            out = F.conv2d(sample, convs, stride=self.stride, padding=self.padding, groups=self.groups)
            res.append(out)
            
        return torch.cat(res, dim=0)

def test():
    x = torch.randn(4, 16 , 32, 32)
    conv = CondConv(x.size(1), 64, 3)
    y = conv(x)
    print(y.size())

# test()