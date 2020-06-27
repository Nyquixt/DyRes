import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ['DynamicConv']

class DynamicConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_experts, kernel_size, 
                    stride=1, padding=0, bias=False):
        super(DynamicConv, self).__init__()

        self.stride = stride
        self.padding = padding

        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        self.conv1_weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.conv2_weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.conv3_weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        alphas = F.adaptive_avg_pool2d(x, 1)
        alphas = alphas.view(alphas.size(0), -1)
        alphas = self.attention(alphas)
        print(alphas)

        a = (torch.sum(alphas, 0) / x.size(0)).detach()

        W1 = a[0] * self.conv1_weight
        W2 = a[1] * self.conv2_weight
        W3 = a[2] * self.conv3_weight

        out = F.conv2d(x, W1, bias=None, stride=self.stride, padding=self.padding) \
            + F.conv2d(x, W2, bias=None, stride=self.stride, padding=self.padding) \
            + F.conv2d(x, W3, bias=None, stride=self.stride, padding=self.padding)
    
        out = self.bn(out)
        out = self.relu(out)

        return out

def test():
    net = DynamicConv(in_channels=4, out_channels=8, num_experts=3, kernel_size=3, stride=1, padding=1, bias=False)
    x = torch.randn(4, 4, 32, 32)
    y = net(x)
    print(y.size())

# test()