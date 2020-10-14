import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channels, r):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace = True),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
def test():
    x = torch.rand(1, 128, 256, 256)
    net = SELayer(128, 8)
    output = net(x)
    print(output.shape)

# test()