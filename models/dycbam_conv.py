import torch
import torch.nn as nn 

__all__ = ['DyCBAMConv', 'DyCBAMConv_2', 'DyCBAMConv_3', 'DyCBAMConv_4']

'''
[avg, max]
softmax(avg) + softmax(max)
'''
class DyCBAMConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DyCBAMConv, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        gap = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        mp = self.max_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        weights_avg = self.attention(gap).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        weights_max = self.attention(mp).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        weights = weights_avg + weights_max
        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        return out

'''
[avg, max]
softmax(avg + max)
'''
class DyCBAMConv_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DyCBAMConv_2, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        self.softmax = nn.Softmax(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        gap = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        mp = self.max_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        weights_avg = self.attention(gap)
        weights_max = self.attention(mp)
        weights = self.softmax(weights_avg + weights_max).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        return out


'''
[avg, std]
softmax(avg + std)
'''
class DyCBAMConv_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DyCBAMConv_3, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        gap = x.view(x.size(0), x.size(1), -1).mean(-1)
        std = x.view(x.size(0), x.size(1), -1).std(-1) # This way will solve the NaN problem with torch.std()
        weights_avg = self.attention(gap)
        weights_std = self.attention(std)
        weights = self.softmax(weights_avg + weights_std).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        return out

'''
[avg, std]
softmax(avg + max) & channel_attention
'''
class DyCBAMConv_4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DyCBAMConv_4, self).__init__()

        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.two_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.three_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

        if in_channels < 16:
            self.channel_attention = nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.ReLU(inplace=True),
                nn.Linear(1, in_channels),
                nn.Sigmoid()
            )
        else:
            self.channel_attention = nn.Sequential(
                nn.Linear(in_channels, in_channels // 16),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // 16, in_channels),
                nn.Sigmoid()
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        gap = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        mp = self.max_pool(x).squeeze(dim=-1).squeeze(dim=-1)
        channel = gap * mp
        channel = self.channel_attention(channel)

        x = x * channel.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(x)

        one_out = self.one_conv(x).unsqueeze(dim=1)
        two_out = self.two_conv(x).unsqueeze(dim=1)
        three_out = self.three_conv(x).unsqueeze(dim=1)
        all_out = torch.cat([one_out, two_out, three_out], dim=1)
        
        weights_avg = self.attention(gap)
        weights_max = self.attention(mp)
        weights = self.softmax(weights_avg + weights_max).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        
        return out

def test():
    x = torch.randn(4, 3, 32, 32)
    conv = DyCBAMConv_4(x.size(1), 64, 3, padding=1)
    y = conv(x)
    print(y.size())

# test()