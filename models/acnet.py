import torch
import torch.nn as nn

__all__ = ['AC_Conv2dBN', 'AC_Conv2dBNReLU']

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, x):
        print("x in crop layer size: {}".format(x.size()))
        if self.rows_to_crop == 0:
            return x[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        if self.cols_to_crop == 0:
            return x[:, :, self.rows_to_crop: -self.rows_to_crop, :]

        return x[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop: -self.cols_to_crop]

class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border) # (0, -1)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        else:
            square_outputs = self.square_conv(x)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())

            vertical_outputs = self.ver_conv_crop_layer(x)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())

            horizontal_outputs = self.hor_conv_crop_layer(x)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs

def AC_Conv2dBN(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=False)

def AC_Conv2dBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
    return nn.Sequential(
        ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=False),
        nn.ReLU(inplace=True)
    )

def test():
    x = torch.randn(32, 64, 32, 32)
    net1 = AC_Conv2dBN(x.size(1), 128, 3, padding=1)
    net2 = AC_Conv2dBNReLU(x.size(1), 128, 3, padding=1)

    y1 = net1(x)
    print('y1 shape: {}'.format(y1.size()))

    y2 = net2(x)
    print('y2 shape: {}'.format(y2.size()))

# test()