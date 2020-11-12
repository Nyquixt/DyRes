from convs.condconv import *
from convs.cc_inf import *
from convs.dyconv import *
from convs.dyres_conv import *
from convs.dyres_inf import *
from convs.ddsnet import *
from convs.dds_exp import *
from convs.dychannel import *
from flops_counter import get_model_complexity_info

import torch

x = torch.randn(1, 16, 32, 32)
net = CondConv(x.size(1), x.size(1), 3, num_experts=4)
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('--CondConv\nFlops: {}\nParams: {}'.format(flops, params))

net = DyConv(x.size(1), x.size(1), 3, num_experts=4)
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('--DyConv\nFlops: {}\nParams: {}'.format(flops, params))

net = DyResConv_Inf(x.size(1), x.size(1), 3, num_experts=4, mode='A')
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('--DyResA\nFlops: {}\nParams: {}'.format(flops, params))

net = DyResConv_Inf(x.size(1), x.size(1), 3, num_experts=4, mode='B')
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('--DyResB\nFlops: {}\nParams: {}'.format(flops, params))

# net = DDSConv(x.size(1), x.size(1), 3, num_experts=4)
# flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
# print('Flops: {}\nParams: {}'.format(flops, params))

net = DDSConv_Exp(x.size(1), x.size(1), 3, num_experts=4)
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('--DDS\nFlops: {}\nParams: {}'.format(flops, params))