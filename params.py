from convs.condconv import *
from convs.dyconv import *
from convs.dyres_conv import *
from convs.ddsnet import *
from flops_counter import get_model_complexity_info

import torch

x = torch.randn(1, 64, 32, 32)
net = CondConv(x.size(1), x.size(1), 3, num_experts=4)
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('Flops: {}\n{}'.format(flops, params))

net = DyConv(x.size(1), x.size(1), 3, num_experts=4)
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('Flops: {}\n{}'.format(flops, params))

net = DyResConv(x.size(1), x.size(1), 3, num_experts=4, mode='A')
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('Flops: {}\n{}'.format(flops, params))

net = DyResConv(x.size(1), x.size(1), 3, num_experts=4, mode='B')
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('Flops: {}\n{}'.format(flops, params))

net = DDSConv(x.size(1), x.size(1), 3, num_experts=4)
flops, params = get_model_complexity_info(net, (x.size(1), 32, 32), as_strings=True, print_per_layer_stat=False)
print('Flops: {}\n{}'.format(flops, params))