import torch
import argparse
from flops_counter import get_model_complexity_info
from utils import get_network

parser = argparse.ArgumentParser(description='Counting network\'s pararmeters')
parser.add_argument('--network', '-n', required=True)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--input-size', '-i', type=int, default=32)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu')
net = get_network(args.network, device, args.input_size, args.nclass)
flops, params = get_model_complexity_info(net, (3, args.input_size, args.input_size), as_strings=True, print_per_layer_stat=True)
print('Flops: {}'.format(flops))