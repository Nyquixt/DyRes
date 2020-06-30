import torch
import argparse
from utils import count_parameters, get_network

parser = argparse.ArgumentParser(description='Counting network\'s pararmeters')
parser.add_argument('--network', '-n', required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu')
print(count_parameters(get_network(args.network, device)))