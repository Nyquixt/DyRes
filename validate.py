import torch

import argparse
from utils import get_dataloader, get_network, accuracy

parser = argparse.ArgumentParser(description='Validating CNN models')

parser.add_argument('--network', '-n', required=True)
parser.add_argument('--path-to-network', '-p', type=str)
parser.add_argument('--dataset', type=str, help='cifar10 or cifar100 or svhn or tinyimagenet', default='cifar10')
parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()
print(args)

device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

if args.dataset == 'cifar10':
    net = get_network(args.network, device, input_size=32, num_classes=10)
    _, testloader = get_dataloader(args.dataset, 10000)
elif args.dataset == 'svhn':
    net = get_network(args.network, device, input_size=32, num_classes=10)
    _, testloader = get_dataloader(args.dataset, 26032)
elif args.dataset == 'cifar100':
    net = get_network(args.network, device, input_size=32, num_classes=100)
    _, testloader = get_dataloader(args.dataset, 10000)
elif args.dataset == 'tinyimagenet':
    net = get_network(args.network, device, input_size=64, num_classes=200)
    _, testloader = get_dataloader(args.dataset, 10000)

inputs, labels = next(iter(testloader))

net.load_state_dict(torch.load(args.path_to_network))
net.eval()
result = accuracy(net(inputs.to(device)), labels.to(device), (1, 5))
print('Top1: {}%\tTop5: {}%'.format(round(float(result[0][0].data), 2), round(float(result[1][0].data), 2)))