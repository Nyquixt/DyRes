from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sys

from config import *

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def count_parameters(net, all=True):
    # If all= Flase, we only return the trainable parameters; tested
    return sum(p.numel() for p in net.parameters() if p.requires_grad or all)

def calculate_acc(dataloader, net, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (correct / total) * 100

def get_network(network, device, num_classes=10):

    if network == 'resnet_ac20':
        from models.resnet_ac import ResNet_AC20
        net = ResNet_AC20(num_classes)
    elif network == 'resnet_ac56':
        from models.resnet_ac import ResNet_AC56
        net = ResNet_AC56(num_classes)
    elif network == 'resnet_ac110':
        from models.resnet_ac import ResNet_AC110
        net = ResNet_AC110(num_classes)
    elif network == 'resnet_ac164':
        from models.resnet_ac import ResNet_AC164
        net = ResNet_AC164(num_classes)
    else:
        print('the network is not supported')
        sys.exit()
    
    net = net.to(device)

    return net

def get_dataloader(num_classes, batch_size):
    if num_classes == 10:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ])

        trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, transform=train_transform, download=True)
        testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform, download=True)
    else:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
            ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
            ])

        trainset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, transform=train_transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, transform=test_transform, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader