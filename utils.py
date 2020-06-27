from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

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

    if network == 'resnet_ac18':
        from models.resnet_ac import ResNet_AC18
        net = ResNet_AC18(num_classes)
    elif network == 'resnet18':
        from models.resnet import ResNet18
        net = ResNet18(num_classes)
    elif network == 'resnet_ac34':
        from models.resnet_ac import ResNet_AC34
        net = ResNet_AC34(num_classes)
    elif network == 'resnet34':
        from models.resnet import ResNet34
        net = ResNet34(num_classes)
    elif network == 'resnet_ac50':
        from models.resnet_ac import ResNet_AC50
        net = ResNet_AC50(num_classes)
    elif network == 'resnet50':
        from models.resnet import ResNet50
        net = ResNet50(num_classes)
    elif network == 'resnet_ac101':
        from models.resnet_ac import ResNet_AC101
        net = ResNet_AC101(num_classes)
    elif network == 'resnet101':
        from models.resnet import ResNet101
        net = ResNet101(num_classes)
    elif network == 'resnet_ac152':
        from models.resnet_ac import ResNet_AC152
        net = ResNet_AC152(num_classes)
    elif network == 'resnet152':
        from models.resnet import ResNet152
        net = ResNet152(num_classes)
    elif network == 'alexnet':
        from models.alexnet import AlexNet
        net = AlexNet(num_classes)
    elif network == 'cc_alexnet':
        from models.cc_alexnet import CC_AlexNet
        net = CC_AlexNet(num_classes)
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

def save_plot(train_losses, train_accuracy, val_losses, val_accuracy, args, time_stamp):
    x = np.array([x for x in range(len(train_losses))]) * args.update
    y1 = np.array(train_losses)
    y2 = np.array(val_losses)

    y3 = np.array(train_accuracy)
    y4 = np.array(val_accuracy)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(x, y1, label='train loss')
    ax1.plot(x, y2, label='val loss')
    ax1.legend()
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel('losses')

    ax2.plot(x, y3, label='train acc')
    ax2.plot(x, y4, label='val acc')
    ax2.legend()
    ax2.set_xlabel('batches')
    ax2.set_ylabel('acc')

    plt.savefig('plots/{}-losses-cifar{}-b{}-e{}-{}.png'.format(args.network, args.nclass, args.batch, args.epoch, time_stamp))