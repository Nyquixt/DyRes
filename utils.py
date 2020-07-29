from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
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

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

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

def get_network(network, device, input_size=32, num_classes=10):

    if network == 'ac_resnet18':
        from models.ac_resnet import AC_ResNet18
        net = AC_ResNet18(num_classes)
    elif network == 'cc_resnet18':
        from models.cc_resnet import CC_ResNet18
        net = CC_ResNet18(num_classes)
    elif network == 'dy_resnet18':
        from models.dy_resnet import Dy_ResNet18
        net = Dy_ResNet18(num_classes)
    elif network == 'dyres_resnet18':
        from models.dyres_resnet import DyRes_ResNet18
        net = DyRes_ResNet18(num_classes)
    elif network == 'wn_resnet18':
        from models.wn_resnet import WN_ResNet18
        net = WN_ResNet18(num_classes)
    elif network == 'resnet18':
        from models.resnet import ResNet18
        net = ResNet18(num_classes)
    elif network == 'ac_resnet34':
        from models.ac_resnet import AC_ResNet34
        net = AC_ResNet34(num_classes)
    elif network == 'cc_resnet34':
        from models.cc_resnet import CC_ResNet34
        net = CC_ResNet34(num_classes)
    elif network == 'dy_resnet34':
        from models.dy_resnet import Dy_ResNet34
        net = Dy_ResNet34(num_classes)
    elif network == 'resnet34':
        from models.resnet import ResNet34
        net = ResNet34(num_classes)
    elif network == 'ac_resnet50':
        from models.ac_resnet import AC_ResNet50
        net = AC_ResNet50(num_classes)
    elif network == 'cc_resnet50':
        from models.cc_resnet import CC_ResNet50
        net = CC_ResNet50(num_classes)
    elif network == 'dy_resnet50':
        from models.dy_resnet import Dy_ResNet50
        net = Dy_ResNet50(num_classes)
    elif network == 'resnet50':
        from models.resnet import ResNet50
        net = ResNet50(num_classes)
    elif network == 'ac_resnet101':
        from models.ac_resnet import AC_ResNet101
        net = AC_ResNet101(num_classes)
    elif network == 'cc_resnet101':
        from models.cc_resnet import CC_ResNet101
        net = CC_ResNet101(num_classes)
    elif network == 'dy_resnet101':
        from models.dy_resnet import Dy_ResNet101
        net = Dy_ResNet101(num_classes)
    elif network == 'resnet101':
        from models.resnet import ResNet101
        net = ResNet101(num_classes)
    elif network == 'ac_resnet152':
        from models.ac_resnet import AC_ResNet152
        net = AC_ResNet152(num_classes)
    elif network == 'cc_resnet152':
        from models.cc_resnet import CC_ResNet152
        net = CC_ResNet152(num_classes)
    elif network == 'dy_resnet152':
        from models.dy_resnet import Dy_ResNet152
        net = Dy_ResNet152(num_classes)
    elif network == 'resnet152':
        from models.resnet import ResNet152
        net = ResNet152(num_classes)
    elif network == 'alexnet':
        from models.alexnet import AlexNet
        net = AlexNet(num_classes, input_size)
    elif network == 'ac_alexnet':
        from models.ac_alexnet import AC_AlexNet
        net = AC_AlexNet(num_classes, input_size)
    elif network == 'cc_alexnet':
        from models.cc_alexnet import CC_AlexNet
        net = CC_AlexNet(num_classes, input_size)
    elif network == 'dy_alexnet':
        from models.dy_alexnet import Dy_AlexNet
        net = Dy_AlexNet(num_classes, input_size)
    elif network == 'dysq_alexnet':
        from models.dysq_alexnet import DySQ_AlexNet
        net = DySQ_AlexNet(num_classes, input_size)
    elif network == 'dyres_alexnet1':
        from models.dyres_alexnet import DyRes_AlexNet1
        net = DyRes_AlexNet1(num_classes, input_size)
    elif network == 'dyres_alexnet2':
        from models.dyres_alexnet import DyRes_AlexNet2
        net = DyRes_AlexNet2(num_classes, input_size)
    elif network == 'wn_alexnet':
        from models.wn_alexnet import WN_AlexNet
        net = WN_AlexNet(num_classes, input_size)
    elif network == 'ac_squeezenet':
        from models.ac_squeezenet import AC_SqueezeNet
        net = AC_SqueezeNet(num_classes)
    elif network == 'cc_squeezenet':
        from models.cc_squeezenet import CC_SqueezeNet
        net = CC_SqueezeNet(num_classes)
    elif network == 'dy_squeezenet':
        from models.dy_squeezenet import Dy_SqueezeNet
        net = Dy_SqueezeNet(num_classes)
    elif network == 'dyres_squeezenet':
        from models.dyres_squeezenet import DyRes_SqueezeNet
        net = DyRes_SqueezeNet(num_classes)
    elif network == 'squeezenet':
        from models.squeezenet import SqueezeNet
        net = SqueezeNet(num_classes)
    elif network == 'ac_mobilenetv2':
        from models.ac_mobilenetv2 import AC_MobileNetV2
        net = AC_MobileNetV2(num_classes)
    elif network == 'cc_mobilenetv2':
        from models.cc_mobilenetv2 import CC_MobileNetV2
        net = CC_MobileNetV2(num_classes)
    elif network == 'dy_mobilenetv2':
        from models.dy_mobilenetv2 import Dy_MobileNetV2
        net = Dy_MobileNetV2(num_classes)
    elif network == 'dyres_mobilenetv2':
        from models.dyres_mobilenetv2 import DyRes_MobileNetV2
        net = DyRes_MobileNetV2(num_classes)
    elif network == 'mobilenetv2':
        from models.mobilenetv2 import MobileNetV2
        net = MobileNetV2(num_classes)
    else:
        print('the network is not supported')
        sys.exit()
    
    net = net.to(device)

    return net

def get_dataloader(dataset, batch_size):
    if dataset == 'cifar10':
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
    
    elif dataset == 'cifar100':
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

    elif dataset == 'svhn':
        train_transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD)
        ])

        test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD)
        ])

        trainset = torchvision.datasets.SVHN(root=DATA_ROOT, split='train', transform=train_transform, download=True)
        testset = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', transform=test_transform, download=True)

    elif dataset == 'tinyimagenet':
        train_transform = transforms.Compose(
            [transforms.RandomCrop(size=64, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
        ])

        test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
        ])

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(TINY_IMAGENET_DATA_DIR, 'train'), transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(TINY_IMAGENET_DATA_DIR, 'validation'), transform=test_transform)
    
    else:
        print('Dataset not supported yet...')
        sys.exit()

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

    plt.savefig('plots/{}-losses-{}-b{}-e{}-{}.png'.format(args.network, args.dataset, args.batch, args.epoch, time_stamp))