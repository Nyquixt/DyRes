import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from datetime import timedelta

from models import *
from config import *
from utils import calculate_acc, get_network, get_dataloader, init_params, count_parameters

parser = argparse.ArgumentParser(description='Training CNN models on CIFAR')

parser.add_argument('--network', '-n', choices=['resnet_ac20', 'resnet_ac56', 'resnet_ac110', 'resnet_ac164'], required=True)
parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch', '-b', type=int, default=256, help='The batch size')
parser.add_argument('--lr', '-l', type=float, default=0.1, help='Learning rate')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--update', '-u', type=int, default=50, help='Print out stats after x batches')
parser.add_argument('--weight-decay', '-d', type=float, default=0.0005, help='Weight decay for SGD optimizer')
parser.add_argument('--step-size', '-s', type=int, default=50, help='Step in learning rate scheduler')
parser.add_argument('--gamma', '-g', type=float, default=0.2, help='Gamma in learning rate scheduler')
parser.add_argument('--nclass', choices=[10, 100], type=int, help='CIFAR10 or CIFAR100', default=10)
parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()
print(args)

# Device
device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu')

# Dataloader
trainloader, testloader = get_dataloader(args.nclass, args.batch)

# Define losses lists to plot
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

# Define model
net = get_network(args.network, device, args.nclass)
init_params(net)

print('Training {} with {} parameters...'.format(args.network, count_parameters(net)))

net.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

# Train the model
start = time.time()
for epoch in range(args.epoch):  # loop over the dataset multiple times

    training_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        # Print statistics
        
        if i % args.update == (args.update - 1):    # print every args.update mini-batches

            with torch.no_grad():
                validation_loss = 0.0
                for j, data in enumerate(testloader): # (10,000 / args.batch) batches
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    validation_loss += loss.item()
            
            train_losses.append(training_loss / args.update)
            val_losses.append(validation_loss / (10000/args.batch))

            train_acc = calculate_acc(trainloader, net, device)
            net.eval()
            val_acc = calculate_acc(testloader, net, device)
            net.train()

            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)

            print('[Epoch: %d, Batch: %5d] Train Loss: %.3f    Train Acc: %.3f%%    Val Loss: %.3f    Val Acc: %.3f%%' %
                  ( epoch + 1, i + 1, training_loss / args.update, train_acc, validation_loss / (10000/args.batch), val_acc ))
            
            training_loss = 0.0

    # Step the scheduler after every epoch
    scheduler.step()

end = time.time()
print('Total time trained: {}'.format( str(timedelta(seconds=int(end - start)) ) ))

# Test the model
net.eval()
val_acc = calculate_acc(testloader, net, device)
print('Test Accuracy of the network on the 10000 test images: {} %'.format(val_acc))

# Save the model
torch.save(net.state_dict(), 'trained_nets/{}-cifar{}-b{}-e{}-{}.pth'.format(args.network, args.nclass, args.batch, args.epoch, int(round(time.time() * 1000))))

# Save plot
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

plt.savefig('plots/{}-losses-cifar{}-b{}-e{}-{}.png'.format(args.network, args.nclass, args.batch, args.epoch, int(round(time.time() * 1000))))
