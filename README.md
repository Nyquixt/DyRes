# DyConv

Results on the CIFAR10 Dataset

| Models        | Basic         | ACNet         | CondConv      | DyConv        | DyConv no AL  |
|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | 86.26%        | 87.77%        | 86.20%        | 86.89%        | 86.89%        |
| SqueezeNet    | 90.09%        | 91.90%        | 90.14%        | 90.75%        | 91.39%        |
| ResNet18      | 94.12%        | 94.00%        | ------        | 94.24%        | ------        |
| MobileNetV2   | 92.99%        | 93.90%        | ------        | 93.52%        | ------        |

Results on the CIFAR100 Dataset

| Models        | Basic         | ACNet         | CondConv      | DyConv        | DyConv no AL  |
|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | ------        | 62.39%        | 61.47%        | 60.60%        | 61.30%        |
| SqueezeNet    | 66.82%        | 69.67%        | 67.93%        | 68.18%        | 69.86%        |
| ResNet18      | ------        | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        | ------        |

Results on the SVHN Dataset

| Models        | Basic         | ACNet         | CondConv      | DyConv        |
|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | ------        | ------        | ------        | ------        |
| SqueezeNet    | ------        | ------        | ------        | ------        |
| ResNet18      | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        |

Results on the Downsample ImageNet Dataset

| Models        | Basic         | ACNet         | CondConv      | DyConv        |
|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | ------        | ------        | ------        | ------        |
| SqueezeNet    | ------        | ------        | ------        | ------        |
| ResNet18      | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        |

Results on the Tiny ImageNet Dataset

| Models        | Basic         | ACNet         | CondConv      | DyConv        |
|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | ------        | ------        | ------        | ------        |
| SqueezeNet    | ------        | ------        | ------        | ------        |
| ResNet18      | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        |

### How To Set Up Python and Pip

https://www.python.org/downloads/

### How To Set Up the Environment

To install the necessary Python packages for training

    pip3 install -r requirements.txt

### How To Run the Training

For simplicity, just run

    python3 train.py --network some_defined_network

If you want to play around with the hyper-parameters run ``python3 train.py -h`` to see the program's ``flags`` or ``arguments``.

    --network               Some predefined network architecture
    
    -e, --epoch             Number of epochs for training
    -b, --batch             Batch size
    -l, --lr                Learning rate for SGD
    -m, --momentum          Momentum for SGD
    -d, --weight-decay      Weight decay for SGD
    -u, --update            Print out the training stats every x number of batches
    -s, --step-size         Update the learning rate every x epochs
    -g, --gamma             Learning rate update factor. new_lr = old_lr * gamma
    
    --nclass                Number of classes, 10 or 100 for CIFAR10 or CIFAR100
    --cuda                  Use GPU to train if the flag is used

Another example to run

    python3 train.py --network resnet_ac50 -e 300 -b 512 -l 0.1 -m 0.9 -d 0.0005 -u 30 -s 80 -g 0.1 --nclass 100 --cuda
