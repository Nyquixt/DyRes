# DyConv

Results on the CIFAR10 Dataset

| Models        | Basic         | CondConv      | DyConv        | WeightNet     | DyRes A       | DyRes B       | DyRes C       | 
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | 86.26%        | 86.20%        | 86.89%        | 86.42%        | 86.94%        | 87.21%        | ------        |
| ResNet18      | 94.12%        | ------        | 94.24%        | ------        | 94.16%        | ------        | ------        |
| MobileNetV2   | 92.99%        | ------        | 93.52%        | ------        | ------        | ------        | ------        |

Results on the CIFAR100 Dataset

| Models        | Basic         | CondConv      | DyConv        | WeightNet     | DyRes2 A      | DyRes B       | DyRes C       |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | 61.01%        | 61.47%        | 60.60%        | ------        | 61.70%        | ------        | ------        |
| ResNet18      | ------        | ------        | ------        | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        | ------        | ------        | ------        |

Results on the SVHN Dataset

| Models        | Basic         | CondConv      | DyConv        | WeightNet     | DyRes A       | DyRes B       | DyRes C       |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | 94.50%        | 94.65%        | 94.55%        | ------        | 94.79%        | 94.82%        | ------        |
| ResNet18      | 96.09%        | 96.41%        | 96.34%        | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        | ------        | ------        | ------        |

Results on the Tiny ImageNet Dataset

| Models        | Basic         | CondConv      | DyConv        | WeightNet     | DyRes A       | DyRes B       | DyRes C       |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | ------        | ------        | ------        | ------        | ------        | ------        | ------        |
| ResNet18      | 63.70%        | ------        | ------        | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        | ------        | ------        | ------        |

Results on the Downsampled ImageNet Dataset

| Models        | Basic         | CondConv      | DyConv        | WeightNet     | DyRes A       | DyRes B       | DyRes C       |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | ------        | ------        | ------        | ------        | ------        | ------        | ------        |
| ResNet18      | ------        | ------        | ------        | ------        | ------        | ------        | ------        |
| MobileNetV2   | ------        | ------        | ------        | ------        | ------        | ------        | ------        |

### Training Configurations

| Parameter                     | Value         |
|------------------------------ |---------------|
| epochs                        | 120           |
| batch                         | 256           |
| learning rate                 | 0.1           |
| update learning rate every    | 30 epochs     |
| learning rate update factor   | 0.1           |
| SGD momentum                  | 0.9           |
| SGD weight decay              | 5e-4          |

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
