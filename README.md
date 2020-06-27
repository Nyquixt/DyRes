# DyConv

Results on the CIFAR10 Dataset

| Models        | Basic         | ACNet         | CondConv      | DynamicConv   |
|---------------|---------------|---------------|---------------|---------------|
| AlexNet       | 86.26%        | -----         | 86.20%        | 86.89%        |
| ResNet18      | 94.12%        | 94.00%        | -----         | -----         |
| ResNet34      | 94.25%        | 93.81%        | -----         | -----         |
| ResNet50      | -----         | -----         | -----         | -----         |
| ResNet101     | -----         | -----         | -----         | -----         |
| ResNet152     | -----         | -----         | -----         | -----         |

Results on the CIFAR100 Dataset

| Models        | Basic         | ACNet         | CondConv      | DynamicConv   |
|---------------|---------------|---------------|---------------|---------------|
| ResNet18      | -----         | -----         | -----         | -----         |
| ResNet34      | -----         | -----         | -----         | -----         |
| ResNet50      | -----         | -----         | -----         | -----         |
| ResNet101     | -----         | -----         | -----         | -----         |
| ResNet152     | -----         | -----         | -----         | -----         |

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
