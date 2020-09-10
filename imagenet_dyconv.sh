#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 nohup python3 train.py -n dy_alexnet --dataset imagenet --ngpu 2 --save --cuda &
CUDA_VISIBLE_DEVICES=2,3,4 nohup python3 train.py -n dy_resnet18 --dataset imagenet --ngpu 3 --save --cuda &
CUDA_VISIBLE_DEVICES=5,6,7 nohup python3 train.py -n dy_mobilenetv2 --dataset imagenet --ngpu 3 --save --cuda &