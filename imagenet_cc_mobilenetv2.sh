#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python3 train.py -n cc_mobilenetv2 --dataset imagenet --ngpu 8 --save --cuda &