#!/bin/bash

datasets=ImageNet
device=0,1,2,3 # doubles GPUs for parallel computing of RWP

model=resnet18
path=... # dir for ImageNet datasets
DST=save_resnet18
CUDA_VISIBLE_DEVICES=$device  python3 train_nwp_imagenet.py -a $model \
    --epochs 90 --workers 16  --dist-url 'tcp://127.0.0.1:4234' --lr 0.1 -b 256 \
    --dist-backend 'nccl' --multiprocessing-distributed --gamma 0.005 --alpha 0.5 \
    --save-dir=$DST/checkpoints --log-dir=$DST \
    --world-size 1 --rank 0 $path
