#!/bin/bash

################################ CIFAR ###################################
datasets=CIFAR100
device=0,1 # use two GPUs for parallel computing
model=resnet18 #  resnet18 VGG16BN WideResNet16x8 WideResNet28x10
schedule=cosine
wd=0.001
epoch=200
bz=256
lr=0.10
port=1234
seed=42
alpha=0.5
gamma=0.01

DST=results/rwp_ddp_cutout_gamma$gamma\_alpha$alpha\_$epoch\_$bz\_$lr\_$model\_$wd\_$datasets\_$schedule\_seed$seed
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node 2 --master_port $port train_nwp_parallel.py --datasets $datasets \
        --arch=$model --epochs=$epoch --wd=$wd --randomseed $seed --lr $lr --gamma $gamma --cutout -b $bz --alpha $alpha --workers 8 \
        --save-dir=$DST/checkpoints --log-dir=$DST -p 100 --schedule $schedule

