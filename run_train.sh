#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际可用的GPU数量调整

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py