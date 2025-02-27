#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际可用的GPU数量调整

# 设置PYTORCH内存分配配置以避免内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py