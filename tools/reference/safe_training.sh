#!/bin/bash

# 安全的多卡分布式训练脚本
# 此脚本包含了针对checkpoint加载问题的修复

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练配置
CONFIG_FILE="configs/deimv2/deimv2_dinov3_x_custom.yml"
NUM_GPUS=4

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    train.py \
    --config ${CONFIG_FILE} \
    --seed 0

# 说明:
# 1. 修复了多卡训练时的checkpoint加载竞争条件问题
# 2. 在load_resume_state和load_tuning_state中添加了分布式barrier
# 3. 确保主进程保存完checkpoint后,其他进程才开始加载
# 4. 添加了文件存在性和完整性检查
# 5. 改进了错误处理和日志输出
