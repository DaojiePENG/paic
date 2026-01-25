#!/bin/bash
set -e

# 检查显卡数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $GPU_COUNT -ne 4 ]; then
    echo "警告：检测到${GPU_COUNT}张显卡，预期4张4090！"
fi

# 用accelerate启动多卡服务
accelerate launch --config_file accelerate_config.yaml server.py