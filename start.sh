#!/bin/bash
set -e

# 设置Hugging Face缓存目录
export HF_HOME="/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"

# 检查显卡数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $GPU_COUNT -ne 4 ]; then
    echo "警告：检测到${GPU_COUNT}张显卡，预期4张4090！"
fi

# 直接使用python运行，多卡服务由脚本内的模型管理
python server.py