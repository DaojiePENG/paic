#!/bin/bash
set -e

# 设置Hugging Face缓存目录
export HF_HOME="/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"

# 设定Python解释器路径
PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/paic/bin/python"

# 直接使用python运行，多卡服务由脚本内的模型管理
# 使用默认配置启动
$PYTHON server_tts.py
