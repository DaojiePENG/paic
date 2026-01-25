#!/bin/bash
set -e

# 设置Hugging Face缓存目录
export HF_HOME="/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"

# 直接使用python运行，多卡服务由脚本内的模型管理
# 使用默认配置启动
# python server.py

# 自定义模型名称和端口
# python server.py --model-name "Qwen/Qwen3-VL-2B-Instruct" --port 8080

# 自定义所有核心参数
python server.py \
  --model-name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --device-map "auto" \
  --max-tokens 4096 \
  --host "0.0.0.0" \
  --port 8000