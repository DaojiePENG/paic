#!/bin/bash

export HF_HOME="/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"
export HF_TOKEN="your_hugging_face_token_here"
# huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
#   --local-dir-use-symlinks False \
#   --revision main \
#   --token $HF_TOKEN


# # 手动单个下载： Qwen/Qwen3-VL-2B-Instruct; Qwen/Qwen3-VL-30B-A3B-Instruct-FP8; Qwen/Qwen3-VL-30B-A3B-Instruct; 
# hf download Qwen/Qwen3-VL-30B-A3B-Instruct \
#   --revision main \
#   --token $HF_TOKEN


# 检查 HF_TOKEN 环境变量是否已设置
if [ -z "$HF_TOKEN" ]; then
    echo "错误：未设置 HF_TOKEN 环境变量！"
    echo "请先执行：export HF_TOKEN='你的Hugging Face Token'"
    exit 1
fi

# 定义需要下载的模型列表
MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-8B-Thinking"
    "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
    "Qwen/Qwen3-VL-30B-A3B-Thinking"
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    # "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
    # "Qwen/Qwen3-VL-32B-Thinking-FP8"
    # "Qwen/Qwen3-VL-32B-Instruct-FP8"
)

# 遍历模型列表并逐个下载
for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "开始下载模型：$MODEL"
    echo "========================================"
    
    # 执行下载命令
    hf download "$MODEL" \
        --revision main \
        --token "$HF_TOKEN"
    
    # 检查上一条命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "✅ 模型 $MODEL 下载完成！"
    else
        echo "❌ 模型 $MODEL 下载失败！"
        # 可选：如果希望某个模型下载失败就停止脚本，取消下面这行注释
        # exit 1
    fi
    
    echo -e "\n"
done

echo "🎉 所有模型下载任务已执行完毕！"