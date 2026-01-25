#!/bin/bash

# 设置Hugging Face缓存目录
export HF_HOME="/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"

# 设定Python解释器路径
PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/paic/bin/python"

# 定义颜色输出，方便区分日志信息
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 定义PID文件和日志文件路径，方便管理
PID_FILE_8000="./qwen30b_vl.pid"
PID_FILE_8001="./qwen2b_vl.pid"
LOG_FILE_8000="./qwen30b_vl_server.log"
LOG_FILE_8001="./qwen2b_vl_server.log"

# 先清理旧的PID文件（避免残留）
rm -f $PID_FILE_8000 $PID_FILE_8001

# 启动第一个服务：Qwen/Qwen3-VL-30B-A3B-Instruct (端口 8000)
echo -e "${YELLOW}正在启动 Qwen/Qwen3-VL-30B-A3B-Instruct 服务 (端口 8000)...${NC}"
nohup $PYTHON server.py \
  --model-name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --device-map "auto" \
  --max-tokens 4096 \
  --host "0.0.0.0" \
  --port 8000 > $LOG_FILE_8000 2>&1 &
# 保存第一个进程的PID到文件
echo $! > $PID_FILE_8000
PID_8000=$(cat $PID_FILE_8000)

# 等待2秒，避免资源竞争
sleep 2

# 启动第二个服务：Qwen/Qwen3-VL-2B-Instruct (端口 8001)
echo -e "${YELLOW}正在启动 Qwen/Qwen3-VL-2B-Instruct 服务 (端口 8001)...${NC}"
nohup $PYTHON server.py \
  --model-name "Qwen/Qwen3-VL-2B-Instruct" \
  --device-map "auto" \
  --max-tokens 4096 \
  --host "0.0.0.0" \
  --port 8001 > $LOG_FILE_8001 2>&1 &
# 保存第二个进程的PID到文件
echo $! > $PID_FILE_8001
PID_8001=$(cat $PID_FILE_8001)

# 提示启动完成，并显示PID信息
echo -e "${GREEN}两个服务已成功启动！${NC}"
echo -e "========================================"
echo -e "30B模型服务 (端口8000):"
echo -e "  PID: ${PID_8000}"
echo -e "  PID文件: ${PID_FILE_8000}"
echo -e "  日志文件: ${LOG_FILE_8000}"
echo -e "----------------------------------------"
echo -e "2B模型服务 (端口8001):"
echo -e "  PID: ${PID_8001}"
echo -e "  PID文件: ${PID_FILE_8001}"
echo -e "  日志文件: ${LOG_FILE_8001}"
echo -e "========================================"
echo -e "停止服务可执行："
echo -e "  停止8000端口: kill -9 $(cat $PID_FILE_8000)"
echo -e "  停止8001端口: kill -9 $(cat $PID_FILE_8001)"
echo -e "  停止所有服务: kill -9 $(cat $PID_FILE_8000) $(cat $PID_FILE_8001)"