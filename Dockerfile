# 基础镜像：CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# 添加PyTorch环境变量，确保使用CUDA
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# 安装系统依赖
RUN apt update && apt install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip git wget \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 验证Python版本
RUN python --version | grep -q "3.10" || (echo "Python version is not 3.10" && exit 1)

# 升级pip
RUN python -m pip install --upgrade pip setuptools wheel

# 先安装PyTorch（指定官方源，针对CUDA 12.1）
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 复制依赖文件并安装其他包
COPY requirements.txt /app/
WORKDIR /app

# 安装除了torch之外的其他依赖
RUN pip install --no-cache-dir -r requirements.txt

# 验证CUDA是否可用（可选，用于调试）
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 复制代码文件
COPY accelerate_config.yaml /app/
COPY server.py /app/
COPY start.sh /app/

# 赋予启动脚本执行权限
RUN chmod +x /app/start.sh

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["/app/start.sh"]
