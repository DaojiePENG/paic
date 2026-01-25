# 基础镜像：CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt update && apt install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip git wget \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN python -m pip install --upgrade pip setuptools wheel

# 复制依赖文件并安装
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

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