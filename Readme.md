# Physical AI Center 服务使用指引

## (1) 构建镜像
```bash
docker build -t paic:v1 .
```

## (2) 运行容器

### 创建模型缓存目录（避免重复下载）
```bash
export HF_CACHE_DIR="/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"
mkdir -p ${HF_CACHE_DIR}
```

### 启动容器（--gpus all 映射所有4张4090）
```bash
docker run -d \
  --name paic-server \
  --gpus all \
  --network host \
  -v ${HF_CACHE_DIR}:/root/.cache/huggingface \
  -v $(pwd):/app \
  --shm-size=64g \  # 多卡通信需要大共享内存
  paic:v1
```

## (3) 查看启动日志（确认模型加载成功）
```bash
docker logs -f paic-server
```