import base64
import io
import json
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

# 初始化FastAPI应用
app = FastAPI(title="Qwen-VL OpenAI-Compatible API", version="1.0")

# 全局变量：模型、tokenizer
model = None
tokenizer = None
device = "cuda"

# 模型配置
MODEL_NAME = "Qwen/Qwen-VL-Chat-7B"
DEVICE_MAP = "auto"
MAX_TOKENS = 2048

# ====================== 模型加载函数（多卡适配）======================
def load_model():
    global model, tokenizer
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 加载模型配置
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 多卡内存分配（适配4张4090）
    max_memory = get_balanced_memory(
        config,
        no_split_module_classes=["QwenBlock"],
        dtype=torch.float16,
        low_zero=False,
        max_memory_per_gpu="28GiB",  # 4090显存32GiB，预留4GiB
        cpu_memory="32GiB"
    )
    
    # 初始化空权重并分发给多卡
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    # 加载权重并分发到4张GPU
    model = load_checkpoint_and_dispatch(
        model,
        MODEL_NAME,
        device_map=DEVICE_MAP,
        max_memory=max_memory,
        no_split_module_classes=["QwenBlock"],
        dtype=torch.float16
    )
    
    # 设置模型为推理模式
    model.eval()
    print(f"模型加载完成，设备映射：{model.hf_device_map}")

# ====================== 请求体定义 ======================
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str  # "text" 或 "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str  # "user" 或 "assistant"
    content: List[ContentItem]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

# ====================== 辅助函数：解析图片 ======================
def parse_image_from_base64(image_b64: str) -> Image.Image:
    """从base64字符串解析图片"""
    try:
        # 移除前缀（如data:image/jpeg;base64,）
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败：{str(e)}")

# ====================== API接口：OpenAI兼容的chat/completions ======================
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # 1. 解析请求消息
        messages = request.messages
        if not messages or messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="最后一条消息必须是user角色")
        
        user_content = messages[-1].content
        text_prompt = ""
        images = []
        
        # 2. 提取文本和图片
        for item in user_content:
            if item.type == "text" and item.text:
                text_prompt = item.text
            elif item.type == "image_url" and item.image_url:
                image = parse_image_from_base64(item.image_url.url)
                images.append(image)
        
        if not text_prompt:
            raise HTTPException(status_code=400, detail="必须提供文本提问")
        
        # 3. 构造模型输入
        query = tokenizer.from_list_format([
            {"image": img} for img in images
        ] + [{"text": text_prompt}])
        
        # 4. 推理（多卡加速）
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=query,
                history=[],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.8,
                repetition_penalty=1.1
            )
        
        # 5. 构造OpenAI格式的响应
        return JSONResponse({
            "id": f"chat-{torch.randint(100000, 999999, (1,)).item()}",
            "object": "chat.completion",
            "created": int(torch.datetime.datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(text_prompt)),
                "completion_tokens": len(tokenizer.encode(response)),
                "total_tokens": len(tokenizer.encode(text_prompt)) + len(tokenizer.encode(response))
            }
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败：{str(e)}")

# ====================== 健康检查接口 ======================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# ====================== 启动时加载模型 ======================
@app.on_event("startup")
async def startup_event():
    print("开始加载Qwen-VL模型（4卡4090）...")
    load_model()
    print("模型加载完成，服务已就绪！")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # 多卡推理单进程即可，accelerate已处理分布式
        log_level="info"
    )