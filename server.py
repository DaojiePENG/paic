import base64
import io
import json
import warnings
import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from PIL import Image
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 忽略所有警告
warnings.filterwarnings("ignore")

# ====================== 核心配置 ======================
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE_MAP = "auto"
MAX_TOKENS = 2048

# ====================== 生命周期管理 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    print(f"开始加载{MODEL_NAME}模型（4卡4090）...")
    load_model()
    print("模型加载完成，服务已就绪！")
    yield
    # 关闭时清理资源
    global model, processor
    if model is not None:
        del model
    if processor is not None:
        del processor
    torch.cuda.empty_cache()

# 初始化FastAPI应用
app = FastAPI(
    title="Qwen3-VL OpenAI-Compatible API", 
    version="1.0",
    lifespan=lifespan
)

# 全局变量
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 模型加载函数（终极兼容版）======================
def load_model():
    global model, processor
    
    # 1. 动态导入必要的类（避免版本问题）
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError:
        from qwen_vl.modeling_qwen import Qwen3VLForConditionalGeneration
        from qwen_vl.processing_qwen_vl import Qwen3VLProcessor as AutoProcessor
    
    # 2. 加载处理器（最基础方式）
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    except Exception as e:
        print(f"处理器加载警告: {e}")
        # 终极备用方案：手动构建
        from transformers import AutoTokenizer, Qwen3VLImageProcessor
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        image_processor = Qwen3VLImageProcessor.from_pretrained(MODEL_NAME)
        processor = type('Qwen3VLProcessor', (object,), {
            'tokenizer': tokenizer,
            'image_processor': image_processor,
            'apply_chat_template': tokenizer.apply_chat_template,
            'batch_decode': tokenizer.batch_decode
        })()
    
    # 3. 手动配置多卡内存（完全避开accelerate的版本敏感API）
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "24GiB" for i in range(num_gpus)}
    max_memory["cpu"] = "32GiB"
    print(f"检测到{num_gpus}张GPU，内存配置: {max_memory}")
    
    # 4. 直接加载模型（使用最基础的方式，避开from_config/_from_config）
    try:
        # 方式1：直接加载（推荐）
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map=DEVICE_MAP,
            max_memory=max_memory,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"直接加载模型失败，尝试备用方案: {e}")
        # 方式2：备用方案（适配旧版transformers）
        from accelerate import load_checkpoint_and_dispatch, init_empty_weights
        
        # 先获取配置
        config = Qwen3VLForConditionalGeneration.config_class.from_pretrained(MODEL_NAME)
        
        # 初始化空模型
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)
        
        # 分发模型
        model = load_checkpoint_and_dispatch(
            model,
            MODEL_NAME,
            device_map=DEVICE_MAP,
            max_memory=max_memory,
            no_split_module_classes=["QwenBlock"],
            dtype=torch.float16
        )
    
    # 5. 设置推理模式
    model.eval()
    print(f"模型加载完成！设备: {model.device if hasattr(model, 'device') else '多卡分布式'}")

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

# ====================== 辅助函数 ======================
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
        # 基础验证
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="最后一条消息必须是user角色")
        
        # 提取文本和图片
        user_content = request.messages[-1].content
        text_prompt = ""
        images = []
        
        for item in user_content:
            if item.type == "text" and item.text:
                text_prompt = item.text
            elif item.type == "image_url" and item.image_url:
                images.append(parse_image_from_base64(item.image_url.url))
        
        if not text_prompt:
            raise HTTPException(status_code=400, detail="必须提供文本提问")
        
        # 构造输入消息
        qwen_messages = [{
            "role": "user",
            "content": []
        }]
        
        # 添加图片
        for img in images:
            qwen_messages[0]["content"].append({"type": "image", "image": img})
        # 添加文本
        qwen_messages[0]["content"].append({"type": "text", "text": text_prompt})
        
        # 处理输入
        inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到模型设备
        if hasattr(model, 'device'):
            inputs = inputs.to(model.device)
        else:
            # 多卡分布式时自动处理
            pass
        
        # 推理生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.8,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=151643,  # Qwen默认pad token id
                eos_token_id=151643,   # Qwen默认eos token id
                use_cache=True
            )
        
        # 裁剪并解码输出
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 构造OpenAI格式响应
        return JSONResponse({
            "id": f"chat-{torch.randint(100000, 999999, (1,)).item()}",
            "object": "chat.completion",
            "created": int(int(datetime.datetime.now().timestamp())),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0])
            }
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败：{str(e)}")

# ====================== 健康检查 ======================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_count": torch.cuda.device_count(),
        "cuda_available": torch.cuda.is_available(),
        "model_name": MODEL_NAME
    }

# ====================== 启动 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        reload=False
    )