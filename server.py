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
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
from pydantic import BaseModel
import threading
from queue import Queue, Empty  # 这行是缺失的关键导入
from transformers import TextStreamer  # 导入原生Streamer
import argparse  # 新增：导入命令行参数解析模块
import re  # 补充：QwenVLStreamer中用到但未导入的模块

# 忽略所有警告
warnings.filterwarnings("ignore")

# ====================== 解析命令行参数 ======================
# 创建参数解析器
parser = argparse.ArgumentParser(description="Qwen3-VL OpenAI-Compatible API Server")
# 添加配置参数（带默认值，保持原有默认配置）
parser.add_argument(
    "--model-name", 
    type=str, 
    default="Qwen/Qwen3-VL-2B-Instruct",
    help="模型名称或本地路径 (默认: Qwen/Qwen3-VL-2B-Instruct)"
)
parser.add_argument(
    "--device-map", 
    type=str, 
    default="auto",
    help="设备映射配置 (默认: auto)"
)
parser.add_argument(
    "--max-tokens", 
    type=int, 
    default=2048,
    help="最大输入token数 (默认: 2048)"
)
parser.add_argument(
    "--port", 
    type=int, 
    default=8000,
    help="服务端口 (默认: 8000)"
)
parser.add_argument(
    "--host", 
    type=str, 
    default="0.0.0.0",
    help="服务监听地址 (默认: 0.0.0.0)"
)

# 解析参数
args = parser.parse_args()

# ====================== 核心配置（从命令行参数读取） ======================
MODEL_NAME = args.model_name
DEVICE_MAP = args.device_map
MAX_TOKENS = args.max_tokens
SERVER_HOST = args.host
SERVER_PORT = args.port

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

# ====================== 模型加载函数（适配MoE架构）======================
def load_model():
    global model, processor
    
    # 1. 动态导入必要的类（优先导入MoE版本，兼容普通版本）
    # 如果是MoE模型，则导入MoE类，否则导入普通类
    
    # 判断是否为 MoE 模型：包含 "moe"（不区分大小写）或符合 A{*}B 模式（如 A3B, A10B）
    def is_moe_model_name(model_name: str) -> bool:
        if "moe" in model_name.lower():
            return True
        # 匹配 A + 任意数字 + B 的模式，前后可有其他字符（如 30B-A3B-Instruct）
        if re.search(r"A\d+B", model_name):
            return True
        return False

    if is_moe_model_name(MODEL_NAME):
        # 优先导入MoE版本的模型类（适配30B-A3B-Instruct）
        from transformers import Qwen3VLMoeForConditionalGeneration  as AutoModelClass
        from transformers import AutoProcessor
        # 标记是否为MoE模型
        is_moe_model = is_moe_model_name(MODEL_NAME)
    else:
        # 备用：导入普通版本（兼容2B/7B等非MoE模型）
        # from transformers import Qwen3VLForConditionalGeneration as Qwen3VLMoeForConditionalGeneration
        # from transformers import AutoProcessor
        # is_moe_model = is_moe_model_name(MODEL_NAME)
        try:
            from transformers import Qwen3VLForConditionalGeneration as AutoModelClass
            from transformers import AutoProcessor
        except ImportError:
            from qwen_vl.modeling_qwen import Qwen3VLForConditionalGeneration as AutoModelClass
            from qwen_vl.processing_qwen_vl import Qwen3VLProcessor as AutoProcessor
    
    # 2. 加载处理器（保持原有逻辑，使用官方推荐的AutoProcessor）
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
    
    # 4. 加载模型（适配MoE架构，沿用官方推荐参数）
    try:
        # 方式1：MoE模型优先加载（官方推荐方式）
        model_kwargs = {
            "dtype": "auto",  # 自动匹配精度（替代原float16，更适配MoE）
            "device_map": DEVICE_MAP,
            "max_memory": max_memory,
            "trust_remote_code": True,
            # 可选：开启flash attention 2（需安装flash-attn）
            # "attn_implementation": "flash_attention_2",
        }
        # MoE模型专用加载
        model = AutoModelClass.from_pretrained(
            MODEL_NAME,
            **model_kwargs
        )
        print(f"成功加载MoE架构模型: {MODEL_NAME}")
    except Exception as e:
        print(f"MoE模型加载失败，尝试普通Qwen3VL方案: {e}")
        # 方式2：备用方案（适配旧版transformers/普通Qwen3VL模型）
        try:
            from transformers import Qwen3VLForConditionalGeneration
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
        except Exception as e2:
            raise Exception(f"所有模型加载方案均失败：{e2}")
    
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

# ====================== 优化的Qwen流式处理器（解决逐字碎片化+标准SSE格式） ======================
class QwenVLStreamer(TextStreamer):
    def __init__(self, processor, queue: Queue, stop_event: threading.Event, skip_prompt: bool = True):
        super().__init__(processor.tokenizer, skip_prompt=skip_prompt)
        self.processor = processor
        self.queue = queue
        self.stop_event = stop_event
        self.chat_id = f"chat-{torch.randint(100000, 999999, (1,)).item()}"
        self.created_time = int(datetime.datetime.now().timestamp())
        self.prompt_length = 0
        # 缓存字符，凑成词/短句再输出（解决逐字碎片化）
        self.char_buffer = ""
        # 中文分词分隔符（遇到这些符号就输出缓存）
        self.separators = re.compile(r"[，。！？；：、\n]")

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.stop_event.is_set():
            return
        
        # 跳过prompt部分
        if self.skip_prompt and self.prompt_length == 0:
            self.prompt_length = len(text)
            return
        
        if stream_end:
            # 输出剩余缓存
            if self.char_buffer.strip():
                self.send_chunk(self.char_buffer.strip())
            # 发送结束标记
            final_chunk = {
                "id": self.chat_id,
                "object": "chat.completion.chunk",
                "created": self.created_time,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            # 标准SSE格式：data: + JSON字符串 + \n\n
            self.queue.put(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n", timeout=1.0)
            return
        
        # 只处理新增的生成内容
        if self.skip_prompt and len(text) > self.prompt_length:
            new_text = text[self.prompt_length:]
            self.prompt_length = len(text)
        else:
            new_text = text

        if not new_text:
            return
        
        # 字符缓存逻辑：凑成词/短句再输出
        self.char_buffer += new_text
        # 检查是否遇到分隔符
        match = self.separators.search(self.char_buffer)
        if match:
            # 分割缓存：分隔符前的内容 + 分隔符 + 剩余内容
            split_pos = match.end()
            output_text = self.char_buffer[:split_pos]
            self.char_buffer = self.char_buffer[split_pos:]
            self.send_chunk(output_text)

    def send_chunk(self, text: str):
        """发送标准SSE格式的数据块"""
        if not text:
            return
        chunk = {
            "id": self.chat_id,
            "object": "chat.completion.chunk",
            "created": self.created_time,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        }
        # 核心修复：生成标准SSE格式字符串
        sse_data = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        try:
            self.queue.put(sse_data, timeout=1.0)
        except:
            pass

# ====================== API接口：保持原有逻辑，无需修改 ======================
@app.post("/v1/chat/completions/stream")
async def create_chat_completion_stream(request: ChatCompletionRequest):
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
        
        # 构造Qwen格式的消息
        qwen_messages = [{
            "role": "user",
            "content": []
        }]
        for img in images:
            qwen_messages[0]["content"].append({"type": "image", "image": img})
        qwen_messages[0]["content"].append({"type": "text", "text": text_prompt})

        # 生成Qwen3-VL标准的文本模板
        text_template = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入：文本+图片
        inputs = processor(
            text=text_template,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS
        )

        # 设备适配
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(model.device if hasattr(model, 'device') else "cuda:0")
        
        # 流式生成准备
        result_queue = Queue(maxsize=50)
        stop_event = threading.Event()
        streamer = QwenVLStreamer(processor, result_queue, stop_event, skip_prompt=True)
        
        # 同步生成函数
        def generate_with_streamer():
            try:
                with torch.no_grad():
                    model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=0.8,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.pad_token_id or 151643,
                        eos_token_id=processor.tokenizer.eos_token_id or 151643,
                        streamer=streamer,
                        use_cache=True,
                        num_beams=1,
                        length_penalty=1.0
                    )
            except Exception as e:
                import traceback
                error_detail = f"{str(e)}\n{traceback.format_exc()}"
                print(f"生成出错: {error_detail}")
                error_resp = {
                    "error": {"message": str(e), "type": "streaming_error", "param": None, "code": 500}
                }
                result_queue.put(f"data: {json.dumps(error_resp, ensure_ascii=False)}\n\n", timeout=1.0)
            finally:
                result_queue.put("[DONE]", timeout=1.0)
                stop_event.set()
        
        # 异步生成器：直接返回队列中的SSE格式数据
        async def stream_generator():
            gen_thread = threading.Thread(target=generate_with_streamer, daemon=True)
            gen_thread.start()
            
            try:
                while True:
                    try:
                        chunk = result_queue.get(timeout=0.1)
                        if chunk == "[DONE]":
                            break
                        # 直接yield标准SSE格式的字符串
                        yield chunk
                    except Empty:
                        if not gen_thread.is_alive() and result_queue.empty():
                            break
                        await asyncio.sleep(0.01)
                        continue
            finally:
                stop_event.set()
                try:
                    gen_thread.join(timeout=2.0)
                except:
                    pass
        
        # 返回流式响应，指定媒体类型为text/event-stream
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"流式接口错误: {error_detail}")
        raise HTTPException(status_code=500, detail=f"流式推理失败：{str(e)}")

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
        host=SERVER_HOST,  # 使用命令行参数
        port=SERVER_PORT,  # 使用命令行参数
        workers=1,
        log_level="info",
        reload=False
    )