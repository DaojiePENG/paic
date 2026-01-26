import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os
import io
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import uvicorn
from typing import Optional, AsyncGenerator
import re

# ====================== 初始化配置 ======================
os.environ["HF_HOME"] = "/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"

# 初始化 TTS 模型（全局加载）
print("正在加载 Qwen3-TTS 模型...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print("模型加载完成！")

app = FastAPI(title="Qwen3-TTS 低延迟流式服务", version="2.0")

# ====================== 数据模型定义 ======================
class TTSRequest(BaseModel):
    text: str
    language: str = "Chinese"
    speaker: str = "Vivian"
    instruct: str = ""
    sentence_split: bool = True  # 是否开启短句分片
    chunk_ms: int = 20  # 每块音频时长（ms）

# ====================== 核心工具函数 ======================
def split_long_text(text: str) -> list:
    """
    将长文本切分为短句（按标点符号分割），减少单次推理时间
    """
    # 中文标点分割
    if re.search(r'[\u4e00-\u9fff]', text):
        sentences = re.split(r'([。！？；，])', text)
    else:
        sentences = re.split(r'([.!?;,\n])', text)
    # 重组句子（避免标点单独成句）
    result = []
    for i in range(0, len(sentences), 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    return [s.strip() for s in result if s.strip()]

async def real_time_tts_generator(
    text: str,
    language: str = "Chinese",
    speaker: str = "Vivian",
    instruct: str = "甜美女声",
    sentence_split: bool = True,
    chunk_ms: int = 20
) -> AsyncGenerator[bytes, None]:
    """
    真正的低延迟流式生成器：
    1. 长文本切分为短句
    2. 逐句推理，生成一句推送一句
    3. 每句音频再按时间分片，零缓冲推送
    """
    try:
        # 1. 切分长文本为短句
        if sentence_split:
            sentences = split_long_text(text)
        else:
            sentences = [text]
        
        sample_rate = model.sample_rate if hasattr(model, "sample_rate") else 16000
        chunk_samples = int(sample_rate * chunk_ms / 1000)  # 每块的采样点数

        # 2. 逐句推理 + 实时推送
        for sent in sentences:
            if not sent:
                continue
            
            # 异步推理单句（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: model.generate_custom_voice(
                    text=sent,
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                )
            )
            audio_data = wavs[0]
            current_pos = 0
            total_samples = len(audio_data)

            # 3. 单句音频按时间分片，零缓冲推送
            while current_pos < total_samples:
                end_pos = min(current_pos + chunk_samples, total_samples)
                audio_chunk = audio_data[current_pos:end_pos]
                current_pos = end_pos

                # 转为 WAV 字节（不封装成完整 WAV，只传裸 PCM 会更高效，这里兼容播放）
                buffer = io.BytesIO()
                sf.write(buffer, audio_chunk, sr, format='WAV')
                buffer.seek(0)
                yield buffer.read()

                # 关键：匹配播放速度，不额外加延迟
                await asyncio.sleep(0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式生成失败: {str(e)}")

# ====================== API 接口 ======================
@app.post("/tts/stream", summary="低延迟流式语音（推荐）")
async def stream_audio_api(request: TTSRequest):
    """
    低延迟流式语音接口：
    - 长文本自动分句，逐句生成推送
    - 零缓冲传输，无额外延迟
    """
    return StreamingResponse(
        real_time_tts_generator(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct,
            sentence_split=request.sentence_split,
            chunk_ms=request.chunk_ms
        ),
        media_type="audio/wav",
        headers={
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲（如果有反向代理）
        }
    )

@app.post("/tts/full", summary="完整语音生成（对比测试）")
async def generate_full_audio_api(request: TTSRequest):
    wavs, sr = model.generate_custom_voice(
        text=request.text,
        language=request.language,
        speaker=request.speaker,
        instruct=request.instruct,
    )
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format='WAV')
    buffer.seek(0)
    return Response(
        content=buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_full.wav"}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ====================== 启动服务 ======================
if __name__ == "__main__":
    uvicorn.run(
        app="server_tts:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # 必须单进程，避免多卡重复加载模型
        reload=False,
        loop="uvloop",  # 更快的事件循环
        http="h11",  # 禁用 http2，减少缓冲
    )