import os
os.environ["HF_HOME"] = "/mnt/slurmfs-A100_msp/user_data/dpeng108/data/huggingface_cache"
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import io
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import uvicorn
# 修复：补充完整的typing导入（关键）
from typing import AsyncGenerator, List, Optional
import re

# ====================== 初始化配置 ======================

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
    batch_size: int = 2  # 批量推理的句子数量（可根据GPU显存调整）

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

def batchify_sentences(sentences: List[str], batch_size: int) -> List[List[str]]:
    """
    将切分后的短句列表按batch_size分组，生成批量推理的批次
    """
    batches = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batches.append(batch)
    return batches

async def real_time_tts_generator(
    text: str,
    language: str = "Chinese",
    speaker: str = "Vivian",
    instruct: str = "甜美女声",
    sentence_split: bool = True,
    chunk_ms: int = 20,
    batch_size: int = 4
) -> AsyncGenerator[bytes, None]:
    """
    优化版低延迟流式生成器：
    1. 长文本切分为短句
    2. 短句按批次批量推理（提升GPU利用率）
    3. 逐句/逐片推送音频，保持低延迟
    4. 每句音频再按时间分片，零缓冲推送
    """
    try:
        # 1. 切分长文本为短句
        if sentence_split:
            sentences = split_long_text(text)
            if not sentences:
                raise ValueError("切分后无有效文本内容")
        else:
            sentences = [text]

        sample_rate = model.sample_rate if hasattr(model, "sample_rate") else 16000
        chunk_samples = int(sample_rate * chunk_ms / 1000)  # 每块的采样点数

        # 2. 将短句列表按batch_size分组
        sentence_batches = batchify_sentences(sentences, batch_size)

        # 3. 逐批次推理 + 实时推送
        for batch in sentence_batches:
            # 构造批量推理的参数（每个句子使用相同的speaker/language/instruct）
            batch_text = batch
            batch_language = [language] * len(batch)
            batch_speaker = [speaker] * len(batch)
            batch_instruct = [instruct] * len(batch)

            # 异步批量推理（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: model.generate_custom_voice(
                    text=batch_text,
                    language=batch_language,
                    speaker=batch_speaker,
                    instruct=batch_instruct,
                )
            )

            # 4. 处理当前批次的所有音频结果（按句子顺序）
            for audio_data in wavs:
                current_pos = 0
                total_samples = len(audio_data)

                # 单句音频按时间分片，零缓冲推送
                while current_pos < total_samples:
                    end_pos = min(current_pos + chunk_samples, total_samples)
                    audio_chunk = audio_data[current_pos:end_pos]
                    current_pos = end_pos

                    # 转为WAV字节（兼容播放，裸PCM可进一步优化）
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_chunk, sr, format='WAV')
                    buffer.seek(0)
                    yield buffer.read()

                    # 匹配播放速度，无额外延迟
                    await asyncio.sleep(0)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"输入参数错误: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式生成失败: {str(e)}")

# ====================== API 接口 ======================
@app.post("/tts/stream", summary="低延迟流式语音（Batch优化版）")
async def stream_audio_api(request: TTSRequest):
    """
    低延迟流式语音接口（Batch优化版）：
    - 长文本自动分句，按批次批量推理（提升效率）
    - 逐句/逐片推送音频，保持零缓冲低延迟
    - 可自定义batch_size适配GPU显存
    """
    return StreamingResponse(
        real_time_tts_generator(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct,
            sentence_split=request.sentence_split,
            chunk_ms=request.chunk_ms,
            batch_size=request.batch_size
        ),
        media_type="audio/wav",
        headers={
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用nginx缓冲
            "X-Content-Type-Options": "nosniff"
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