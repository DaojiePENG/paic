import requests
import base64
import os
import json
from typing import Dict, List, Optional

class PhysicalAICenterClient:
    """Physical AI Center Client 多图交互客户端"""
    
    # 默认支持的图片格式
    DEFAULT_ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp"]
    
    def __init__(
        self,
        ip: str = "10.120.17.131",
        port: int = 8001,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        timeout: int = 60,
        allowed_image_formats: Optional[List[str]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05
    ):
        """
        初始化Qwen3-VL客户端
        
        Args:
            ip: 服务器IP地址
            port: 服务端口
            model_name: 模型名称
            timeout: 请求超时时间（秒）
            allowed_image_formats: 支持的图片格式列表，默认使用DEFAULT_ALLOWED_IMAGE_FORMATS
            max_tokens: 生成的最大token数
            temperature: 生成温度
            top_p: 采样top_p值
            repetition_penalty: 重复惩罚系数
        """
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.timeout = timeout
        self.allowed_image_formats = allowed_image_formats or self.DEFAULT_ALLOWED_IMAGE_FORMATS
        
        # 生成参数配置
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        # 构建服务地址
        self.server_url_stream = f"http://{self.ip}:{self.port}/v1/chat/completions/stream"
        self.server_url_completion = f"http://{self.ip}:{self.port}/v1/chat/completions"
        self.health_check_url = f"http://{self.ip}:{self.port}/health"

    def image_to_base64(self, image_path: str) -> str:
        """将单张图片转为base64编码"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在：{image_path}")
        file_ext = image_path.split(".")[-1].lower()
        if file_ext not in self.allowed_image_formats:
            raise ValueError(f"不支持的图片格式：{file_ext}")
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"图片编码失败：{str(e)}")

    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_check_url, timeout=10)
            if response.status_code == 200:
                health_info = response.json()
                print(f"✅ 服务器健康状态：{health_info}")
                return health_info.get("status") == "healthy" and health_info.get("model_loaded")
            else:
                print(f"❌ 服务器健康检查失败，状态码：{response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到服务器：{str(e)}")
            return False

    def call_qwen_vl_multi_image(
        self,
        text_prompt: str,
        image_paths: Optional[List[str]] = None,  # 支持多张图片
        stream: bool = True  # 控制流式/非流式
    ) -> str:
        """
        调用Qwen3-VL服务（支持多图输入 + 流式/非流式切换）
        """
        if not self.check_server_health():
            return "❌ 服务器未就绪，无法调用"
        
        # 1. 构造基础content（文本 + 多张图片）
        content = [{"type": "text", "text": text_prompt}]
        if image_paths and len(image_paths) > 0:
            for idx, img_path in enumerate(image_paths):
                try:
                    img_b64 = self.image_to_base64(img_path)
                    img_ext = img_path.split(".")[-1].lower()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img_ext};base64,{img_b64}"}
                    })
                    print(f"📸 已加载图片 {idx+1}/{len(image_paths)}：{img_path}")
                except Exception as e:
                    return f"❌ 图片 {img_path} 处理失败：{str(e)}"
        
        # 2. 构造请求体
        payload: Dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "stream": stream  # 切换流式/非流式
        }
        
        # 3. 选择请求地址
        server_url = self.server_url_stream if stream else self.server_url_completion
        print(f"📡 正在调用服务器：{server_url}")

        try:
            response = requests.post(
                server_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                stream=stream  # 流式需开启stream参数
            )
            response.raise_for_status()
            print(f"🔍 响应状态码：{response.status_code}")

            # 4. 处理流式响应
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: ") and line_str != "data: [DONE]":
                            json_str = line_str[6:]
                            chunk = json.loads(json_str)
                            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                                content_chunk = chunk["choices"][0]["delta"]["content"]
                                full_response += content_chunk
                                print(content_chunk, end="", flush=True)
                print("\n")
                return full_response
            # 5. 处理非流式响应
            else:
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"❌ 调用异常：{str(e)}"