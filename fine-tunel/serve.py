import os
import time  # 添加 time 模块导入
from typing import List, Literal, Optional, Union, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import uvicorn
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取配置
MODEL_PATH = os.getenv('MODEL_PATH', './output/best_model')  # 模型路径
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2048'))  # 最大生成长度

app = FastAPI(title="Qwen API Server")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 风格的请求模型
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(default="user")
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = MAX_TOKENS
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]

# 初始化模型
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    llm = LLM(model=MODEL_PATH, trust_remote_code=True)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def format_chat_messages(messages: List[ChatMessage]) -> str:
    """将消息列表格式化为模型输入"""
    formatted_text = ""
    for msg in messages:
        if msg.role == "system":
            formatted_text += f"系统: {msg.content}\n"
        elif msg.role == "user":
            formatted_text += f"用户: {msg.content}\n"
        elif msg.role == "assistant":
            formatted_text += f"助手: {msg.content}\n"
    formatted_text += "助手: "  # 添加提示词
    return formatted_text

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # 准备采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # 格式化输入
        prompt = format_chat_messages(request.messages)
        
        # 生成回复
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # 计算 token 使用量
        input_tokens = len(prompt)
        output_tokens = len(generated_text)
        total_tokens = input_tokens + output_tokens

        # 构造响应
        response = {
            "id": "chatcmpl-" + os.urandom(4).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text.strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        }

        return response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT) 