from typing import List
from pydantic import BaseModel, Field

class ChatResponse(BaseModel):
    content: str
    model: str
    usage: dict

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

class ChatRequest(BaseModel):
    """聊天请求参数"""
    messages: list[dict] = Field(..., description="对话消息列表")
    model: str = Field("deepseek-reasoner", description="使用的模型名称")
    temperature: float = Field(0.7, ge=0, le=2, description="生成随机性控制")
    max_tokens: int = Field(2048, ge=1, description="最大生成token数")

class EmbeddingRequest(BaseModel):
    """嵌入向量请求参数"""
    text: str = Field(..., min_length=1, description="需要向量化的文本")
    model: str = Field("text-embedding-3-small", description="使用的嵌入模型名称")