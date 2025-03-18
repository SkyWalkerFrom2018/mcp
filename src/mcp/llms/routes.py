from fastapi import APIRouter, Body, Depends
from mcp.llms.schemas import ChatRequest, EmbeddingRequest
from mcp.llms.providers.factory import LLMFactory
from mcp.config.llm_config import Config as LLMConfig, get_llm_config
router = APIRouter()

@router.post("/chat")
async def chat_completion(
    request: ChatRequest = Body(...),
    config: LLMConfig = Depends(get_llm_config)
):
    """聊天补全接口"""
    llm = LLMFactory.create_provider(config)
    return await llm.async_chat_completion(
        messages=request.messages,
        **request.model_dump(exclude={"messages"})
    )

@router.post("/embeddings")
async def generate_embeddings(
    request: EmbeddingRequest = Body(...),
    config: LLMConfig = Depends(get_llm_config)
):
    """生成嵌入向量接口"""
    llm = LLMFactory.create_provider(config)
    return llm.embeddings(request.text) 