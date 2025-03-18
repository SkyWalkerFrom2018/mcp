import json
import requests
import aiohttp
from typing import List, Dict, AsyncGenerator

from mcp.config.llm_config import GoogleConfig
from mcp.llms.schemas import ChatResponse, EmbeddingResponse
from mcp.llms.exceptions import LLMRequestError, LLMRateLimitError
from mcp.llms.providers.base import LLMProvider
from tenacity import retry, wait_exponential, retry_if_exception_type

class GoogleProvider(LLMProvider):
    def __init__(self, config: GoogleConfig):
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = config.api_key
        self.timeout = config.timeout
        self.headers = {
            "Content-Type": "application/json"
        }
        
    def _validate_params(self, **kwargs):
        """参数验证"""
        if kwargs.get('temperature', 0.7) < 0 or kwargs.get('temperature', 0.7) > 1:
            raise ValueError("Temperature must be between 0 and 1")
        
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMRateLimitError)
    )
    def chat_completion(self, messages: List[Dict], **kwargs) -> ChatResponse:
        """同步聊天补全"""
        self._validate_params(**kwargs)
        
        # 转换消息格式为Google API格式
        formatted_messages = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]
        
        payload = {
            "contents": formatted_messages,
            "model": kwargs.get("model", "gemini-pro"),
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 2048),
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/models/{payload['model']}:generateContent?key={self.api_key}",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return ChatResponse(
                content=data['candidates'][0]['content']['parts'][0]['text'],
                model=payload['model'],
                usage={}  # Google API 目前不返回token使用信息
            )
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise LLMRateLimitError("API rate limit exceeded") from e
            raise LLMRequestError(f"HTTP error occurred: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise LLMRequestError("Invalid JSON response") from e

    async def async_chat_completion(self, messages: List[Dict], **kwargs) -> ChatResponse:
        """异步聊天补全"""
        self._validate_params(**kwargs)
        
        formatted_messages = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]
        
        payload = {
            "contents": formatted_messages,
            "model": kwargs.get("model", "gemini-pro"),
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 2048),
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/models/{payload['model']}:generateContent?key={self.api_key}",
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return ChatResponse(
                        content=data['candidates'][0]['content']['parts'][0]['text'],
                        model=payload['model'],
                        usage={}
                    )
            except aiohttp.ClientError as e:
                if response.status == 429:
                    raise LLMRateLimitError("API rate limit exceeded") from e
                raise LLMRequestError(f"HTTP error occurred: {str(e)}") from e

    async def stream_chat_completion(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        """流式聊天补全"""
        self._validate_params(**kwargs)
        
        formatted_messages = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]
        
        payload = {
            "contents": formatted_messages,
            "model": kwargs.get("model", "gemini-pro"),
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 2048),
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/models/{payload['model']}:streamGenerateContent?key={self.api_key}",
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    async for chunk in response.content:
                        if chunk:
                            data = json.loads(chunk.decode('utf-8'))
                            if 'candidates' in data and data['candidates']:
                                yield data['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                raise LLMRequestError(f"Streaming error: {str(e)}") from e

    def embeddings(self, text: str) -> List[float]:
        """生成嵌入向量"""
        payload = {
            "text": text,
            "model": "embedding-001"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/models/{payload['model']}:embedText?key={self.api_key}",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data['embedding']['values']
        except requests.exceptions.RequestException as e:
            raise LLMRequestError(f"Embedding request failed: {str(e)}") from e
