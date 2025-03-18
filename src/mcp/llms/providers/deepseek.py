import json
import requests
import aiohttp
from typing import List, Dict, AsyncGenerator

from mcp.config.llm_config import DeepSeekConfig
from mcp.llms.schemas import ChatResponse, EmbeddingResponse
from mcp.llms.exceptions import LLMRequestError, LLMRateLimitError
from mcp.llms.providers.base import LLMProvider
from tenacity import retry, wait_exponential, retry_if_exception_type

class DeepSeekProvider(LLMProvider):
    def __init__(self, config: 'DeepSeekConfig'):
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model
        self.role_prompt = config.role_prompt
        self.timeout = config.timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
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
        
        if self.role_prompt:
            messages.insert(0, {"role": "system", "content": self.role_prompt})
        
        payload = {
            "messages": messages,
            "model": kwargs.get("model", "deepseek-chat"),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return ChatResponse(
                content=data['choices'][0]['message']['content'],
                model=data['model'],
                usage=data['usage']
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
        
        if self.role_prompt:
            messages.insert(0, {"role": "system", "content": self.role_prompt})
        payload = {
            "messages": messages,
            "model": kwargs.get("model", "deepseek-chat"),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # 新增调试输出
                    print("\n=== API Response ===")
                    print(f"Model: {data['model']}")
                    print(f"Content: {data['choices'][0]['message']['content']}")
                    print(f"Usage: {json.dumps(data['usage'], indent=2)}")
                    print("====================\n")
                    
                    return ChatResponse(
                        content=data['choices'][0]['message']['content'],
                        model=data['model'],
                        usage=data['usage']
                    )
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    raise LLMRateLimitError("API rate limit exceeded") from e
                raise LLMRequestError(f"HTTP error occurred: {str(e)}") from e

    def generate_text(self, prompt: str, **kwargs) -> str:
        """同步文本生成（适配聊天接口）"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, **kwargs).content

    async def async_generate_text(self, prompt: str, **kwargs) -> str:
        """异步文本生成"""
        messages = [{"role": "user", "content": prompt}]
        response = await self.async_chat_completion(messages, **kwargs)
        return response.content

    async def stream_chat_completion(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        """流式聊天补全"""
        self._validate_params(**kwargs)
        
        if self.role_prompt:
            messages.insert(0, {"role": "system", "content": self.role_prompt})
        payload = {
            "messages": messages,
            "model": kwargs.get("model", "deepseek-chat"),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    async for chunk in response.content:
                        if chunk:
                            data = chunk.decode('utf-8')
                            if data.startswith("data: "):
                                json_data = json.loads(data[6:])
                                yield json_data['choices'][0]['delta'].get('content', '')
            except Exception as e:
                raise LLMRequestError(f"Streaming error: {str(e)}") from e

    def embeddings(self, text: str) -> List[float]:
        """生成嵌入向量"""
        payload = {
            "input": text,
            "model": "text-embedding-001"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data['data'][0]['embedding']
        except requests.exceptions.RequestException as e:
            raise LLMRequestError(f"Embedding request failed: {str(e)}") from e 