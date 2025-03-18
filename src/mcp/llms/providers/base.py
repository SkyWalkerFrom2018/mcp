from abc import ABC, abstractmethod
from typing import Optional, Dict, List, AsyncGenerator

class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """同步生成文本"""
        pass
    
    @abstractmethod
    async def async_generate_text(self, prompt: str, **kwargs) -> str:
        """异步生成文本"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict], **kwargs) -> str:
        """同步聊天补全"""
        pass
    
    @abstractmethod
    async def async_chat_completion(self, messages: List[Dict], **kwargs) -> str:
        """异步聊天补全"""
        pass
    
    @abstractmethod
    def embeddings(self, text: str) -> List[float]:
        """生成嵌入向量"""
        pass 