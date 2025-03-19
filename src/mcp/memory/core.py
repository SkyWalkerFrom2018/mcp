from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from pydantic import BaseModel
import numpy as np

class MemoryItem(BaseModel):
    """记忆项数据模型"""
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()
    embedding: Optional[np.ndarray] = None

class BaseMemory(ABC):
    """记忆系统抽象基类"""
    
    @abstractmethod
    async def add(self, item: MemoryItem):
        """添加记忆项"""
        pass
    
    @abstractmethod
    async def retrieve(
        self, 
        query: Optional[str] = None,
        limit: int = 5,
        recency_weight: float = 0.5
    ) -> List[MemoryItem]:
        """
        检索记忆项
        :param query: 查询文本（用于语义搜索）
        :param limit: 返回结果数量
        :param recency_weight: 时间权重（0-1, 0表示只考虑相关性）
        """
        pass
    
    @abstractmethod
    async def clear(self, before: Optional[datetime] = None):
        """清空记忆"""
        pass

class ShortTermMemory(BaseMemory):
    """短期记忆（基于内存）"""
    
    def __init__(self, max_items: int = 20):
        self._storage: List[MemoryItem] = []
        self.max_items = max_items
    
    async def add(self, item: MemoryItem):
        self._storage.append(item)
        # 维护存储大小
        if len(self._storage) > self.max_items:
            self._storage = self._storage[-self.max_items:]
    
    async def retrieve(self, **kwargs) -> List[MemoryItem]:
        return self._storage[-kwargs.get('limit', 5):]
    
    async def clear(self, **kwargs):
        self._storage = []

class VectorMemory(BaseMemory):
    """向量记忆（支持语义搜索）"""
    
    def __init__(self, embedding_model: Any, max_items: int = 1000):
        self.embedding_model = embedding_model
        self._storage: List[MemoryItem] = []
        self.max_items = max_items
    
    async def add(self, item: MemoryItem):
        # 生成嵌入向量
        if item.embedding is None:
            item.embedding = self.embedding_model.encode(item.content)
        self._storage.append(item)
        # 维护存储大小
        if len(self._storage) > self.max_items:
            self._storage = self._storage[-self.max_items:]
    
    async def retrieve(self, **kwargs) -> List[MemoryItem]:
        query = kwargs.get('query')
        limit = kwargs.get('limit', 5)
        recency_weight = kwargs.get('recency_weight', 0.5)
        
        if query is None:
            return self._storage[-limit:]
            
        # 计算查询向量
        query_embedding = self.embedding_model.encode(query)
        
        # 计算相似度得分
        scores = []
        for item in self._storage:
            recency = (item.timestamp.timestamp() / 
                      datetime.now().timestamp())
            similarity = np.dot(query_embedding, item.embedding)
            combined_score = (recency_weight * recency + 
                             (1 - recency_weight) * similarity)
            scores.append((combined_score, item))
        
        # 按得分排序
        sorted_items = sorted(scores, key=lambda x: x[0], reverse=True)
        return [item for _, item in sorted_items[:limit]]
    
    async def clear(self, **kwargs):
        self._storage = []

class MemoryManager:
    """综合记忆管理器"""
    
    def __init__(self):
        self.memories: Dict[str, BaseMemory] = {}
        self.default_memory = ShortTermMemory()
        
    def register_memory(self, name: str, memory: BaseMemory):
        self.memories[name] = memory
        
    def get_memory(self, name: str = "default") -> BaseMemory:
        return self.memories.get(name, self.default_memory)
    
    async def add_to_all(self, item: MemoryItem):
        """将记忆项添加到所有注册的记忆系统"""
        for memory in self.memories.values():
            await memory.add(item) 