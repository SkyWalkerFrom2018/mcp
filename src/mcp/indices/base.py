from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import os
import logging

from llama_index.core import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

class BaseIndex:
    """基于LlamaIndex的索引基类"""
    
    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        persist_dir: Optional[str] = None,
    ):
        """初始化基础索引
        
        Args:
            service_context: LlamaIndex服务上下文
            storage_context: LlamaIndex存储上下文
            persist_dir: 索引持久化目录
        """
        self.service_context = service_context
        self.storage_context = storage_context
        self.persist_dir = persist_dir
        self.index = None
        
    def load(self) -> bool:
        """从持久化目录加载索引
        
        Returns:
            是否成功加载
        """
        if not self.persist_dir or not os.path.exists(self.persist_dir):
            return False
            
        try:
            self.index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=self.persist_dir),
                service_context=self.service_context
            )
            return True
        except Exception as e:
            logging.error(f"加载索引失败: {e}")
            return False
            
    def save(self) -> bool:
        """保存索引到持久化目录
        
        Returns:
            是否成功保存
        """
        if not self.persist_dir or not self.index:
            return False
            
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            return True
        except Exception as e:
            logging.error(f"保存索引失败: {e}")
            return False 