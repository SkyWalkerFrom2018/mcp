from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import os
import logging
from abc import ABC, abstractmethod

from llama_index.core import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    Document,
)

class BaseIndex(ABC):
    """索引基类"""
    
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
        self.storage_context = storage_context or StorageContext()
        self.persist_dir = persist_dir
        self.index = None
        
    @classmethod
    @abstractmethod
    def from_documents(cls, documents: List[Document], **kwargs) -> "BaseIndex":
        """从文档构建索引（抽象方法）"""
        pass

    @abstractmethod
    def as_retriever(self, **kwargs):
        """获取检索器（抽象方法）"""
        pass

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

    # 新增公共方法
    def create_from_files(self, file_paths: Union[str, List[str], Path, List[Path]], **kwargs):
        """从文件创建索引（公共实现）"""
        from .utils import FileLoader
        documents = FileLoader.load_files(file_paths)
        return self.from_documents(documents, **kwargs)

    def create_from_directory(self, directory_path: Union[str, Path], **kwargs):
        """从目录创建索引（公共实现）"""
        from .utils import FileLoader
        documents = FileLoader.load_directory(directory_path, **kwargs)
        return self.from_documents(documents, **kwargs) 