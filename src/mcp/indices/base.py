import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.graph_stores import SimpleGraphStore
class BaseIndex(ABC):
    """索引基类"""
    
    def __init__(
        self,
        persist_dir: str = "./storage",
        vector_store_type: str = "chroma",
        service_context: Settings=None,
        **kwargs
    ):
        self.persist_dir = persist_dir
        self.vector_store_type = vector_store_type
        self.service_context = service_context
        
        # 初始化核心存储组件
        self.docstore = SimpleDocumentStore()
        self.index_store = SimpleIndexStore()
        self.vector_store = self._create_vector_store()
        self.graph_store = SimpleGraphStore()

        # 构建存储上下文
        self.storage_context = StorageContext.from_defaults(
            vector_stores={"default": self.vector_store},
            docstore=self.docstore,
            index_store=self.index_store,
            graph_store=self.graph_store,
            persist_dir=self.persist_dir
        )
        self.index = None
        
    @abstractmethod
    def create_from_datasource(self, **kwargs) -> "BaseIndex":
        """从数据源重新构建索引"""
        pass
    
    @abstractmethod
    def update(self, documents: List[Document], **kwargs) -> "BaseIndex":
        """更新索引"""
        pass

    @abstractmethod
    def delete(self, **kwargs) -> "BaseIndex":
        """删除索引"""
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        keyword_filter: Optional[Dict] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索"""
        pass

    @abstractmethod
    def query(
        self, 
        query_text: str, 
        similarity_top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """查询"""
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
                self.storage_context,
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
            self.storage_context.persist(persist_dir=self.persist_dir)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            return True
        except Exception as e:
            logging.error(f"保存索引失败: {e}")
            return False 


    def _create_vector_store(self):
        """创建向量存储"""
        if self.vector_store_type == "chroma":
            # 确保目录存在
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            
            # 显式设置持久化配置
            chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=chromadb.config.Settings(
                    persist_directory=self.persist_dir,
                    allow_reset=True
                )
            )
            return ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection("main")
            )
        raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}") 