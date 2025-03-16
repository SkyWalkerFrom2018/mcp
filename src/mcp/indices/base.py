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
        **kwargs
    ):
        # 分离存储路径（关键修改）
        self.chroma_persist_dir = os.path.join(persist_dir, "chroma")
        self.llama_persist_dir = persist_dir

        self.vector_store_type = vector_store_type
        
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
            persist_dir=self.llama_persist_dir
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
        """独立加载逻辑"""
        try:
            # 先加载 Chroma
            self.vector_store = self._create_vector_store()
            # 再加载其他组件
            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.llama_persist_dir,
                vector_stores={"default": self.vector_store}
            )
            self.index = load_index_from_storage(self.storage_context)
            return True
        except Exception as e:
            logging.error(f"加载失败: {e}")
            return False

    def save(self) -> bool:
        """分离持久化操作"""
        try:
            # 保存 llama-index 元数据
            self.storage_context.persist(persist_dir=self.llama_persist_dir)
            # Chroma 会自动持久化，无需额外操作
            return True
        except Exception as e:
            logging.error(f"保存失败: {e}")
            return False

    def _create_vector_store(self):
        """创建向量存储"""
        if self.vector_store_type == "chroma":
            # Chroma 使用独立存储路径
            Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True, mode=0o777)
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=chromadb.config.Settings(is_persistent=True)
            )
            return ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection("main")
            )
        raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}") 