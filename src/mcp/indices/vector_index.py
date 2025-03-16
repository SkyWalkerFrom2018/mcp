from typing import List, Optional, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
            
from .base import BaseIndex

class VectorIndex(BaseIndex):
    """专注向量检索操作的轻量类"""
    
    def __init__(self, index, persist_dir):
        super().__init__(persist_dir)
        self.index = index  # 接收已构建好的索引

    def as_retriever(self, **kwargs):
        """检索入口"""
        return self.index.as_retriever(**kwargs)

    def hybrid_search(self, query: str, top_k: int = 5):
        """混合检索示例"""
        return self.index.as_retriever(
            vector_store_query_mode="hybrid", 
            top_k=top_k
        ).retrieve(query)