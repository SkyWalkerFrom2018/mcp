from typing import List, Optional, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from .base import BaseIndex

class VectorIndex(BaseIndex):
    """向量索引实现，处理文档的向量化和检索"""
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self._init_vector_store()
        
    def _init_vector_store(self):
        """初始化向量存储后端（示例使用Chroma）"""
        if not self.vector_store and self.persist_dir:
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb
            
            chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self.vector_store = ChromaVectorStore(
                chroma_collection=chroma_client.get_or_create_collection("vectors")
            )
        
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        **kwargs
    ) -> "VectorIndex":
        """从文档构建向量索引"""
        instance = cls(**kwargs)
        
        # 构建索引（自动处理分块和向量化）
        instance.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=instance.storage_context,
            service_context=instance.service_context
        )
        return instance
        
    def as_retriever(self, similarity_threshold: float = 0.7, top_k: int = 5):
        """获取检索器"""
        return self.index.as_retriever(
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )
        
    def persist(self):
        """持久化向量索引"""
        if self.persist_dir and self.index:
            self.storage_context.persist(persist_dir=self.persist_dir)
        return super().persist()
