from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import logging
import uuid

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Node
from llama_index.core.readers.download import download_loader

from mcp.indices.base import BaseIndex
from mcp.indices.utils import FileLoader

class DocumentStore(BaseIndex):
    """专注文档存储管理的实现"""
    
    def __init__(
        self,
        service_context: Settings,  # 强制要求传入service_context
        persist_dir: str,
        node_parser: Optional[Any] = None,
        **kwargs
    ):
        # 先让基类初始化存储上下文
        super().__init__(persist_dir=persist_dir, **kwargs)
        
        # 然后初始化子类特有属性
        self.service_context = service_context
        self.node_parser = node_parser or SentenceSplitter()
        self.nodes: List[Node] = []
        self.documents: Dict[str, Document] = {}
        self._observers = []
        self.index = None

    def add_observer(self, callback: Callable):
        """添加观察者"""
        self._observers.append(callback)

    def add_document(self, document: Document):
        """独立实现文档添加逻辑"""
        if not document.doc_id:
            document.doc_id = str(uuid.uuid4())
            
        if document.doc_id in self.documents:
            raise ValueError(f"文档 {document.doc_id} 已存在，请使用 update_document 方法")

        # 处理节点
        nodes = self.node_parser.get_nodes_from_documents([document])
        self.nodes.extend(nodes)
        
        # 存储文档
        self.documents[document.doc_id] = document
        self._notify_observers("add", [document])

    def _notify_observers(self, event_type: str, documents: List[Document]):
        """通知观察者"""
        for callback in self._observers:
            try:
                callback(event_type, documents)
            except Exception as e:
                logging.error(f"观察者回调失败: {str(e)}")

    @classmethod
    def from_documents(cls, documents: List[Document], **kwargs) -> "DocumentStore":
        instance = cls(**kwargs)
        for doc in documents:
            instance.add_document(doc)
        return instance

    def search(
        self, 
        query: str, 
        keyword_filter: Optional[Dict] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """混合搜索（示例实现）"""
        # 这里可以添加关键字过滤逻辑
        return [
            {"doc_id": doc_id, "metadata": doc.metadata}
            for doc_id, doc in self.documents.items()
            if query.lower() in doc.text.lower()
        ][:max_results]

    def create_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> "DocumentStore":
        """从LlamaIndex文档创建索引（简化版）"""
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context,
            storage_context=self.storage_context,
            show_progress=show_progress
        )
        
        if self.persist_dir:
            self.save()
            
        return self

    def create_from_files(
        self,
        file_paths: Union[str, List[str], Path, List[Path]],
        file_extractor: Optional[Callable] = None,
        show_progress: bool = True
    ) -> "DocumentStore":
        """从文件创建索引（优化版）"""
        # 使用统一的文件加载工具
        documents = FileLoader.load_files(
            file_paths=file_paths,
            file_extractor=file_extractor
        )
        return self.create_from_documents(documents, show_progress)

    def create_from_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "**/*.*",
        exclude_hidden: bool = True,
        show_progress: bool = True
    ) -> "DocumentStore":
        """从目录创建索引（优化版）"""
        # 使用统一的目录加载工具
        documents = FileLoader.load_directory(
            directory=directory_path,
            glob_pattern=glob_pattern,
            exclude_hidden=exclude_hidden
        )
        return self.create_from_documents(documents, show_progress)
        
    def query(
        self, 
        query_text: str, 
        similarity_top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """查询索引
        
        Args:
            query_text: 查询文本
            similarity_top_k: 返回的最相似节点数量
            **kwargs: 其他查询参数
            
        Returns:
            检索到的节点列表
        """
        if not self.index:
            raise ValueError("索引尚未创建或加载")
            
        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k, 
            **kwargs
        )
        nodes = retriever.retrieve(query_text)
        
        results = []
        for node in nodes:
            results.append({
                "content": node.text,
                "score": node.score,
                "metadata": node.metadata
            })
            
        return results 

    def persist(self):
        """覆盖父类方法实现定制化存储"""
        # 这里可以添加文档版本管理逻辑
        super().persist() 

    def as_retriever(self, **kwargs):
        """实现抽象方法"""
        if not self.index:
            raise ValueError("索引尚未创建或加载")
        return self.index.as_retriever(**kwargs) 

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> Dict[str, Any]:
        """批量添加文档（返回操作结果）"""
        results = {
            "total": len(documents),
            "success": 0,
            "errors": []
        }
        
        for doc in documents:
            try:
                # 调用当前类的 add_document 方法
                self.add_document(doc)
                results["success"] += 1
            except Exception as e:
                results["errors"].append({
                    "doc_id": getattr(doc, 'doc_id', 'unknown'),
                    "error": str(e)
                })
                logging.error(f"添加文档失败: {str(e)}")
        
        return results
    
    def get_index(self):
        """获取索引"""
        if not self.index:
            self.load()
        return self.index
