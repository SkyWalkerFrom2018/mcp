from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Node
from .base import BaseIndex
from .utils import FileLoader  # 假设已有文件加载工具
import json
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore

class DocumentIndex(BaseIndex):
    """基于文档数据源的索引实现"""
    
    def __init__(
        self,
        persist_dir: str = "./storage/docs",
        node_parser: Any = None,
        **kwargs
    ):
        # 初始化节点解析器
        self.node_parser = node_parser or SentenceSplitter()
        self.nodes: List[Node] = []
        super().__init__(persist_dir=persist_dir, **kwargs)

    def create_from_datasource(
        self,
        data_source: Union[str, Path, List[Document]],
        **kwargs
    ) -> "DocumentIndex":
        # 自动判断是否为目录
        if isinstance(data_source, (str, Path)):
            is_dir = Path(data_source).is_dir()
            documents = FileLoader.load(data_source, is_directory=is_dir, **kwargs)
        else:
            documents = FileLoader.load(data_source, **kwargs)
        
        # 初始化存储上下文
        self.storage_context.docstore.add_documents(documents)
        
        # 构建索引
        self.nodes = self.node_parser.get_nodes_from_documents(documents)
        self.index = self._build_index(nodes=self.nodes)  # 使用节点模式
        
        self.save()
            
        return self

    def update(self, documents: List[Document], **kwargs) -> "DocumentIndex":
        """修复版增量更新"""
        # 删除旧节点
        doc_ids = [doc.doc_id for doc in documents]
        self.nodes = [n for n in self.nodes if n.ref_doc_id not in doc_ids]
        
        # 生成新节点
        new_nodes = self.node_parser.get_nodes_from_documents(documents)
        self.nodes.extend(new_nodes)
        
        # 重建索引（关键修改点）
        self.index = self._build_index(nodes=self.nodes)  # 显式传递节点
        self._notify_observers("update", documents)
        return self

    def delete(self, **kwargs) -> "DocumentIndex":
        """删除整个索引"""
        # 清除存储
        self.storage_context.vector_store.delete(**kwargs)
        self.storage_context.docstore.delete_all_documents()
        
        # 重置状态
        self.nodes.clear()
        self.index = None
        
        # 删除持久化文件
        if Path(self.persist_dir).exists():
            for f in Path(self.persist_dir).glob("*"):
                f.unlink()
            Path(self.persist_dir).rmdir()
        return self

    def search(
        self, 
        query: str, 
        keyword_filter: Optional[Dict] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """混合搜索（关键词+向量）"""
        # 构建过滤条件
        filters = []
        if keyword_filter:
            filters.append(lambda x: all(
                x.metadata.get(k) == v for k, v in keyword_filter.items()
            ))
        
        # 构建统一查询参数
        query_obj = VectorStoreQuery(
            query_str=query,
            node_ids=[n.node_id for n in self.nodes],
            filters=filters,
            similarity_top_k=max_results
        )
        
        retriever = self.index.as_retriever(
            vector_store_query=query_obj  # 统一通过这个参数传递
        )
        return self._format_results(retriever.retrieve(query))

    def query(
        self, 
        query_text: str, 
        similarity_top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """纯向量检索"""
        retriever = self.index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=similarity_top_k,
            **kwargs
        )
        return self._format_results(retriever.retrieve(query_text))

    def _build_index(self, ref_docs: List[Document] = None, nodes: List[Node] = None) -> Any:
        """独立实现索引构建逻辑"""
        # 参数校验
        if not ref_docs and not nodes:
            raise ValueError("需要提供ref_docs或nodes参数")
            
        # 优先使用节点数据
        if nodes:
            if not self.index:
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.storage_context,
                    service_context=Settings
                )
            else:
                # 增量更新节点
                self.index.insert_nodes(nodes)
            return self.index
            
        # 文档模式构建
        if not self.index:
            self.index = VectorStoreIndex.from_documents(
                documents=ref_docs,
                storage_context=self.storage_context,
                service_context=Settings
            )
        else:
            # 增量更新文档
            for doc in ref_docs:
                self.index.insert(doc)
                
        return self.index

    def _format_results(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """统一格式化结果"""
        return [{
            "content": node.node.text,
            "score": node.score,
            "metadata": node.node.metadata,
            "doc_id": node.node.ref_doc_id
        } for node in nodes]

    # 便捷构造方法
    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        persist_dir: str = "./storage/docs",
        node_parser: Optional[Any] = None,
        **kwargs
    ) -> "DocumentIndex":
        """从目录创建（完整参数版）"""
        return cls(
            persist_dir=persist_dir,
            service_context= Settings,
            node_parser=node_parser,
            **kwargs
        ).create_from_datasource(
            data_source=directory,
            **{
                k: v for k, v in kwargs.items()
                if k not in ['service_context', 'node_parser']
            }
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        persist_dir: str = "./storage/docs",
        node_parser: Optional[Any] = None,
        **kwargs
    ) -> "DocumentIndex":
        """从文档列表创建（完整参数版）"""
        return cls(
            persist_dir=persist_dir,
            service_context= Settings,
            node_parser=node_parser,
            **kwargs
        ).create_from_datasource(
            data_source=documents,
            **{
                k: v for k, v in kwargs.items()
                if k not in ['service_context', 'node_parser']
            }
        )
