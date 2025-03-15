from typing import Optional, List, Dict, Any
from llama_index.core import Document, Node
from .document_store import DocumentStore
from .vector_index import VectorIndex
import logging

class IndexCoordinator:
    """索引协调器，实现文档与向量的自动同步"""
    
    def __init__(
        self,
        doc_store: DocumentStore,
        vector_index: VectorIndex,
        sync_policy: str = "auto",  # auto|manual
        batch_size: int = 100,
        **kwargs
    ):
        self.doc_store = doc_store
        self.vector_index = vector_index
        self.sync_policy = sync_policy
        self.batch_size = batch_size
        self._pending_nodes: List[Node] = []
        
        # 注册文档存储的观察者
        if sync_policy == "auto":
            self.doc_store.add_observer(self._on_document_updated)

    def _on_document_updated(self, event_type: str, documents: List[Document]):
        """文档更新回调"""
        if event_type == "add":
            nodes = self.doc_store.node_parser(documents)
            self._pending_nodes.extend(nodes)
            
            if len(self._pending_nodes) >= self.batch_size:
                self.sync_index()

    def add_documents(
        self,
        documents: List[Document],
        sync: Optional[bool] = None
    ) -> Dict[str, Any]:
        """添加文档并同步索引"""
        # 添加文档到存储
        result = self.doc_store.add_documents(documents)
        
        # 根据策略决定是否同步
        should_sync = sync if sync is not None else (self.sync_policy == "auto")
        if should_sync:
            sync_result = self.sync_index()
            result["sync_result"] = sync_result
            
        return result

    def sync_index(self) -> Dict[str, Any]:
        """手动触发索引同步"""
        if not self._pending_nodes:
            return {"status": "skipped", "reason": "no pending updates"}
            
        try:
            # 获取当前所有节点（全量更新时需要）
            all_nodes = self.doc_store.nodes + self._pending_nodes
            
            # 更新向量索引
            self.vector_index.index.insert_nodes(all_nodes)
            self._pending_nodes.clear()
            
            # 持久化存储
            self.persist()
                
            return {"status": "success", "updated_nodes": len(all_nodes)}
        except Exception as e:
            logging.error(f"索引同步失败: {str(e)}")
            return {"status": "error", "reason": str(e)}

    def persist(self):
        """统一持久化管理"""
        # 优先持久化文档存储
        if self.doc_store.persist_dir:
            self.doc_store.persist()
        
        # 向量索引仅在独立存储时持久化
        if (self.vector_index.persist_dir and 
            self.vector_index.persist_dir != self.doc_store.persist_dir):
            self.vector_index.persist()

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """删除文档并同步索引"""
        # 从文档存储删除
        doc = self.doc_store.documents.pop(doc_id, None)
        if not doc:
            return {"status": "not_found"}
            
        # 从向量索引删除相关节点
        related_nodes = [n for n in self.vector_index.index.docstore.docs.values()
                        if n.ref_doc_id == doc_id]
        for node in related_nodes:
            self.vector_index.index.delete_node(node.node_id)
            
        return {
            "status": "success",
            "deleted_doc": doc_id,
            "deleted_nodes": len(related_nodes)
        }

    def search(
        self,
        query: str,
        mode: str = "hybrid",  # hybrid|vector|keyword
    ) -> List[Dict[str, Any]]:
        """统一搜索接口"""
        if mode == "hybrid":
            vector_results = self.vector_index.as_retriever().retrieve(query)
            keyword_results = self.doc_store.search(query)
            return self._merge_results(vector_results, keyword_results)
        elif mode == "vector":
            return self.vector_index.as_retriever().retrieve(query)
        else:
            return self.doc_store.search(query)

    def _merge_results(
        self,
        vector_results: List[Any],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并混合搜索结果"""
        
        # 合并逻辑
        merged = []
        seen_docs = set()
        
        # 优先处理向量结果
        for res in vector_results:
            doc_id = res.node.ref_doc_id
            if doc_id not in seen_docs:
                merged.append({
                    "doc_id": doc_id,
                    "score": res.score,
                    "type": "vector",
                    "content": res.node.text[:500] + "..."
                })
                seen_docs.add(doc_id)
                
        # 补充关键词结果
        for res in keyword_results:
            if res["doc_id"] not in seen_docs:
                merged.append({
                    "doc_id": res["doc_id"],
                    "score": 0.5,  # 默认相关性分数
                    "type": "keyword",
                    "content": self.doc_store.documents[res["doc_id"]].text[:500] + "..."
                })
                
        return sorted(merged, key=lambda x: x["score"], reverse=True)[:10]
