from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import os
import logging

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document,
    Node,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.readers.download import download_loader
from transformers import AutoTokenizer, AutoModelForCausalLM

from mcp.indices.base import BaseIndex
from mcp.indices.utils import FileLoader

# 辅助函数：获取BGE嵌入模型
def get_bge_embedding(model_name="BAAI/bge-large-zh-v1.5", device="cuda"):
    """获取BGE嵌入模型
    
    Args:
        model_name: 模型名称，可选值包括
                   BAAI/bge-large-zh-v1.5 (推荐但需要更多资源)
                   BAAI/bge-small-zh-v1.5 (轻量版)
        device: 设备，"cuda"或"cpu"
        
    Returns:
        嵌入模型
    """
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            trust_remote_code=True
        )
        return embed_model
    except Exception as e:
        logging.warning(f"加载BGE模型失败: {str(e)}，尝试使用更小的模型")
        # 如果大模型加载失败，尝试加载小模型
        return HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            device=device,
            trust_remote_code=True
        )

# 辅助函数：获取Qwen LLM模型
def get_qwen_llm(model_name="Qwen/Qwen-7B-Chat", device="cuda"):
    """获取Qwen LLM模型
    
    Args:
        model_name: 模型名称，可选较小模型如Qwen/Qwen-1.8B-Chat
        device: 设备，"cuda"或"cpu"
        
    Returns:
        LLM模型
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            revision=None
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            revision=None
        )
        
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=2048,
            max_new_tokens=512,
            generate_kwargs={
                "temperature": 0.1,
                "repetition_penalty": 1.1
            }
        )
        return llm
    except Exception as e:
        logging.warning(f"加载Qwen模型失败: {str(e)}，使用较小的Qwen模型")
        try:
            return get_qwen_llm("Qwen/Qwen-1.8B-Chat", device)
        except Exception as e2:
            logging.error(f"加载所有Qwen模型失败: {str(e2)}")
            raise

class DocumentStore(BaseIndex):
    """专注文档存储管理的实现"""
    
    def __init__(
        self,
        node_parser: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.node_parser = node_parser or SentenceSplitter()
        self.nodes: List[Node] = []
        self.documents: Dict[str, Document] = {}
        self._observers = []

    def add_observer(self, callback: Callable):
        """添加观察者"""
        self._observers.append(callback)

    def add_document(self, document: Document):
        """添加文档（增强版）"""
        super().add_document(document)
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
        """从LlamaIndex文档创建索引
        
        Args:
            documents: LlamaIndex文档列表
            show_progress: 是否显示进度
            
        Returns:
            self，用于链式调用
        """
        if not self.service_context:
            try:
                # 使用BGE模型做嵌入
                embed_model = get_bge_embedding(
                    model_name=self.embedding_model,
                    device=self.device
                )
                
                # 使用Qwen模型做LLM
                llm = get_qwen_llm(
                    model_name=self.llm_model,
                    device=self.device
                )
                
                Settings.llm = llm
                Settings.embed_model = embed_model
                Settings.node_parser = self.node_parser
                self.service_context = Settings
            except Exception as e:
                logging.error(f"创建模型失败: {str(e)}")
                raise
        
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
