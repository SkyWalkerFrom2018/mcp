from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import os
import logging

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.readers.download import download_loader
from transformers import AutoTokenizer, AutoModelForCausalLM

from mcp.indices.base import BaseIndex

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

class DocumentIndex(BaseIndex):
    """文档向量索引实现"""
    
    def __init__(
        self,
        service_context: Optional[Settings] = None,
        storage_context: Optional[StorageContext] = None,
        persist_dir: Optional[str] = None,
        node_parser: Optional[Any] = None,
        embedding_model: Optional[str] = "BAAI/bge-large-zh-v1.5",
        llm_model: Optional[str] = "Qwen/Qwen-7B-Chat",
        device: str = "cuda"
    ):
        """初始化文档索引
        
        Args:
            service_context: LlamaIndex服务上下文
            storage_context: LlamaIndex存储上下文
            persist_dir: 索引持久化目录
            node_parser: 节点解析器
            embedding_model: 嵌入模型名称
            llm_model: LLM模型名称
            device: 设备，"cuda"或"cpu"
        """
        super().__init__(
            service_context=service_context,
            storage_context=storage_context,
            persist_dir=persist_dir
        )
        
        # 使用默认解析器或自定义解析器
        self.node_parser = node_parser or SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )
        
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.device = device
        
    def create_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> "DocumentIndex":
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
    ) -> "DocumentIndex":
        """从文件创建索引
        
        Args:
            file_paths: 文件路径或路径列表
            file_extractor: 自定义文件提取器
            show_progress: 是否显示进度
            
        Returns:
            self，用于链式调用
        """
        # 将单个路径转换为列表
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
            
        # 转换为字符串路径
        str_paths = [str(p) for p in file_paths]
        
        # 根据文件类型自动选择加载器
        documents = []
        for file_path in str_paths:
            if os.path.isdir(file_path):
                # 处理目录
                simple_directory_reader = download_loader("SimpleDirectoryReader")
                loader = simple_directory_reader(file_path)
                docs = loader.load_data()
                documents.extend(docs)
            else:
                # 处理单个文件
                ext = os.path.splitext(file_path)[1].lower()
                
                if file_extractor:
                    # 使用自定义提取器
                    docs = file_extractor(file_path)
                    documents.extend(docs)
                elif ext == '.pdf':
                    # PDF文件
                    pdf_reader = download_loader("PDFReader")
                    loader = pdf_reader()
                    docs = loader.load_data(file=file_path)
                    documents.extend(docs)
                elif ext in ['.docx', '.doc']:
                    # Word文件
                    docx_reader = download_loader("DocxReader")
                    loader = docx_reader()
                    docs = loader.load_data(file=file_path)
                    documents.extend(docs)
                elif ext in ['.txt', '.md']:
                    # 文本文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    document = Document(text=text, metadata={"source": file_path})
                    documents.append(document)
                else:
                    logging.warning(f"不支持的文件类型: {file_path}")
        
        return self.create_from_documents(documents, show_progress)
        
    def create_from_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "**/*.*",
        exclude_hidden: bool = True,
        show_progress: bool = True
    ) -> "DocumentIndex":
        """从目录创建索引
        
        Args:
            directory_path: 目录路径
            glob_pattern: 文件匹配模式
            exclude_hidden: 是否排除隐藏文件
            show_progress: 是否显示进度
            
        Returns:
            self，用于链式调用
        """
        simple_directory_reader = download_loader("SimpleDirectoryReader")
        loader = simple_directory_reader(
            str(directory_path),
            recursive=True,
            exclude_hidden=exclude_hidden,
            file_extractor=None,  # 使用默认提取器
        )
        documents = loader.load_data()
        
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

if __name__ == "__main__":
    index = DocumentIndex(
        persist_dir="./data/indices",
        embedding_model="BAAI/bge-large-zh-v1.5",
        llm_model="Qwen/Qwen-7B-Chat",
        device="cuda"
    )
