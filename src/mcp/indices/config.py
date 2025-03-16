from cmath import e
import json
import logging
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Config:
    """配置类"""

    def __init__(self):
        with open("./configs/indices.json", "r") as f:
            items = json.load(f)
        self.host = items["host"]
        self.port = items["port"]
        self.watch_dir = items["watch_dir"]
        self.persist_dir = items["persist_dir"]
        self.vector_store_type = items["vector_store_type"]
        self.embedding_model = items["embedding_model"]
        self.device = items["device"]
        self.chunk_size = items["chunk_size"]
        self.chunk_overlap = items["chunk_overlap"]
        self.chunk_size_limit = items["chunk_size_limit"]
        self.llm_model = items["llm_model"]
        

    def create_service_context(self) -> None:
        """配置全局设置（新版API要求）"""
        # 直接设置类属性而不是实例化
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model,
            device=self.device
        )
        Settings.llm = self._create_llm()
        Settings.chunk_size = self.chunk_size  # 直接设置类属性
        Settings.chunk_overlap = self.chunk_overlap

    def _create_embedding(self, model_name: str, device: str) -> HuggingFaceEmbedding:
        """创建嵌入模型"""
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

    def _create_llm(self) -> HuggingFaceLLM:
        """创建LLM"""
        # 转换为绝对路径
        import os
        abs_model_path = os.path.abspath(self.llm_model)
        logging.info(f"LLM模型路径: {abs_model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model,
                trust_remote_code=True,
                local_files_only=True,
                revision=None
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
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
            logging.warning(f"加载Qwen模型失败: {str(e)}")
            raise