from cmath import e
import logging
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class ServiceContextFactory:
    """服务上下文工厂"""
    
    @staticmethod
    def create(
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        llm_model: str = "Qwen/Qwen-1_8B-Chat",
        device: str = "cuda"
    ) -> None:
        """配置全局设置（新版API要求）"""
        # 直接设置类属性而不是实例化
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device=device
        )
        Settings.llm = ServiceContextFactory._create_llm(llm_model, device)
        Settings.chunk_size = 512  # 直接设置类属性
        Settings.chunk_overlap = 64

    @staticmethod
    def _create_embedding(model_name: str, device: str) -> HuggingFaceEmbedding:
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

    @staticmethod
    def _create_llm(model_name: str, device: str) -> HuggingFaceLLM:
        """创建LLM"""
        # 转换为绝对路径
        import os
        abs_model_path = os.path.abspath(model_name)
        logging.info(f"LLM模型路径: {abs_model_path}")
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
                return ServiceContextFactory._create_llm("Qwen/Qwen-1.8B-Chat", device)
            except Exception as e2:
                logging.error(f"加载所有Qwen模型失败: {str(e2)}")
                raise

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
