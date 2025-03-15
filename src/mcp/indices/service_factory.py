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
        llm_model: str = "Qwen/Qwen-7B-Chat",
        device: str = "cuda"
    ) -> Settings:
        """创建服务上下文"""
        return Settings(
            llm=ServiceContextFactory._create_llm(llm_model, device),
            embed_model=ServiceContextFactory._create_embedding(embedding_model, device)
        )

    @staticmethod
    def _create_embedding(model_name: str, device: str) -> HuggingFaceEmbedding:
        """创建嵌入模型"""
        try:
            return HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                trust_remote_code=True
            )
        except Exception as e:
            logging.warning(f"加载嵌入模型失败: {e}")
            return HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
                device=device
            )

    @staticmethod
    def _create_llm(model_name: str, device: str) -> HuggingFaceLLM:
        """创建LLM"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            return HuggingFaceLLM(
                model=model,
                tokenizer=tokenizer,
                context_window=2048,
                max_new_tokens=512,
                generate_kwargs={"temperature": 0.1}
            )
        except Exception as e:
            logging.error(f"加载LLM失败: {e}")
            raise
