from mcp.llms.providers.base import LLMProvider
from mcp.llms.providers.deepseek import DeepSeekProvider
from mcp.llms.providers.google import GoogleProvider
from mcp.config.llm_config import Config as LLMConfig
from src.mcp.llms.providers.openai import OpenAIProvider

class LLMFactory:
    @staticmethod
    def create_provider(config: LLMConfig) -> LLMProvider:
        providers = {
            "deepseek": DeepSeekProvider,
            "openai": OpenAIProvider,
            "google": GoogleProvider
        }
        return providers[config.provider.lower()](config.provider_config)