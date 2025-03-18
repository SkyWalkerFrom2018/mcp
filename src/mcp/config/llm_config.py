from functools import lru_cache
import json
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config:
    """配置类"""

    def __init__(self):
        with open("./configs/llms.json", "r") as f:
            items = json.load(f)
        self.host = items["host"]
        self.port = items["port"]
        self.provider = items["provider"]
        if self.provider == "deepseek":
            self.provider_config = DeepSeekConfig(**items["deepseek"])
        elif self.provider == "openai":
            self.provider_config = OpenAIConfig(items["openai"])
        elif self.provider == "google":
            self.provider_config = GoogleConfig(items["google"])
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

class DeepSeekConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    api_key: str
    base_url: Optional[str] = None
    model: str = "deepseek-reasoner"
    role_prompt: Optional[str] = None
    timeout: int = 30

class OpenAIConfig(BaseSettings):
    api_key: str = Field(..., env="LLM_API_KEY")
    base_url: Optional[str] = Field(None, env="LLM_ENDPOINT")
    model: str = Field("gpt-4o", env="LLM_MODEL")
    role_prompt: Optional[str] = Field(None, env="LLM_ROLE_PROMPT")
    timeout: int = Field(30, env="LLM_TIMEOUT")

class GoogleConfig(BaseSettings):
    api_key: str = Field(..., env="LLM_API_KEY")
    base_url: Optional[str] = Field(None, env="LLM_ENDPOINT")
    model: str = Field("gemini-1.5-flash", env="LLM_MODEL")
    role_prompt: Optional[str] = Field(None, env="LLM_ROLE_PROMPT")
    timeout: int = Field(30, env="LLM_TIMEOUT")

@lru_cache
def get_llm_config():
    return Config()