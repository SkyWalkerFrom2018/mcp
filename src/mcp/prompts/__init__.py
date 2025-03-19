# 管理提示词模板和自定义

from pathlib import Path
from .core import PromptTemplate, SystemPrompt, ChainPrompt

# 先建立基础提示词模板
BASIC_SYSTEM_PROMPT = """
你是一位专业的人工智能助手，请用简洁准确的中文回答问题。遵循以下规则：
1. 保持回答结构化
2. 避免主观臆测
3. 必要时提供推理过程
"""

REASONING_PROMPT_TEMPLATE = """
已知信息：{context}

问题：{question}

请分步骤解释你的推理过程：
"""

# 基础提示词
BASIC_SYSTEM = SystemPrompt("""\
你是一位专业的${assistant_role}，请用${language}回答问题。
遵循以下规则：
${rules}
""")

REASONING_PROMPT = PromptTemplate(
    template="""\
已知信息：{context}
问题：{question}
请分步骤解释你的推理过程：""",
    required_keys={"context", "question"},
    metadata={"type": "reasoning"}
)

# 从目录加载所有模板
_PROMPT_REGISTRY = {}
for p_file in Path(__file__).parent.glob("prompts/*.json"):
    _PROMPT_REGISTRY[p_file.stem] = PromptTemplate.from_file(p_file)

def get_prompt(name: str) -> PromptTemplate:
    """获取注册的提示模板"""
    return _PROMPT_REGISTRY[name]

__all__ = ['PromptTemplate', 'SystemPrompt', 'ChainPrompt', 'get_prompt']