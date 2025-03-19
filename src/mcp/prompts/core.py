from typing import Dict, Any
from string import Template
import json

class PromptTemplate:
    """提示词模板基类"""
    def __init__(self, 
                 template: str, 
                 required_keys: set = None,
                 metadata: Dict[str, Any] = None):
        self.template = template
        self.required_keys = required_keys or set()
        self.metadata = metadata or {"category": "general"}
        
    def format(self, **kwargs) -> str:
        """格式化模板并验证参数"""
        missing = self.required_keys - kwargs.keys()
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return Template(self.template).safe_substitute(kwargs)
    
    @classmethod
    def from_file(cls, file_path: str):
        """从文件加载模板"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            template=data['template'],
            required_keys=set(data.get('required_keys', [])),
            metadata=data.get('metadata', {})
        )

class SystemPrompt(PromptTemplate):
    """系统级提示模板"""
    def __init__(self, template: str, role: str = "assistant"):
        super().__init__(
            template=template,
            required_keys={"current_date", "user_name"},
            metadata={"type": "system", "role": role}
        )
        
    def format_with_context(self, context: Dict[str, Any]) -> str:
        """添加上下文信息"""
        return self.format(**context) + f"\n当前对话上下文：{json.dumps(context, ensure_ascii=False)}"

class ChainPrompt:
    """链式提示组合器"""
    def __init__(self, *prompts: PromptTemplate):
        self.prompts = prompts
        
    def format_chain(self, **kwargs) -> str:
        """按顺序组合多个提示"""
        return "\n\n".join(p.format(**kwargs) for p in self.prompts) 