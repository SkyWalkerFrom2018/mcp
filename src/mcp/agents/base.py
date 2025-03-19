from src.mcp.memory.base import BaseMemory
from src.mcp.prompts import BASIC_SYSTEM_PROMPT
from src.mcp.tools import ToolManager
from src.mcp.config import AgentConfig
from src.mcp.llm import LLMAdapter


class AgentState:
    """Agent运行时状态容器"""
    def __init__(self):
        self.current_task = None
        self.conversation_history = []
        self.context_window = []  # 短期上下文缓存
        self.last_tool_output = None

class BaseAgent:
    """智能体基类"""
    def __init__(self, 
                 memory: BaseMemory,
                 tools: ToolManager,
                 config: AgentConfig):
        self.memory = memory
        self.tools = tools
        self.state = AgentState()
        self.llm_adapter = LLMAdapter(config.llm_settings)
        self.system_prompt = BASIC_SYSTEM_PROMPT
    
    async def main_loop(self, input: str) -> str:
        """主交互循环"""
        # 1. 输入预处理
        processed_input = await self._preprocess(input)
        
        # 2. 生成响应
        response = await self._generate_response(processed_input)
        
        # 3. 后处理
        final_output = await self._postprocess(response)
        
        # 4. 更新状态
        self._update_state(final_output)
        
        return final_output

    async def respond(self, input_text: str) -> str:
        """生成响应并更新记忆"""
        self.memory.add("user", input_text)
        # 后续接入LLM调用
        response = f"模拟响应：{input_text}"
        self.memory.add("assistant", response)
        return response 