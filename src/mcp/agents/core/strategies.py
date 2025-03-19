from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """策略模式基类"""
    @abstractmethod
    async def execute(self, context: dict) -> dict:
        """执行策略并返回结果"""
        pass

class DirectStrategy(BaseStrategy):
    """直接执行策略"""
    async def execute(self, context: dict) -> dict:
        # 直接执行动作的逻辑
        return {"action": "direct", "confidence": 0.9}

class ChainOfThoughtStrategy(BaseStrategy):
    """思维链策略"""
    async def execute(self, context: dict) -> dict:
        # 包含多步推理的逻辑
        return {"action": "reasoning", "steps": 3}

class ReActStrategy(BaseStrategy):
    """ReAct推理策略"""
    async def execute(self, context: dict) -> dict:
        # 结合推理和行动的循环逻辑
        return {"action": "react_loop", "iterations": 5} 