class BaseMemory:
    """记忆系统基类"""
    def __init__(self):
        self.messages = []
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_recent(self, n: int = 5):
        return self.messages[-n:]

class LongTermMemory(BaseMemory):
    """长期记忆实现"""
    # 后续可添加向量存储等实现 