class EnhancedMemoryManager:
    def __init__(self, memory: BaseMemory):
        self.memory = memory
        self.importance_weights = {
            'user_query': 0.7,
            'tool_output': 0.5,
            'system_msg': 0.3
        }

    async def retrieve_context(self, query: str, top_k: int = 5) -> list:
        """带权重的记忆检索"""
        # 实现基于语义相似度的记忆检索
        # 结合时间衰减因子和重要性权重
        # ... 

    async def compress_memory(self):
        """记忆压缩与摘要生成"""
        # 当记忆超过阈值时自动执行
        # 使用LLM生成摘要并归档
        # ... 