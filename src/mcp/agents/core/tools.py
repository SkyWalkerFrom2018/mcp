class ToolManager:
    def __init__(self, tools: list):
        self.tools = {tool.name: tool for tool in tools}
        self.dependency_graph = self._build_dependency_graph()

    async def execute_tool_chain(self, tool_sequence: list) -> dict:
        """带依赖关系的工具链执行"""
        results = {}
        for tool_name in tool_sequence:
            tool = self.tools[tool_name]
            # 检查依赖是否满足
            if not self._check_dependencies(tool, results):
                raise ToolDependencyError(...)
            
            # 异步执行工具
            results[tool_name] = await tool.execute(
                inputs=self._resolve_inputs(tool, results)
            )
        return results 