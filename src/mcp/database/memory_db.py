from sqlalchemy import create_engine
from mcp.memory import SQLMemory
from mcp.memory.core import MemoryItem
import asyncio

def setup_memory_db():
    # 初始化SQL记忆
    sql_memory = SQLMemory(db_url="sqlite:///memory.db")
    
    # 示例：添加初始数据
    async def init_data():
        await sql_memory.add(MemoryItem(
            content="系统默认配置",
            metadata={"type": "configuration"}
        ))
    
    # 立即执行异步初始化
    asyncio.run(init_data())
    
    return sql_memory 