from typing import Optional, List
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import numpy as np

from mcp.memory.base import BaseMemory
from mcp.memory.core import MemoryItem

Base = declarative_base()

class SQLMemoryItem(Base):
    """SQL数据库存储的记忆项"""
    __tablename__ = 'memory_items'
    
    id = Column(String, primary_key=True)
    content = Column(String)
    metadata = Column(JSON)
    timestamp = Column(DateTime)
    embedding = Column(JSON)  # 存储序列化的向量

class SQLMemory(BaseMemory):
    """基于SQL的长期记忆"""
    
    def __init__(self, db_url: str = "sqlite:///memory.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    async def add(self, item: MemoryItem):
        session = self.Session()
        db_item = SQLMemoryItem(
            id=str(item.timestamp.timestamp()),
            content=item.content,
            metadata=json.dumps(item.metadata),
            timestamp=item.timestamp,
            embedding=item.embedding.tolist() if item.embedding else None
        )
        session.add(db_item)
        session.commit()
    
    async def retrieve(self, **kwargs) -> List[MemoryItem]:
        session = self.Session()
        query = session.query(SQLMemoryItem)
        
        # 实现基本的时间过滤
        if 'start_time' in kwargs:
            query = query.filter(SQLMemoryItem.timestamp >= kwargs['start_time'])
        if 'end_time' in kwargs:
            query = query.filter(SQLMemoryItem.timestamp <= kwargs['end_time'])
            
        return [
            MemoryItem(
                content=item.content,
                metadata=json.loads(item.metadata),
                timestamp=item.timestamp,
                embedding=np.array(item.embedding) if item.embedding else None
            ) for item in query.limit(kwargs.get('limit', 5)).all()
        ]
    
    async def clear(self, **kwargs):
        session = self.Session()
        session.query(SQLMemoryItem).delete()
        session.commit() 