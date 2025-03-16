from llama_index.core import Document
from mcp.indices.document_store import DocumentStore
from mcp.indices.coordinator import IndexCoordinator
from src.mcp.indices.config import ServiceContextFactory

from mcp.indices.vector_index import VectorIndex
from src.mcp.indices.document_index import DocumentIndex

if __name__ == "__main__":
    # 初始化组件
    embedding_model="./models/bge-large-zh-v1.5"
    llm_model="./models/qwen-7b-int4"
    ServiceContextFactory.create(
        embedding_model=embedding_model,
        llm_model=llm_model
    )

    doc_index = DocumentIndex.from_directory(
        directory="./data/documents",
        persist_dir="./data/indices",
    )
    doc_index.save()
    doc_index = DocumentIndex(persist_dir="./data/indices", vector_store_type='chroma')
    doc_index.load()

    result = doc_index.search("什么是生产管理系统？")
    print(result)