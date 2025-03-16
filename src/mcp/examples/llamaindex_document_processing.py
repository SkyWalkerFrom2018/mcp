from llama_index.core import Document
from mcp.indices.document_store import DocumentStore
from mcp.indices.coordinator import IndexCoordinator
from mcp.indices.service_factory import ServiceContextFactory

from mcp.indices.vector_index import VectorIndex
from src.mcp.indices.document_index import DocumentIndex

if __name__ == "__main__":
    # 初始化组件
    embedding_model="./models/bge-large-zh-v1.5"
    llm_model="./models/qwen-7b-int4"
    service_context = ServiceContextFactory.create(
        embedding_model=embedding_model,
        llm_model=llm_model
    )

    doc_index = DocumentIndex.from_directory(
        directory="./data/documents",
        persist_dir="./data/indices",
        service_context=service_context
    )

    doc_store = DocumentStore(
        service_context=service_context,
        persist_dir="./data/indices"
    )

    vector_index = VectorIndex(doc_store.get_index(), persist_dir="./data/indices")

    # 创建协调器（自动同步模式）
    coordinator = IndexCoordinator(
        doc_store=doc_store,
        vector_index=vector_index,
        sync_policy="auto",
        batch_size=50
    )

    # 添加文档（自动触发向量索引更新）
    coordinator.add_documents([
        Document(text="RAG系统介绍..."),
        Document(text="向量索引原理...")
    ])

    # 执行查询
    results = coordinator.search("什么是向量索引？", mode="hybrid")

    # 输出结果
    for i, result in enumerate(results):
        print(f"结果 {i+1} (相关度: {result['score']:.4f})")
        print(f"内容: {result['content'][:200]}..." if len(result['content']) > 200 else result['content'])
        print(f"来源: {result['metadata'].get('source', '未知')}")
        print("-" * 50)

    # 删除文档
    coordinator.delete_document("doc_001")

    # 手动同步（适用于manual模式）
    coordinator.sync_index()

    # 执行查询
    results = index.query(
        "实现原理是什么？", 
        similarity_top_k=3
    )

    # 输出结果
    for i, result in enumerate(results):
        print(f"结果 {i+1} (相关度: {result['score']:.4f})")
        print(f"内容: {result['content'][:200]}..." if len(result['content']) > 200 else result['content'])
        print(f"来源: {result['metadata'].get('source', '未知')}")
        print("-" * 50) 