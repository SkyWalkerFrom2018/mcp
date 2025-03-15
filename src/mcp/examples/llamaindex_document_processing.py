from mcp.indices.document_store import DocumentStore
from mcp.indices.coordinator import IndexCoordinator
from llama_index.core import Document, ServiceContextFactory

if __name__ == "__main__":

    # 创建文档索引 - GPU版本
    index = DocumentStore(
        persist_dir="./data/indices",
        embedding_model="./models/bge-large-zh-v1.5",
        llm_model="/home/itcamel/Workspace/mcp/models/qwen-7b-int4",
        device="cuda"
    )

    # CPU版本（资源有限时）
    #index_cpu = DocumentIndex(
    #    persist_dir="./data/indices", 
    #    embedding_model="BAAI/bge-small-zh-v1.5",
    #    llm_model="Qwen/Qwen-1.8B-Chat",
    #    device="cpu"
    #)

    # 方法1: 从目录加载文档
    index.create_from_directory(
        "./data/documents",
        exclude_hidden=True
    )

    # 方法2: 从特定文件加载
    # doc_index.create_from_files([
    #     "./documents/report.pdf",
    #     "./documents/article.txt",
    #     "./documents/paper.docx"
    # ])

    # 保存索引
    index.save()

    # 或在另一个应用中加载已有索引
    # loaded_index = DocumentIndex(persist_dir="./data/document_index")
    # loaded_index.load()

    # 初始化组件
    service_context = ServiceContextFactory.create()
    doc_store = DocumentStore(service_context=service_context)
    vector_index = VectorIndex(service_context=service_context)

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