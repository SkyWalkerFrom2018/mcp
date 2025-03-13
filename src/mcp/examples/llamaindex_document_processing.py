
from mcp.indices.document_store import DocumentIndex

if __name__ == "__main__":

    # 创建文档索引 - GPU版本
    index = DocumentIndex(
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