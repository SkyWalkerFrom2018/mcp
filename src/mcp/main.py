import os
from indices.document_store import DocumentIndex


if __name__ == "__main__":
    index = DocumentIndex(
        persist_dir="./data/indices",
        embedding_model="BAAI/bge-large-zh-v1.5",
        llm_model="Qwen/Qwen-7B-Chat",
        device="cuda"
    )
