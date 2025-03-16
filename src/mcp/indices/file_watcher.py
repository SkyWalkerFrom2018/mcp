import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mcp.indices.document_index import DocumentIndex
from llama_index.core import Document

class FileUpdateHandler(FileSystemEventHandler):
    def __init__(self, index_service: DocumentIndex):
        self.service = index_service
        
    def on_modified(self, event):
        if not event.is_directory:
            self._handle_file_change(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            # 新增文件时立即加载并建立索引
            self._handle_file_change(event.src_path)
            print(f"检测到新文件已创建: {event.src_path}")  # 添加日志用于区分

    def on_deleted(self, event):
        if not event.is_directory:
            self._handle_file_remove(event.src_path)

    def _handle_file_change(self, file_path: str):
        try:
            doc = self._load_document(file_path)
            self.service.update([doc])
        except Exception as e:
            print(f"文件更新失败: {file_path}, 错误: {str(e)}")

    def _handle_file_remove(self, file_path: str):
        doc_id = hashlib.md5(file_path.encode()).hexdigest()
        self.service.delete(doc_ids=[doc_id])

    def _load_document(self, path: str):
        with open(path, 'r') as f:
            return Document(
                text=f.read(),
                metadata={"source": path},
                id_=hashlib.md5(path.encode()).hexdigest()
            )

def start_file_watcher(index_service: DocumentIndex, watch_dir: str):
    if Path(watch_dir).is_dir():
        event_handler = FileUpdateHandler(index_service)
        observer = Observer()
        observer.schedule(event_handler, watch_dir, recursive=True)
        observer.start()
        return observer
    return None 