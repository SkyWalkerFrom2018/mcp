
from fastapi import FastAPI
from mcp.indices.routes import create_api_router
from mcp.indices.file_watcher import start_file_watcher
from mcp.indices.document_index import DocumentIndex
from mcp.indices.config import Config

class DocumentIndexService:
    def __init__(self, watch_dir: str = None):
        self.index = DocumentIndex(persist_dir=Config().persist_dir)
        self.index.load()
        self.app = FastAPI(title="Document Index Service")
        self.app.include_router(create_api_router(self.index))
        self.observer = start_file_watcher(self.index, watch_dir)

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)