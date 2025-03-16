from mcp.indices.server import DocumentIndexService
from mcp.indices.config import Config

if __name__ == "__main__":
    config = Config()
    config.create_service_context()
    service = DocumentIndexService(watch_dir=config.watch_dir)
    service.start(host=config.host, port=config.port)