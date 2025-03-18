from mcp.indices.server import DocumentIndexService
from mcp.indices.config import Config as IndicesConfig
from mcp.config.llm_config import Config as LlmsConfig
from mcp.llms.server import LlmsService

if __name__ == "__main__":
    # 索引服务
    #indices_config = IndicesConfig()
    #indices_config.create_service_context()
    #indices_service = DocumentIndexService(watch_dir=indices_config.watch_dir)
    
    # LLM服务
    llms_config = LlmsConfig()
    llms_service = LlmsService(config=llms_config)

    # 使用多进程并行启动
    import multiprocessing
    processes = [
        #multiprocessing.Process(target=indices_service.start, kwargs={"host": indices_config.host, "port": indices_config.port}),
        multiprocessing.Process(target=llms_service.start, kwargs={"host": llms_config.host, "port": llms_config.port})
    ]

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()  # 等待所有进程结束