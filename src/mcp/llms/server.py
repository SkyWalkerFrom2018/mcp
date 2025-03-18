from fastapi import FastAPI, Depends
from mcp.llms.routes import router as llm_router
from mcp.llms.middleware import AuthMiddleware, LoggingMiddleware
from mcp.config.llm_config import Config

class LlmsService:
    def __init__(self, config: Config):
        self.config = config
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """创建并配置FastAPI应用实例"""
        app = FastAPI(
            title="LLM微服务",
            version="1.0.0",
            docs_url="/api-docs",
            redoc_url="/alt-docs"
        )
        
        # 中间件
        # app.add_middleware(AuthMiddleware, api_key=self.config.provider_config.api_key)
        #app.add_middleware(LoggingMiddleware)
        
        # 路由
        app.include_router(llm_router, prefix="/api/v1/llm")
        
        # 健康检查
        @app.get("/health")
        async def health_check():
            return {"status": "ok"}
        
        return app

    def start(self, host: str = "0.0.0.0", port: int = 8001):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)