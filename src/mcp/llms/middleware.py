from starlette.datastructures import MutableHeaders
from fastapi import Request, HTTPException
import time
import logging
import json

logger = logging.getLogger(__name__)

class AuthMiddleware:
    """认证中间件"""
    def __init__(self, app, api_key: str):  # 修正参数接收方式
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        # 仅处理HTTP请求
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        headers = MutableHeaders(scope=scope)
        auth_header = headers.get("Authorization")
        
        if not auth_header or auth_header != f"Bearer {self.api_key}":
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"error": "Unauthorized"}).encode(),
            })
            return
        
        return await self.app(scope, receive, send)

class LoggingMiddleware:
    """日志记录中间件"""
    async def __call__(self, request, call_next):
        # 记录请求开始
        start_time = time.time()
        logger.info(f"Request started: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
        except Exception as e:
            # 记录异常信息
            logger.error(f"Request error: {str(e)}", exc_info=True)
            raise
        
        # 记录请求完成
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"Status: {response.status_code} Duration: {process_time:.2f}s"
        )
        return response 