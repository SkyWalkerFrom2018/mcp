class LLMError(Exception):
    """LLM相关异常基类"""
    pass

class LLMRequestError(LLMError):
    """通用请求异常"""
    def __init__(self, message: str):
        super().__init__(f"LLM request failed: {message}")

class LLMRateLimitError(LLMError):
    """API速率限制异常"""
    def __init__(self, message: str = "API rate limit exceeded"):
        super().__init__(message)

class LLMTimeoutError(LLMError):
    """请求超时异常"""
    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)

class LLMAuthenticationError(LLMError):
    """认证失败异常"""
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message)
