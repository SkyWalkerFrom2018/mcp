from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM API request latency',
    ['method', 'endpoint']
) 