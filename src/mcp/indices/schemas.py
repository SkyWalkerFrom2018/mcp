from pydantic import BaseModel
from typing import Optional, Dict, Any

class SearchRequest(BaseModel):
    query: str
    keyword_filter: Optional[Dict] = None
    max_results: int = 10

class DocumentUpdate(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any]

class HealthCheckResponse(BaseModel):
    status: str
    indexed_docs: int 