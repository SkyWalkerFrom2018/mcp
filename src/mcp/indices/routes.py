from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from .document_index import DocumentIndex
from .schemas import SearchRequest, DocumentUpdate, HealthCheckResponse

def create_api_router(index_service: DocumentIndex) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["Document Index"])

    @router.post("/documents/upload")
    async def upload_documents(files: List[UploadFile] = File(...)):
        """批量上传文档接口"""
        try:
            documents = []
            for file in files:
                content = await file.read()
                documents.append({
                    "text": content.decode(),
                    "metadata": {"filename": file.filename}
                })
            index_service.update(documents)
            return {"status": "success", "count": len(documents)}
        except Exception as e:
            raise HTTPException(500, str(e))

    @router.post("/search")
    def search_documents(request: SearchRequest):
        """混合搜索接口"""
        return index_service.search(
            query=request.query,
            keyword_filter=request.keyword_filter,
            max_results=request.max_results
        )

    @router.get("/health")
    def health_check():
        """服务健康检查"""
        return {
            "status": "healthy",
            "indexed_docs": len(index_service.nodes)
        }
    
    return router 