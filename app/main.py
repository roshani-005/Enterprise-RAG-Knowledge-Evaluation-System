from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.pipeline import RAGPipeline
from app.schemas import IngestResponse, QueryRequest, QueryResponse

app = FastAPI(title=settings.app_name, version='1.0.0')
pipeline = RAGPipeline()


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/documents/ingest', response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail='Uploaded file is empty.')
        return pipeline.ingest(file.filename or 'document', content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/query', response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    return pipeline.query(request.question, request.top_k)
