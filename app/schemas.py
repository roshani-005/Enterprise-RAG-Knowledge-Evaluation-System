from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    human_review: bool
    retrieval_attempts: int


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int
    used_ocr: bool
