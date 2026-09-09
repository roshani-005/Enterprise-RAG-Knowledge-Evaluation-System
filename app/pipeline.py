from __future__ import annotations

from app.config import settings
from app.generation import GroundedGenerator
from app.ingestion import chunk_documents, parse_document
from app.retrieval import HybridRetriever
from app.schemas import IngestResponse, QueryResponse


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.generator = GroundedGenerator()

    def ingest(self, filename: str, content: bytes) -> IngestResponse:
        parsed = parse_document(filename, content)
        chunks = chunk_documents(parsed.pages)
        self.retriever.add_documents(chunks)
        return IngestResponse(
            source=filename,
            chunks_indexed=len(chunks),
            used_ocr=parsed.used_ocr,
        )

    def query(self, question: str, top_k: int) -> QueryResponse:
        attempts = 0
        docs = []
        answer = ''
        grounded = False

        while attempts < settings.max_retrieval_attempts:
            attempts += 1
            candidate_k = settings.retrieval_candidate_k * attempts
            docs = self.retriever.retrieve(question, k=top_k, candidate_k=candidate_k)
            answer = self.generator.answer(question, docs)
            grounded = self.generator.validate(question, answer, docs)
            if grounded:
                break

        human_review = not grounded
        return QueryResponse(
            answer=answer,
            citations=self.generator.citations(docs),
            grounded=grounded,
            human_review=human_review,
            retrieval_attempts=attempts,
        )
