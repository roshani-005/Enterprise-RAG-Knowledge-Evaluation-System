from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import chromadb

from app.config import settings


class HybridRetriever:
    def __init__(self) -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(name=settings.collection_name)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        texts = [d.page_content for d in documents]
        vectors = self.embeddings.embed_documents(texts)
        ids = [d.metadata['chunk_id'] for d in documents]
        metadatas = [d.metadata for d in documents]
        self.collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    def _all_documents(self) -> list[Document]:
        result = self.collection.get(include=['documents', 'metadatas'])
        docs = []
        for text, metadata, chunk_id in zip(result.get('documents', []), result.get('metadatas', []), result.get('ids', [])):
            meta = dict(metadata or {})
            meta.setdefault('chunk_id', chunk_id)
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def _semantic(self, query: str, k: int) -> list[Document]:
        if self.collection.count() == 0:
            return []
        query_vector = self.embeddings.embed_query(query)
        result = self.collection.query(query_embeddings=[query_vector], n_results=min(k, self.collection.count()), include=['documents', 'metadatas'])
        docs: list[Document] = []
        for text, metadata, chunk_id in zip(result['documents'][0], result['metadatas'][0], result['ids'][0]):
            meta = dict(metadata or {})
            meta.setdefault('chunk_id', chunk_id)
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in text.split() if token.strip()]

    def _bm25(self, query: str, k: int) -> list[Document]:
        corpus = self._all_documents()
        if not corpus:
            return []
        tokenized = [self._tokenize(doc.page_content) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def retrieve(self, query: str, k: int, candidate_k: int | None = None) -> list[Document]:
        candidate_k = candidate_k or settings.retrieval_candidate_k
        semantic = self._semantic(query, candidate_k)
        lexical = self._bm25(query, candidate_k)

        # Reciprocal Rank Fusion: robustly combines two rankings without assuming
        # their raw similarity scores are on the same scale.
        fused_scores: dict[str, float] = defaultdict(float)
        by_id: dict[str, Document] = {}
        rrf_k = 60
        for ranking in (semantic, lexical):
            for rank, doc in enumerate(ranking, start=1):
                chunk_id = doc.metadata['chunk_id']
                by_id[chunk_id] = doc
                fused_scores[chunk_id] += 1.0 / (rrf_k + rank)

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [by_id[chunk_id] for chunk_id in ranked_ids[:k]]
