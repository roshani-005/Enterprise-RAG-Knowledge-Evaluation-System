# Enterprise RAG Knowledge & Evaluation System

A production-oriented Retrieval-Augmented Generation (RAG) backend for answering questions from company documents with hybrid retrieval, reranking, grounded citations, validation, retry logic, and an evaluation harness.

## Why this project

This is intentionally **not** a generic "chat with PDF" demo. The system separates ingestion, retrieval, generation, validation, and evaluation so each stage can be tested and improved independently.

## Architecture

```text
Company Documents (PDF / DOCX / TXT)
        |
        v
Native Text Extraction
        |
        +-- insufficient text --> OCR fallback
        |
        v
Chunking + Metadata
        |
        v
Embeddings
        |
        v
Vector Store
        |
        +-----------------------------+
                                      |
User Question                         |
        |                             |
        v                             |
Query Embedding                       |
        |                             |
        v                             |
Hybrid Retrieval (semantic + BM25) <--+
        |
        v
Reranking
        |
        v
LLM Generation using retrieved evidence only
        |
        v
Citations
        |
        v
Groundedness Validation
      /   \
   PASS   FAIL
    |      |
 answer   retry with broader retrieval
           |
           v
      still uncertain
           |
           v
      human_review=true
```

## Key capabilities

- PDF, DOCX and TXT ingestion
- OCR fallback for scanned/low-text PDFs
- Recursive chunking with document/page metadata
- Embeddings and vector search
- BM25 keyword retrieval
- Reciprocal Rank Fusion hybrid retrieval
- Reranking stage
- Grounded LLM answer generation
- Source citations
- Post-generation groundedness validation
- Retry path with expanded retrieval
- Human-review flag when evidence remains insufficient
- FastAPI endpoints for ingestion and querying
- Evaluation harness for retrieval quality
- Docker support
- Unit tests and GitHub Actions CI

## Stack

- Python 3.11+
- FastAPI
- LangChain components
- OpenAI-compatible embeddings/chat models
- ChromaDB
- rank-bm25
- PyMuPDF
- python-docx
- Pydantic
- pytest
- Docker

## API

### Health

```http
GET /health
```

### Ingest a document

```http
POST /documents/ingest
Content-Type: multipart/form-data
file=@policy.pdf
```

### Ask a question

```http
POST /query
Content-Type: application/json

{
  "question": "What is the refund policy?",
  "top_k": 5
}
```

Example response:

```json
{
  "answer": "...",
  "citations": [
    {
      "source": "refund_policy.pdf",
      "page": 2,
      "chunk_id": "..."
    }
  ],
  "grounded": true,
  "human_review": false,
  "retrieval_attempts": 1
}
```

## Evaluation

Place verified questions in `eval/data/sample_questions.json` and run:

```bash
python -m eval.run_eval
```

The harness calculates retrieval metrics such as Recall@K and MRR. The included sample dataset is illustrative; do not report benchmark scores until you evaluate against your own verified corpus.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t enterprise-rag .
docker run --env-file .env -p 8000:8000 enterprise-rag
```

## Engineering choices

### OCR is a fallback, not the default

Digital PDFs already contain machine-readable text. Running OCR on all files adds latency and can degrade clean text. This project attempts native extraction first and invokes OCR only when extracted content is below a configurable threshold.

### Hybrid retrieval is explicit

Semantic search and lexical BM25 retrieval are computed separately and fused using Reciprocal Rank Fusion. This makes retrieval behavior inspectable and evaluable instead of hiding everything behind one convenience chain.

### Validation is evidence-based

The generation prompt requires use of retrieved context only. A second validation step checks whether the answer is supported by the retrieved evidence. Failed validation causes one broader retrieval attempt before the system marks the request for human review.

## Scope boundaries

This repository is focused on **RAG and retrieval quality**. It deliberately does not add multi-agent planning, tool-calling workflows, or unrelated automation features. Those belong in a separate agentic system rather than being forced into this project.

## License

For portfolio and educational use.
