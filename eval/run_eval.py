from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from app.retrieval import HybridRetriever


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids) & relevant_ids)
    return hits / len(relevant_ids)


def run(dataset_path: str = 'eval/data/sample_questions.json', k: int = 5) -> dict[str, float]:
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8'))
    retriever = HybridRetriever()
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for item in dataset:
        docs = retriever.retrieve(item['question'], k=k)
        retrieved_ids = [str(doc.metadata.get('chunk_id', '')) for doc in docs]
        relevant_ids = set(item.get('relevant_chunk_ids', []))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids))
        reciprocal_ranks.append(reciprocal_rank(retrieved_ids, relevant_ids))

    metrics = {
        f'recall@{k}': mean(recalls) if recalls else 0.0,
        'mrr': mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        'questions': float(len(dataset)),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    run()
