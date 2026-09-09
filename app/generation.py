from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.schemas import Citation


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    grounded: bool


class GroundedGenerator:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key or None,
            temperature=0,
        )

    @staticmethod
    def _context(documents: list[Document]) -> str:
        blocks = []
        for index, doc in enumerate(documents, start=1):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page')
            chunk_id = doc.metadata.get('chunk_id', '')
            blocks.append(
                f'[SOURCE {index}] source={source}; page={page}; chunk_id={chunk_id}\n{doc.page_content}'
            )
        return '\n\n'.join(blocks)

    def answer(self, question: str, documents: list[Document]) -> str:
        if not documents:
            return 'I do not have enough evidence in the indexed documents to answer this question.'
        prompt = f'''You are an enterprise knowledge assistant.
Answer the question using ONLY the evidence below.
If the evidence is insufficient, say that you do not have enough evidence.
Do not invent policies, numbers, dates, or procedures.
Keep the answer concise and factual.

Question:
{question}

Evidence:
{self._context(documents)}
'''
        return self.llm.invoke(prompt).content.strip()

    def validate(self, question: str, answer: str, documents: list[Document]) -> bool:
        if not documents:
            return False
        prompt = f'''You are a strict groundedness verifier.
Determine whether every factual claim in the proposed answer is supported by the supplied evidence.
Return JSON only, exactly in this form: {{"grounded": true}} or {{"grounded": false}}.
If the answer adds unsupported facts, return false.
If the answer explicitly says evidence is insufficient, return true only if the evidence indeed does not answer the question.

Question: {question}
Proposed answer: {answer}

Evidence:
{self._context(documents)}
'''
        raw = self.llm.invoke(prompt).content.strip()
        try:
            data = json.loads(raw)
            return bool(data.get('grounded', False))
        except json.JSONDecodeError:
            return False

    @staticmethod
    def citations(documents: list[Document]) -> list[Citation]:
        seen: set[str] = set()
        result: list[Citation] = []
        for doc in documents:
            chunk_id = str(doc.metadata.get('chunk_id', ''))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            result.append(
                Citation(
                    source=str(doc.metadata.get('source', 'unknown')),
                    page=int(doc.metadata['page']) if doc.metadata.get('page') is not None else None,
                    chunk_id=chunk_id,
                )
            )
        return result
