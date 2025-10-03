"""Lightweight retrieval and reranking over tool cards."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analysis.ic_score import ToolICResult, ToolRecord


@dataclass
class RetrievalConfig:
    k: int = 8
    ic_penalty_mu: float = 0.6
    redundancy_penalty_nu: float = 0.3


@dataclass
class ToolDocument:
    tool: ToolRecord
    ic_score: float
    text: str
    vector: Dict[str, float]
    norm: float


@dataclass
class RetrievedTool:
    tool: ToolRecord
    similarity: float
    ic_score: float
    redundancy: float
    score: float


def _tokenise(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def _tf(tokens: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    total = 0.0
    for tok in tokens:
        total += 1.0
        counts[tok] = counts.get(tok, 0.0) + 1.0
    if total == 0:
        return counts
    return {tok: freq / total for tok, freq in counts.items()}


class ToolRetriever:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.documents: List[ToolDocument] = []
        self.idf: Dict[str, float] = {}

    def build(self, tools: Sequence[ToolRecord], ic_results: Dict[str, ToolICResult]) -> None:
        documents: List[ToolDocument] = []
        doc_freq: Dict[str, int] = {}
        for tool in tools:
            text = self._make_text(tool)
            tokens = _tokenise(text)
            tf = _tf(tokens)
            for tok in tf:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
            ic = ic_results.get(tool.tool_id)
            documents.append(
                ToolDocument(
                    tool=tool,
                    ic_score=ic.score if ic else 0.0,
                    text=text,
                    vector=tf,
                    norm=0.0,
                )
            )
        total_docs = max(len(documents), 1)
        idf = {tok: math.log(total_docs / (freq + 1)) + 1.0 for tok, freq in doc_freq.items()}
        # Store weighted vectors and norms
        for doc in documents:
            weighted = {tok: weight * idf.get(tok, 1.0) for tok, weight in doc.vector.items()}
            norm = math.sqrt(sum(val * val for val in weighted.values())) or 1.0
            doc.vector = weighted
            doc.norm = norm
        self.documents = documents
        self.idf = idf

    def _make_text(self, tool: ToolRecord) -> str:
        schema = tool.input_schema or {}
        props = schema.get("properties") or {}
        prop_names = " ".join(props.keys())
        required = " ".join(schema.get("required") or [])
        return f"{tool.name} {tool.description} {prop_names} required {required}"

    def query(self, text: str, k: Optional[int] = None) -> List[RetrievedTool]:
        if not self.documents:
            return []
        k = k or self.config.k
        query_tokens = _tokenise(text)
        query_vector = self._make_query_vector(query_tokens)
        query_norm = math.sqrt(sum(val * val for val in query_vector.values())) or 1.0

        scored: List[Tuple[ToolDocument, float]] = []
        for doc in self.documents:
            dot = sum(query_vector.get(tok, 0.0) * doc.vector.get(tok, 0.0) for tok in query_vector)
            similarity = dot / (query_norm * doc.norm)
            scored.append((doc, similarity))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_docs = scored[:k]
        redundancies = self._compute_redundancy([doc for doc, _ in top_docs])

        results: List[RetrievedTool] = []
        for (doc, sim), redundancy in zip(top_docs, redundancies):
            adjusted = sim - self.config.ic_penalty_mu * doc.ic_score - self.config.redundancy_penalty_nu * redundancy
            results.append(
                RetrievedTool(
                    tool=doc.tool,
                    similarity=sim,
                    ic_score=doc.ic_score,
                    redundancy=redundancy,
                    score=adjusted,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _make_query_vector(self, tokens: Iterable[str]) -> Dict[str, float]:
        tf = _tf(tokens)
        return {tok: weight * self.idf.get(tok, 1.0) for tok, weight in tf.items()}

    def _compute_redundancy(self, docs: Sequence[ToolDocument]) -> List[float]:
        redundancies: List[float] = []
        for idx, anchor in enumerate(docs):
            if len(docs) == 1:
                redundancies.append(0.0)
                continue
            sims = []
            for jdx, other in enumerate(docs):
                if idx == jdx:
                    continue
                dot = sum(anchor.vector.get(tok, 0.0) * other.vector.get(tok, 0.0) for tok in anchor.vector)
                sim = dot / (anchor.norm * other.norm)
                sims.append(sim)
            redundancies.append(sum(sims) / len(sims) if sims else 0.0)
        return redundancies


__all__ = [
    "RetrievalConfig",
    "ToolRetriever",
    "RetrievedTool",
]
