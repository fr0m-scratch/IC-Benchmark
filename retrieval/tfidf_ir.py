#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised TF‑IDF retrieval with optional ICI penalty.
Exports: tfidf_rank(queries, id2doc, ici_map=None, lam=0.0)
"""
import re, math
from typing import Dict, List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SK = True
except Exception:
    SK = False

def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())

def _pure_tfidf_cosine(queries: List[str], docs: List[str]) -> List[List[float]]:
    tok_docs = [_tokenize(d) for d in docs]
    df = {}
    for toks in tok_docs:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    N = len(docs) or 1

    def vec(toks):
        tf = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        out = {}
        length = len(toks) or 1
        for w, c in tf.items():
            idf = math.log((N + 1) / (1 + df.get(w, 0))) + 1.0
            out[w] = (c / length) * idf
        return out

    doc_vecs = [vec(t) for t in tok_docs]

    def cosine(a, b):
        keys = set(a.keys()) | set(b.keys())
        num = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        den_a = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
        den_b = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))
        den = (den_a * den_b) or 1e-12
        return num / den

    sims = []
    for q in queries:
        qv = vec(_tokenize(q))
        sims.append([cosine(qv, dv) for dv in doc_vecs])
    return sims

def tfidf_rank(queries: List[str], id2doc: Dict[str, str], ici_map: Dict[str, float] = None, lam: float = 0.0
              ) -> Tuple[List[str], List[Tuple[List[int], List[int], List[float], List[float]]]]:
    ids = list(id2doc.keys())
    docs = [id2doc[i] for i in ids]

    if SK:
        vectorizer = TfidfVectorizer(stop_words="english")
        doc_matrix = vectorizer.fit_transform(docs)
        query_matrix = vectorizer.transform([q.lower() for q in queries])
        base = cosine_similarity(query_matrix, doc_matrix).tolist()
    else:
        base = _pure_tfidf_cosine(queries, docs)

    adjusted = [row[:] for row in base]
    if ici_map and lam > 0.0:
        ici_vals = [ici_map[i] for i in ids]
        mn, mx = min(ici_vals), max(ici_vals)
        rng = (mx - mn) or 1.0
        ici_norm = [(v - mn) / rng for v in ici_vals]
        for row in adjusted:
            for j, norm in enumerate(ici_norm):
                row[j] = row[j] - lam * norm

    ranked = []
    for row_plain, row_adj in zip(base, adjusted):
        ord_plain = sorted(range(len(ids)), key=lambda j: row_plain[j], reverse=True)
        ord_adj = sorted(range(len(ids)), key=lambda j: row_adj[j], reverse=True)
        ranked.append((ord_plain, ord_adj, row_plain, row_adj))
    return ids, ranked
