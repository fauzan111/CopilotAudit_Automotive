"""Hybrid retrieval over the documents table: dense (pgvector cosine) +
sparse (Postgres full-text) search, fused with Reciprocal Rank Fusion.

Pure vector search blurs exact-match lookups, a DTC code like "P0A80" is
exactly the kind of token dense embeddings are bad at distinguishing from
similar-looking codes. Full-text search catches the exact code; vector
search catches semantically related questions that never mention a code at
all ("my EV won't charge past half"). RRF combines both rankings without needing to tune a blend weight by hand,
each result's final score is 1/(rank + 60) summed across whichever list(s)
it appeared in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # standard RRF constant; de-emphasizes rank differences deep in either list


@dataclass
class RetrievedDoc:
    doc_id: int
    doc_ref: str
    title: str
    content: str
    rrf_score: float


class HybridRetriever:
    def __init__(self) -> None:
        self._model = TextEmbedding(model_name=EMBEDDING_MODEL)

    def _embed_query(self, query: str) -> list[float]:
        return list(self._model.embed([query]))[0].tolist()

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[RetrievedDoc]:
        query_embedding = self._embed_query(query)

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, doc_ref, title, content
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, candidate_pool),
                )
                dense_results = cur.fetchall()

                cur.execute(
                    """
                    SELECT id, doc_ref, title, content
                    FROM documents
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                    ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC
                    LIMIT %s
                    """,
                    (query, query, candidate_pool),
                )
                sparse_results = cur.fetchall()

        return self._fuse(dense_results, sparse_results, top_k)

    @staticmethod
    def _fuse(dense_results, sparse_results, top_k: int) -> list[RetrievedDoc]:
        scores: dict[int, float] = {}
        docs: dict[int, tuple] = {}

        for rank, row in enumerate(dense_results):
            doc_id = row[0]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs[doc_id] = row

        for rank, row in enumerate(sparse_results):
            doc_id = row[0]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs[doc_id] = row

        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedDoc(
                doc_id=doc_id,
                doc_ref=docs[doc_id][1],
                title=docs[doc_id][2],
                content=docs[doc_id][3],
                rrf_score=scores[doc_id],
            )
            for doc_id in ranked_ids
        ]
