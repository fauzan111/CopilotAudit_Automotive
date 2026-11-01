"""Embed the DTC reference docs and load them into pgvector.

Uses fastembed (ONNX Runtime, no torch) with BAAI/bge-small-en-v1.5, a
384-dimension model matching the documents.embedding column's VECTOR(384)
type defined in db/init.sql. Each DTC record's title, description, likely
causes, and typical fix are concatenated into one chunk, since they're short
and belong together, splitting them would just force the retriever to
reassemble context that was never meaningfully separable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "dtc_reference.jsonl"
DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _record_to_text(record: dict) -> str:
    causes = "; ".join(record["likely_causes"])
    return (
        f"{record['code']}: {record['title']}\n"
        f"{record['description']}\n"
        f"Likely causes: {causes}\n"
        f"Typical fix: {record['typical_fix']}"
    )


def main() -> None:
    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    records = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    texts = [_record_to_text(r) for r in records]
    embeddings = list(model.embed(texts))

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Idempotent: clear prior runs of this source before reloading,
            # so re-running this script after editing seed_dtc_reference.py
            # doesn't accumulate duplicate rows.
            cur.execute("DELETE FROM documents WHERE source = %s", ("obd2_dtc_reference",))
            for record, text, embedding in zip(records, texts, embeddings):
                cur.execute(
                    """
                    INSERT INTO documents (source, doc_ref, title, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        "obd2_dtc_reference",
                        record["code"],
                        record["title"],
                        text,
                        list(embedding),
                    ),
                )
        conn.commit()

    print(f"Embedded and loaded {len(records)} DTC reference documents (model: {EMBEDDING_MODEL}, dim: {len(embeddings[0])}).")


if __name__ == "__main__":
    main()
