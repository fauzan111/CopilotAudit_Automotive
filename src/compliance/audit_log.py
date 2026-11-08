"""Append-only audit log for EU AI Act Article 12 provenance, with
hash-chain tamper evidence.

Each row's row_hash commits to its own content plus the row_hash of the row
immediately before it in the table (prev_hash). This makes the whole table
a hash chain: altering any row's content after the fact changes that row's
hash, which no longer matches what the NEXT row recorded as its prev_hash,
and every row after that becomes provably inconsistent too. This is the
concrete implementation of the "signed, audit-ready traces" language AI4I's
AI Security Lab uses to describe its own mission, this project has been
explicitly framed around since the EU AI Act compliance research phase.
It's a standard, well-understood integrity technique (an append-only hash
chain, the same idea behind a Merkle chain or a minimal blockchain), not
full PKI/certificate infrastructure, which would be real overkill here.

Known limitation: the chain is maintained by reading the latest row_hash
immediately before each write, which has a race condition under concurrent
writers, two processes could read the same prev_hash and both chain from
it, corrupting the single-linear-chain invariant. Fine for this project's
single-process use; a production version would need a DB-level
serialization guarantee (e.g. SELECT ... FOR UPDATE on a chain-head marker,
or a Postgres sequence-backed approach) to be safe under real concurrency.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GENESIS_HASH = "GENESIS"


def _canonical_json(row: dict) -> str:
    def default(o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"not serializable in audit row: {o!r}")

    return json.dumps(row, sort_keys=True, default=default)


def _compute_row_hash(row: dict, prev_hash: str) -> str:
    payload = _canonical_json(row) + prev_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class AuditLogEntry:
    interaction_id: uuid.UUID
    step: str  # 'query' | 'retrieval' | 'generation' | 'guardrail'
    model_name: str | None = None
    input_summary: str | None = None
    retrieved_doc_ids: list[int] | None = None
    output_summary: str | None = None
    confidence: float | None = None
    faithfulness_score: float | None = None
    decision: str | None = None
    risk_tier: str | None = None


def _row_dict(entry: AuditLogEntry, created_at: datetime) -> dict:
    # Fields explicitly cast to plain float/str/list here, not left as
    # whatever type the caller passed in (e.g. numpy float64), so the exact
    # same dict shape and value types are produced here at write time and
    # in verify_chain()'s reconstruction at read time. A mismatch in either
    # would make every hash check fail even on untampered data.
    return {
        "interaction_id": str(entry.interaction_id),
        "step": entry.step,
        "model_name": entry.model_name,
        "input_summary": entry.input_summary,
        "retrieved_doc_ids": list(entry.retrieved_doc_ids) if entry.retrieved_doc_ids else None,
        "output_summary": entry.output_summary,
        "confidence": float(entry.confidence) if entry.confidence is not None else None,
        "faithfulness_score": float(entry.faithfulness_score) if entry.faithfulness_score is not None else None,
        "decision": entry.decision,
        "risk_tier": entry.risk_tier,
        "created_at": created_at,
    }


class AuditLogger:
    def _latest_hash(self, cur) -> str:
        cur.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        result = cur.fetchone()
        return result[0] if result and result[0] else GENESIS_HASH

    def log(self, entry: AuditLogEntry) -> str:
        """Write one audit_log row, chained to the current latest row.
        Returns the new row's row_hash.
        """
        created_at = datetime.now(timezone.utc)
        row = _row_dict(entry, created_at)

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                prev_hash = self._latest_hash(cur)
                row_hash = _compute_row_hash(row, prev_hash)

                cur.execute(
                    """
                    INSERT INTO audit_log
                        (interaction_id, step, model_name, input_summary, retrieved_doc_ids,
                         output_summary, confidence, faithfulness_score, decision, risk_tier,
                         created_at, row_hash, prev_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry.interaction_id,
                        entry.step,
                        entry.model_name,
                        entry.input_summary,
                        entry.retrieved_doc_ids,
                        entry.output_summary,
                        entry.confidence,
                        entry.faithfulness_score,
                        entry.decision,
                        entry.risk_tier,
                        created_at,
                        row_hash,
                        prev_hash,
                    ),
                )
            conn.commit()

        return row_hash


def verify_chain() -> tuple[bool, str | None]:
    """Walk the whole audit_log table in insertion order and verify the
    hash chain is intact: every row's prev_hash must match the previous
    row's actual row_hash, and every row's own row_hash must match what
    recomputing it from the row's current content produces. Either check
    failing means the log was altered after the fact.

    Returns (is_valid, description_of_first_break_or_None).
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, interaction_id, step, model_name, input_summary, retrieved_doc_ids,
                       output_summary, confidence, faithfulness_score, decision, risk_tier,
                       created_at, row_hash, prev_hash
                FROM audit_log ORDER BY id ASC
                """
            )
            rows = cur.fetchall()

    expected_prev = GENESIS_HASH
    for r in rows:
        (
            row_id,
            interaction_id,
            step,
            model_name,
            input_summary,
            retrieved_doc_ids,
            output_summary,
            confidence,
            faithfulness_score,
            decision,
            risk_tier,
            created_at,
            row_hash,
            prev_hash,
        ) = r

        if prev_hash != expected_prev:
            return False, f"row id={row_id}: prev_hash does not match the previous row's actual hash (chain link broken)"

        row = {
            "interaction_id": str(interaction_id),
            "step": step,
            "model_name": model_name,
            "input_summary": input_summary,
            "retrieved_doc_ids": list(retrieved_doc_ids) if retrieved_doc_ids else None,
            "output_summary": output_summary,
            "confidence": float(confidence) if confidence is not None else None,
            "faithfulness_score": float(faithfulness_score) if faithfulness_score is not None else None,
            "decision": decision,
            "risk_tier": risk_tier,
            "created_at": created_at,
        }
        recomputed = _compute_row_hash(row, prev_hash)
        if recomputed != row_hash:
            return False, f"row id={row_id}: content does not match its recorded row_hash (row was altered after being written)"

        expected_prev = row_hash

    return True, None
