"""Retention-policy awareness for the audit log.

EU AI Act Article 12 sets AUDIT_LOG_RETENTION_DAYS (180, in .env) as a
MINIMUM retention period, logs must be kept at least that long. It is not
a mandate to delete them afterward. This module reports rows that have
aged past that minimum, candidates an organization's own data-retention
policy might choose to prune, it deliberately does not delete anything.

Why deletion isn't implemented here: this audit log is a single linear
hash chain anchored at GENESIS_HASH on its first row. Deleting early rows,
even ones that have already satisfied the minimum retention period,
breaks verify_chain() for every row after them: the next surviving row's
prev_hash would point to a hash that no longer exists in the table, which
verify_chain() is specifically built to flag as "chain link broken." A
production system that needs real pruning would have to anchor pruned
segments with a signed checkpoint (e.g. "rows 1-500 purged, their last
hash was X, the chain continues from X instead of GENESIS") so
verify_chain can resume from that checkpoint. That's real added
complexity this project doesn't need just to demonstrate retention
awareness, so it's left as a documented limitation rather than a
half-built deletion stub that would quietly defeat the tamper-evidence
this whole layer exists for.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
RETENTION_DAYS = int(os.environ.get("AUDIT_LOG_RETENTION_DAYS", "180"))


def rows_past_minimum_retention() -> list[dict]:
    """Rows older than RETENTION_DAYS, i.e. rows where the Article 12
    minimum retention period has already been satisfied. Reporting only,
    see module docstring for why this doesn't delete anything.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, interaction_id, step, created_at
                FROM audit_log
                WHERE created_at < now() - (%s || ' days')::interval
                ORDER BY id
                """,
                (RETENTION_DAYS,),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def retention_status() -> dict:
    """Summary used by the compliance report: how much history exists, and
    whether any of it has crossed the minimum retention threshold yet.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT min(created_at), max(created_at), count(*) FROM audit_log")
            oldest, _newest, total = cur.fetchone()

    if oldest is None:
        return {
            "retention_days": RETENTION_DAYS,
            "total_rows": 0,
            "oldest_row_age_days": None,
            "rows_past_minimum_retention": 0,
        }

    age_days = (datetime.now(timezone.utc) - oldest).days
    return {
        "retention_days": RETENTION_DAYS,
        "total_rows": total,
        "oldest_row_age_days": age_days,
        "rows_past_minimum_retention": len(rows_past_minimum_retention()),
    }


if __name__ == "__main__":
    status = retention_status()
    print(f"Retention policy minimum: {status['retention_days']} days")
    print(f"Total audit_log rows: {status['total_rows']}")
    print(f"Oldest row age: {status['oldest_row_age_days']} day(s)")
    print(
        "Rows past minimum retention (purge candidates, not auto-deleted): "
        f"{status['rows_past_minimum_retention']}"
    )
