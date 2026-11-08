"""Audit report generator: turns the hash-chain audit_log and the
ai_system_registry into a single human-readable compliance report.

This is the "signed, audit-ready traces" artifact this project has been
framed around from the start, the kind of document a compliance reviewer
(or AI4I's AI Security Lab, the real audience this project pitches to)
would actually want handed to them: not raw rows, a report that states the
chain's integrity status up front, then the risk inventory, then a
per-interaction summary.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv

from src.compliance.audit_log import verify_chain
from src.compliance.retention import retention_status

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


def fetch_registry() -> list[dict]:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT component_name, risk_tier, classification_basis, last_reviewed
                FROM ai_system_registry ORDER BY component_name
                """
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_interactions() -> list[dict]:
    """One summary row per interaction_id: its final decision and the step
    count logged for it, the level of detail a reviewer skimming the report
    needs, not the full row-by-row trace (that's what the raw table is for).
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT interaction_id,
                       min(created_at) AS started_at,
                       count(*) AS step_count,
                       max(CASE WHEN step = 'guardrail' THEN decision END) AS decision,
                       max(CASE WHEN step = 'guardrail' THEN faithfulness_score END) AS faithfulness_score,
                       max(risk_tier) AS risk_tier
                FROM audit_log
                GROUP BY interaction_id
                ORDER BY started_at
                """
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def generate_report() -> str:
    """Build the report as a markdown string, print-and-redirect-friendly
    rather than requiring a separate file-writing entry point.
    """
    chain_valid, chain_reason = verify_chain()
    registry = fetch_registry()
    interactions = _fetch_interactions()
    generated_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# CopilotAudit_Automotive -- EU AI Act Compliance Report")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    lines.append("## 1. Audit Log Integrity")
    if chain_valid:
        lines.append(f"Hash chain verified intact across {len(interactions)} interaction(s). No tampering detected.")
    else:
        lines.append(f"**INTEGRITY FAILURE**: {chain_reason}")
    lines.append("")

    lines.append("## 2. AI System Risk Registry (Article 6 / Annex III)")
    for c in registry:
        lines.append(f"### {c['component_name']} -- risk tier: {c['risk_tier']}")
        lines.append(c["classification_basis"] or "(no basis recorded)")
        lines.append(f"Last reviewed: {c['last_reviewed'].isoformat()}")
        lines.append("")

    lines.append("## 3. Interaction Log Summary (Article 12)")
    if not interactions:
        lines.append("No interactions logged yet.")
    else:
        lines.append("| Interaction | Started | Steps | Decision | Faithfulness | Risk tier |")
        lines.append("|---|---|---|---|---|---|")
        for i in interactions:
            faith = f"{i['faithfulness_score']:.2f}" if i["faithfulness_score"] is not None else "n/a"
            lines.append(
                f"| {str(i['interaction_id'])[:8]} | {i['started_at'].isoformat()} | "
                f"{i['step_count']} | {i['decision'] or 'n/a'} | {faith} | {i['risk_tier'] or 'n/a'} |"
            )
    lines.append("")

    decisions = [i["decision"] for i in interactions if i["decision"]]
    if decisions:
        lines.append("## 4. Decision Breakdown")
        for value in sorted(set(decisions)):
            lines.append(f"- {value}: {decisions.count(value)}")
        lines.append("")

    retention = retention_status()
    lines.append("## 5. Retention (Article 12 minimum)")
    lines.append(f"Minimum retention policy: {retention['retention_days']} days.")
    lines.append(f"Total rows: {retention['total_rows']}, oldest row age: {retention['oldest_row_age_days']} day(s).")
    lines.append(f"Rows past minimum retention (purge candidates, not auto-deleted): {retention['rows_past_minimum_retention']}.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_report())
