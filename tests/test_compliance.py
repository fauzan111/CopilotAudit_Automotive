"""Tests for the Phase 5 compliance layer: hash-chain tamper evidence,
the risk registry, the audited pipeline, retention reporting, and the
report generator.

test_hash_chain_detects_tampering is a permanent regression test for the
exact adversarial check this layer exists to pass: not just that
verify_chain() returns True on clean data, but that it actually catches a
row altered after the fact via direct SQL, the same failure mode a
compromised or careless operator could introduce. It cleans up its own
test row afterward so it doesn't leave the real chain permanently broken.

Run with: pytest tests/test_compliance.py -v
"""

from __future__ import annotations

import uuid

import psycopg

from src.compliance.audit_log import DATABASE_URL, AuditLogEntry, AuditLogger, verify_chain
from src.compliance.pipeline import AuditedDiagnosticPipeline
from src.compliance.report import generate_report
from src.compliance.retention import retention_status
from src.compliance.risk_registry import COMPONENTS, register_components


def test_hash_chain_detects_tampering():
    logger = AuditLogger()
    interaction_id = uuid.uuid4()
    row_hash = logger.log(
        AuditLogEntry(interaction_id=interaction_id, step="query", input_summary="tamper test", confidence=0.5)
    )

    valid, _ = verify_chain()
    assert valid is True

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE audit_log SET confidence = 0.99 WHERE row_hash = %s", (row_hash,))
        conn.commit()

    valid, reason = verify_chain()
    assert valid is False
    assert "altered after being written" in reason

    # Clean up the tampered test row so the real chain is valid again for
    # any later test or real use, this row was the latest at insert time so
    # removing it restores the chain exactly to its pre-test state.
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audit_log WHERE row_hash = %s", (row_hash,))
        conn.commit()

    valid, _ = verify_chain()
    assert valid is True


def test_chain_detects_broken_prev_hash_link():
    """Separate from content tampering: deleting a row out from under its
    successor should also be caught, the next row's prev_hash no longer
    matches anything in the table.
    """
    logger = AuditLogger()
    interaction_id = uuid.uuid4()
    first_hash = logger.log(AuditLogEntry(interaction_id=interaction_id, step="query", input_summary="link test 1"))
    second_hash = logger.log(AuditLogEntry(interaction_id=interaction_id, step="retrieval", input_summary="link test 2"))

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audit_log WHERE row_hash = %s", (first_hash,))
        conn.commit()

    valid, reason = verify_chain()
    assert valid is False
    assert "chain link broken" in reason

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audit_log WHERE row_hash = %s", (second_hash,))
        conn.commit()

    valid, _ = verify_chain()
    assert valid is True


def test_risk_registry_has_both_components_classified():
    register_components()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT component_name, risk_tier, classification_basis FROM ai_system_registry")
            rows = {name: (tier, basis) for name, tier, basis in cur.fetchall()}

    assert len(COMPONENTS) == 2
    for component in COMPONENTS:
        tier, basis = rows[component["component_name"]]
        assert tier == "limited"
        assert basis  # a real justification was recorded, not left blank


def test_retention_status_reports_current_data():
    status = retention_status()
    assert status["total_rows"] >= 0
    assert status["retention_days"] == 180
    if status["total_rows"] > 0:
        # Nothing in this test run is 180 days old yet.
        assert status["rows_past_minimum_retention"] == 0


def test_audited_pipeline_logs_full_interaction_and_keeps_chain_valid():
    pipeline = AuditedDiagnosticPipeline()
    result = pipeline.run("What is P0300 and how do I fix it?")
    assert result.decision == "answered"

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT step FROM audit_log ORDER BY id DESC LIMIT 4"
            )
            steps = {row[0] for row in cur.fetchall()}

    assert steps == {"query", "retrieval", "generation", "guardrail"}

    valid, reason = verify_chain()
    assert valid is True, reason


def test_report_contains_expected_sections():
    report = generate_report()
    assert "Audit Log Integrity" in report
    assert "AI System Risk Registry" in report
    assert "Interaction Log Summary" in report
    assert "Retention" in report
