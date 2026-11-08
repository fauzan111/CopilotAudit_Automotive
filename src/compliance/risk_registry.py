"""EU AI Act risk-tier inventory for this project's AI components.

Populates ai_system_registry: one row per logical AI system, not per
interaction (that's what audit_log is for). The EU AI Act compliance
research behind this project flagged that most enterprises don't even
have this inventory, classifying each component under Annex III before
anyone asks is the actual deliverable here, not just the audit log.

Both components below are classified "limited" risk, not "high_risk":
neither makes an autonomous safety-critical decision. The diagnostic
copilot is explicitly human-in-the-loop (Phase 4's guardrail escalates
to a technician rather than acting on an ungrounded answer), and the
forecaster only predicts a maintenance figure, it doesn't control the
vehicle or gate any safety system. Annex III's high-risk categories
(biometric ID, critical infrastructure safety components, employment
decisions, law enforcement, etc.) don't cover either use case. This
classification itself needs to be recorded with its reasoning, not just
asserted, that's the whole point of the registry.
"""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

COMPONENTS = [
    {
        "component_name": "battery_degradation_forecaster",
        "description": (
            "XGBoost regression model predicting battery capacity fade (%) from "
            "discharge-cycle telemetry, used to flag vehicles approaching end-of-life "
            "battery health for scheduled maintenance."
        ),
        "risk_tier": "limited",
        "classification_basis": (
            "Not Annex III high-risk: output is an advisory maintenance figure, not an "
            "autonomous safety or access-control decision, and the model doesn't control "
            "any vehicle system. Classified 'limited' rather than 'minimal' because its "
            "output does influence a real maintenance recommendation shown to a human; "
            "Article 13 transparency (users should know this figure is model-generated) "
            "applies. If this component's output were later wired directly into an "
            "automatic safety shutoff or warranty-denial decision, it would need "
            "re-classification toward high_risk."
        ),
    },
    {
        "component_name": "automotive_diagnostic_copilot",
        "description": (
            "RAG-based assistant answering vehicle fault/DTC diagnostic questions, grounded "
            "in reference documentation, gated by a faithfulness guardrail that escalates "
            "ungrounded answers to a human technician instead of displaying them."
        ),
        "risk_tier": "limited",
        "classification_basis": (
            "Not Annex III high-risk: it advises on vehicle diagnostics, it does not "
            "operate a safety component itself, and Phase 4's guardrail enforces "
            "human-in-the-loop review by escalating any answer the faithfulness judge "
            "can't ground in retrieved reference docs, rather than letting the model act "
            "or respond unsupervised in that case. Classified 'limited' (not 'minimal') "
            "because it's a user-facing generative system making domain-specific claims; "
            "Article 13 transparency and Article 12 logging obligations apply, which is "
            "exactly what the audit_log table and this registry exist to satisfy."
        ),
    },
]


def register_components() -> None:
    """Idempotent upsert of the known components into ai_system_registry."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for c in COMPONENTS:
                cur.execute(
                    """
                    INSERT INTO ai_system_registry
                        (component_name, description, risk_tier, classification_basis)
                    VALUES (%(component_name)s, %(description)s, %(risk_tier)s, %(classification_basis)s)
                    ON CONFLICT (component_name) DO UPDATE SET
                        description = EXCLUDED.description,
                        risk_tier = EXCLUDED.risk_tier,
                        classification_basis = EXCLUDED.classification_basis,
                        last_reviewed = now()
                    """,
                    c,
                )
        conn.commit()


if __name__ == "__main__":
    register_components()
    print(f"Registered {len(COMPONENTS)} components in ai_system_registry.")
