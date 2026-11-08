"""Compliance layer: wraps Phase 4's guardrail pipeline and writes a full
EU AI Act Article 12 provenance trail (query -> retrieval -> generation ->
guardrail) to audit_log for every interaction, via the Phase 5 hash-chain
AuditLogger.

A wrapper around the same agent/judge/decide() pieces GuardrailedDiagnosticPipeline
uses, not a modification of Phase 4 itself: that pipeline stays audit-agnostic
and fully unit-testable without a live DB connection, exactly as built. This
class is the only place DiagnosticAgent, FaithfulnessJudge, and AuditLogger
meet.
"""

from __future__ import annotations

import uuid

from src.compliance.audit_log import AuditLogEntry, AuditLogger
from src.guardrail.faithfulness import FaithfulnessJudge
from src.guardrail.gate import GuardedResult, decide
from src.rag.agent import DiagnosticAgent

OLLAMA_MODEL = "llama3.2"

# See risk_registry.py: automotive_diagnostic_copilot is classified "limited".
RISK_TIER = "limited"


def _faithfulness_score(result: GuardedResult) -> float | None:
    """Fraction of statements the judge found supported, or None when no
    judgment was made at all (the "blocked", no-retrieved-docs path).
    FaithfulnessVerdict only carries a boolean grounded flag plus the list
    of per-statement checks, so this is computed here rather than read off
    the verdict directly.
    """
    if result.faithfulness is None or not result.faithfulness.statement_checks:
        return None
    checks = result.faithfulness.statement_checks
    return sum(1 for c in checks if c.supported) / len(checks)


class AuditedDiagnosticPipeline:
    def __init__(self) -> None:
        self._agent = DiagnosticAgent()
        self._judge = FaithfulnessJudge()
        self._logger = AuditLogger()

    def run(self, question: str) -> GuardedResult:
        interaction_id = uuid.uuid4()

        self._logger.log(
            AuditLogEntry(
                interaction_id=interaction_id,
                step="query",
                input_summary=question,
                risk_tier=RISK_TIER,
            )
        )

        agent_result = self._agent.run(question)
        retrieved_docs = agent_result["retrieved_docs"]
        answer = agent_result["answer"]

        self._logger.log(
            AuditLogEntry(
                interaction_id=interaction_id,
                step="retrieval",
                retrieved_doc_ids=[d.doc_id for d in retrieved_docs],
                risk_tier=RISK_TIER,
            )
        )

        self._logger.log(
            AuditLogEntry(
                interaction_id=interaction_id,
                step="generation",
                model_name=OLLAMA_MODEL,
                output_summary=answer,
                risk_tier=RISK_TIER,
            )
        )

        result = decide(question, answer, retrieved_docs, self._judge)

        self._logger.log(
            AuditLogEntry(
                interaction_id=interaction_id,
                step="guardrail",
                output_summary=result.answer,
                faithfulness_score=_faithfulness_score(result),
                decision=result.decision,
                risk_tier=RISK_TIER,
            )
        )

        return result
