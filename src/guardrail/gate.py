"""Guardrail layer: wraps the Phase 3 diagnostic agent and decides whether
its answer is safe to show directly, or should be escalated to a human
technician.

Deliberately a separate layer wrapping DiagnosticAgent, not a node folded
into its LangGraph, matching the architecture diagram (compliance/audit ->
guardrail -> rag -> forecasting -> pii -> data) and keeping Phase 3's
already-tested retrieve/generate graph untouched.

Decision values match the audit_log.decision column in db/init.sql:
  - "answered":   faithfulness judge confirmed the answer is grounded.
  - "escalated":  the judge found claims the retrieved docs don't support.
  - "blocked":    no reference documents were retrieved at all (distinct
                  from "escalated": there was nothing to even check the
                  answer against, a different failure mode worth being
                  able to tell apart later in the audit log).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.guardrail.faithfulness import FaithfulnessJudge, FaithfulnessVerdict
from src.rag.agent import AgentState, DiagnosticAgent
from src.rag.retrieval import RetrievedDoc

ESCALATION_MESSAGE = (
    "I can't confidently ground an answer to this in the available reference "
    "documentation. Please route this to a technician for manual review "
    "rather than relying on my answer as-is."
)


@dataclass
class GuardedResult:
    question: str
    answer: str
    retrieved_docs: list[RetrievedDoc]
    decision: str  # "answered" | "escalated" | "blocked"
    faithfulness: FaithfulnessVerdict | None


def decide(
    question: str,
    answer: str,
    retrieved_docs: list[RetrievedDoc],
    judge: FaithfulnessJudge,
) -> GuardedResult:
    """Pure decision logic, separated from DiagnosticAgent's run() call so
    the "blocked" (no retrieved docs) path can be tested directly with a
    synthetic empty list, without needing a live agent/Ollama round trip.
    In practice, dense search always returns its k-nearest neighbors as
    long as the documents table has any rows, so this path can't easily be
    forced through a real end-to-end call.
    """
    if not retrieved_docs:
        return GuardedResult(
            question=question,
            answer=ESCALATION_MESSAGE,
            retrieved_docs=retrieved_docs,
            decision="blocked",
            faithfulness=None,
        )

    verdict = judge.score(answer, retrieved_docs)

    if not verdict.grounded:
        return GuardedResult(
            question=question,
            answer=ESCALATION_MESSAGE,
            retrieved_docs=retrieved_docs,
            decision="escalated",
            faithfulness=verdict,
        )

    return GuardedResult(
        question=question,
        answer=answer,
        retrieved_docs=retrieved_docs,
        decision="answered",
        faithfulness=verdict,
    )


class GuardrailedDiagnosticPipeline:
    def __init__(self) -> None:
        self._agent = DiagnosticAgent()
        self._judge = FaithfulnessJudge()

    def run(self, question: str) -> GuardedResult:
        agent_result: AgentState = self._agent.run(question)
        return decide(question, agent_result["answer"], agent_result["retrieved_docs"], self._judge)
