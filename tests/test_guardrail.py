"""Tests for the Phase 4 guardrail layer: statement splitting, the
faithfulness judge, and the answer/escalate/block decision.

The adversarial fabricated-claim test and the real-agent-answer test are
not incidental, they're regression tests for two genuine bugs found during
development: an entailment-direction confusion in llama3.2 3B's open-ended
claim extraction (fixed by decomposing into per-statement checks), and a
list-splitting bug that orphaned short list items from their introducing
context (fixed by context carry-forward plus a lexical fast path). See
src/guardrail/faithfulness.py's module docstring and inline comments for
the full history; these tests exist so a future change can't silently
reintroduce either bug.

Run with: pytest tests/test_guardrail.py -v
"""

from src.guardrail.faithfulness import FaithfulnessJudge, split_into_statements
from src.guardrail.gate import GuardrailedDiagnosticPipeline, decide
from src.rag.retrieval import HybridRetriever, RetrievedDoc


# --- split_into_statements: fast, no LLM/DB calls ---


def test_split_handles_plain_sentences():
    statements = split_into_statements("First sentence here. Second sentence here.")
    assert statements == ["First sentence here.", "Second sentence here."]


def test_split_handles_numbered_list_without_orphaning_items():
    answer = (
        "The likely causes are:\n\n1. Worn spark plugs\n2. Vacuum leak\n3. Low fuel pressure\n\n"
        "The typical fix is to check the spark plugs first."
    )
    statements = split_into_statements(answer)

    # Each list item should carry its introducing context, not appear as a
    # bare, context-free fragment like the original splitter produced.
    assert any(s == "The likely causes are: Worn spark plugs" for s in statements)
    assert any(s == "The likely causes are: Vacuum leak" for s in statements)
    assert any(s == "The likely causes are: Low fuel pressure" for s in statements)
    assert "The typical fix is to check the spark plugs first." in statements


def test_split_resets_context_after_a_complete_sentence():
    """A complete sentence after a list should NOT inherit the list's intro
    context, this was a real bug: a trailing unrelated line incorrectly
    inherited an intro from several lines earlier.
    """
    answer = "Causes are:\n\n1. A thing\n2. Another thing\n\nUnrelated closing note here."
    statements = split_into_statements(answer)
    assert "Unrelated closing note here." in statements
    assert not any(s.startswith("Causes are: Unrelated") for s in statements)


def test_split_filters_stray_marker_only_fragments():
    answer = "Real sentence here.\n\n-\n\nAnother real sentence."
    statements = split_into_statements(answer)
    assert "-" not in statements
    assert len(statements) == 2


# --- FaithfulnessJudge: real LLM calls against the real DB, integration tests ---


def test_judge_catches_fabricated_specifics_not_in_reference_docs():
    """Regression test for the entailment-direction confusion bug: a
    previous open-ended prompt design let this exact fabricated answer
    through as 'grounded'.
    """
    retriever = HybridRetriever()
    docs = retriever.search("What is P0300?", top_k=3)

    fabricated = (
        "P0300 indicates a random cylinder misfire. Replace the spark plugs with "
        "NGK Iridium IX plugs torqued to exactly 18 Nm, and the issue will be "
        "resolved in 94 percent of cases according to manufacturer data."
    )

    verdict = FaithfulnessJudge().score(fabricated, docs)
    assert verdict.grounded is False


def test_judge_passes_a_genuinely_grounded_multi_sentence_answer():
    """Regression test for the list-item-orphaning bug: a previous version
    of the splitter/judge pipeline incorrectly escalated this exact
    real-world-shaped answer, flagging 'Vacuum leak' and 'Low fuel
    pressure' as unsupported even though they're verbatim in the source doc.
    """
    retriever = HybridRetriever()
    docs = retriever.search("What is P0300?", top_k=3)

    honest = (
        "P0300 is a code for Random/Multiple Cylinder Misfire Detected. Common causes include "
        "worn spark plugs or ignition coils, a vacuum leak, or low fuel pressure. Checking spark "
        "plug condition and ignition coil resistance first is recommended since they are the most "
        "common cause."
    )

    verdict = FaithfulnessJudge().score(honest, docs)
    assert verdict.grounded is True


# --- decide(): pure decision logic, no agent/LLM round trip needed for "blocked" ---


def test_decide_blocks_when_no_documents_retrieved():
    result = decide("some question", "some answer", retrieved_docs=[], judge=FaithfulnessJudge())
    assert result.decision == "blocked"
    assert result.faithfulness is None


# --- Full pipeline, real agent + real judge ---


def test_pipeline_answers_a_well_grounded_real_question():
    pipeline = GuardrailedDiagnosticPipeline()
    result = pipeline.run("What is P0300 and how do I fix it?")
    assert result.decision == "answered"
    assert result.retrieved_docs[0].doc_ref == "P0300"
