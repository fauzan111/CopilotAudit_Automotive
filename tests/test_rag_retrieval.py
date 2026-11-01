"""Tests for hybrid retrieval and the end-to-end diagnostic agent.

Requires Postgres (with documents embedded via scripts/embed_dtc_reference.py)
and, for the agent test, a running local Ollama instance. These are
integration tests against the real stack, not mocks, consistent with how
Phase 1 and Phase 2 were verified.

Run with: pytest tests/test_rag_retrieval.py -v
"""

from src.rag.agent import DiagnosticAgent
from src.rag.retrieval import HybridRetriever, RetrievedDoc


def test_exact_code_query_surfaces_matching_doc_highest():
    retriever = HybridRetriever()
    results = retriever.search("What is P0300?", top_k=3)
    assert results[0].doc_ref == "P0300"


def test_exact_code_match_scores_higher_than_semantic_only_matches():
    """A query containing an exact DTC code should appear in both the dense
    and sparse result lists, so its fused RRF score should be noticeably
    higher than results that only matched one of the two searches.
    """
    retriever = HybridRetriever()
    results = retriever.search("What is P0300?", top_k=5)
    assert results[0].doc_ref == "P0300"
    assert results[0].rrf_score > results[1].rrf_score * 1.5


def test_semantic_query_with_no_code_mentioned_returns_relevant_doc():
    retriever = HybridRetriever()
    results = retriever.search(
        "My EV won't fully charge and stops partway through a charging session", top_k=5
    )
    doc_refs = {d.doc_ref for d in results}
    # Should surface at least one of the charge/battery-related codes, not
    # an unrelated one like the cylinder misfire or CAN-bus codes.
    assert doc_refs & {"P0A1F", "P0D54", "P0A7F", "P0A80"}


def test_top_k_is_respected():
    retriever = HybridRetriever()
    results = retriever.search("battery", top_k=2)
    assert len(results) <= 2


def test_rrf_fusion_sums_scores_for_docs_in_both_lists():
    """Unit test of the fusion math itself, no DB or network needed."""
    dense = [(1, "P0001", "title1", "content1"), (2, "P0002", "title2", "content2")]
    sparse = [(2, "P0002", "title2", "content2"), (3, "P0003", "title3", "content3")]

    fused = HybridRetriever._fuse(dense, sparse, top_k=3)
    fused_by_id = {d.doc_id: d.rrf_score for d in fused}

    # doc 2 appeared at rank 1 in dense (index 1) and rank 0 in sparse,
    # so its score is the sum of both contributions.
    expected_doc2 = 1 / (60 + 2) + 1 / (60 + 1)
    assert abs(fused_by_id[2] - expected_doc2) < 1e-9
    # doc 1 only appeared in dense at rank 0.
    assert abs(fused_by_id[1] - 1 / (60 + 1)) < 1e-9
    # doc 2 should rank first since it's in both lists.
    assert fused[0].doc_id == 2


def test_agent_answer_cites_the_retrieved_code():
    """One real end-to-end test against the local Ollama instance. Slower
    than the rest of the suite by design, this is the test that actually
    proves retrieval and generation work together, not just in isolation.
    """
    agent = DiagnosticAgent()
    result = agent.run("What is P0300 and how do I fix it?")

    assert result["retrieved_docs"], "expected at least one retrieved doc"
    assert result["retrieved_docs"][0].doc_ref == "P0300"
    assert "P0300" in result["answer"]
