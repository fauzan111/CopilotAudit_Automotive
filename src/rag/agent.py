"""LangGraph diagnostic assistant: retrieve grounded DTC reference docs,
then generate an answer that cites only what was retrieved.

Two-node graph: retrieve -> generate. Kept deliberately simple for Phase 3;
Phase 4 adds the guardrail node (faithfulness scoring, escalate-or-answer
gate) in front of whatever this returns, rather than building guardrails
and generation in the same pass where they're harder to test independently.

Runs against a local Ollama instance (llama3.2), not an external API, same
data-locality reasoning as the embedding model choice in retrieval.py.
Context window is explicitly bounded (num_ctx=4096): Ollama's default
context for this model is large enough to attempt allocating ~14GB of KV
cache on this machine and fail outright, 4096 tokens is comfortably more
than this use case (a handful of short DTC docs plus one question) needs.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.rag.retrieval import HybridRetriever, RetrievedDoc

OLLAMA_MODEL = "llama3.2"
OLLAMA_NUM_CTX = 4096

SYSTEM_PROMPT = (
    "You are a diagnostic assistant for automotive technicians. Answer only "
    "using the reference documents provided below. If the documents don't "
    "contain enough information to answer, say so explicitly rather than "
    "guessing. Cite the DTC code(s) you drew on in your answer."
)


class AgentState(TypedDict):
    question: str
    retrieved_docs: list[RetrievedDoc]
    answer: str


def _format_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no matching reference documents found)"
    return "\n\n".join(f"--- {d.doc_ref}: {d.title} ---\n{d.content}" for d in docs)


class DiagnosticAgent:
    def __init__(self) -> None:
        self._retriever = HybridRetriever()
        self._llm = ChatOllama(model=OLLAMA_MODEL, num_ctx=OLLAMA_NUM_CTX, temperature=0.1)
        self._graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> AgentState:
        docs = self._retriever.search(state["question"], top_k=5)
        return {**state, "retrieved_docs": docs}

    def _generate_node(self, state: AgentState) -> AgentState:
        context = _format_context(state["retrieved_docs"])
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Reference documents:\n{context}\n\n"
            f"Technician question: {state['question']}\n\n"
            f"Answer:"
        )
        response = self._llm.invoke(prompt)
        return {**state, "answer": response.content}

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def run(self, question: str) -> AgentState:
        return self._graph.invoke({"question": question, "retrieved_docs": [], "answer": ""})
