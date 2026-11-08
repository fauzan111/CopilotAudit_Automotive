"""LLM-as-judge faithfulness scoring: does a generated answer actually stay
grounded in the retrieved reference documents, or does it introduce claims
the documents don't support.

Implements the real RAGAS faithfulness algorithm: split the answer into
individual statements, then verify each statement against the reference
context in its own separate call, rather than one open-ended "find and
check all the claims" call. This isn't a simplification, it's how RAGAS's
faithfulness metric actually works, and it matters here for a concrete
reason this project ran into directly during testing:

A single open-ended "list the claims and check each one" prompt, tried
first, repeatedly produced a specific, well-documented small-model failure
mode: entailment-direction confusion. Tested with llama3.2 3B against a
single-sentence, verbatim-correct answer ("P0300 is Random/Multiple
Cylinder Misfire Detected."), the model explicitly listed content FROM THE
REFERENCE DOCUMENTS ("Likely causes: worn spark plugs...") as claims,
correctly observed they were "not present in the quoted answer", and then
counted that absence AGAINST the answer's faithfulness, checking whether
doc content appears in the answer, the opposite of the actual task
(whether answer content is supported by the docs). This happened
consistently across three different prompt rewrites (snap verdict first,
reasoning-then-verdict, count-based, verbatim-digit-only), so it's a
genuine capability ceiling for a 3B local model on open-ended multi-claim
extraction, not a prompt-wording bug.

Decomposing into one-statement-at-a-time YES/NO checks removes the room for
that confusion by construction: each call hands the model exactly one
pre-identified statement (split by code, not discovered by the model) and
asks a single constrained question, there's no "find the claims" step left
for direction confusion to creep into.

Deliberately not using a dedicated classifier model (e.g. the Laya/
StartLux-Decision family researched earlier) here either: those run on a
transformers/ModernBERT backbone that typically pulls in torch, and this
project has already hit that exact native-library conflict twice (Phase 1's
spaCy, Phase 3's initial sentence-transformers attempt).

Retrieval score (RRF) was tested as a candidate signal for the escalation
gate and rejected: dense vector search always returns its k-nearest
neighbors regardless of true relevance, there's no natural "no match"
output. A genuinely off-topic query (tested: pizza dough) scored in the
exact same RRF range (~0.0154-0.0164) as legitimate semantic-only matches
for in-domain questions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from src.rag.retrieval import RetrievedDoc

OLLAMA_MODEL = "llama3.2"
OLLAMA_NUM_CTX = 4096
# See src/rag/agent.py's identical OLLAMA_BASE_URL comment: None keeps ChatOllama's
# localhost default, set to host.docker.internal only when running in the api container.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")

STATEMENT_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")

STATEMENT_CHECK_TEMPLATE = """Reference documents:
{context}

Statement to verify: "{statement}"

Based ONLY on the reference documents above, is this specific statement supported? A
statement is supported if the reference documents convey the same meaning, even if the
wording differs, paraphrasing is not a reason to say unsupported.

Respond with ONLY one word, YES or NO.
"""


@dataclass
class StatementCheck:
    statement: str
    supported: bool
    raw_response: str


@dataclass
class FaithfulnessVerdict:
    grounded: bool
    statement_checks: list[StatementCheck]
    reasoning: str


def split_into_statements(answer: str) -> list[str]:
    """Split an answer into individual claims to verify separately.

    Splits on newlines FIRST, then on sentence punctuation within each
    line, rather than running a single regex sentence-splitter over the
    whole answer at once. A plain `split on '. '` approach, tried first,
    breaks numbered lists apart incorrectly, e.g. "1. Worn spark plugs\n2.
    Vacuum leak\n3. Low fuel pressure" became fragments like "Vacuum
    leak\n3.", glomming the next item's bare list marker onto the previous
    item's text. List-formatted causes/steps are a normal, expected output
    shape from this agent (seen directly in testing), not an edge case
    worth ignoring.

    A second, separate problem showed up once the line-based split above
    was in place: bare list items like "Vacuum leak" are grammatically
    orphaned once split out on their own, they only mean anything in
    relation to the line that introduced them ("The likely causes of P0300
    are:"). Shown in total isolation, the judge treated an item directly
    present in the reference doc's causes list as unsupported, not because
    it was wrong, but because a two-word noun phrase with no context isn't
    really a checkable claim. Fix: track the most recent colon-terminated
    "intro" line and prepend it to subsequent short list items, so "Vacuum
    leak" becomes "The likely causes of P0300 are: Vacuum leak", a
    statement that can actually be verified on its own.

    Not a full NLP sentence tokenizer (avoids pulling in another dependency
    for a splitting task), but line-aware splitting, context carry-forward,
    leading list-marker stripping, and a minimum-length filter together
    handle this agent's actual output shapes correctly.
    """
    statements: list[str] = []
    intro_context: str | None = None

    for raw_line in answer.strip().splitlines():
        stripped_line = LIST_MARKER_PATTERN.sub("", raw_line).strip()
        if not stripped_line:
            continue

        if stripped_line.endswith(":"):
            intro_context = stripped_line.rstrip(":").strip()
            continue

        for sentence in STATEMENT_SPLIT_PATTERN.split(stripped_line):
            sentence = sentence.strip()
            if not (len(sentence) >= 8 and any(c.isalpha() for c in sentence)):
                continue  # stray fragment, e.g. a bare leftover marker

            # Short, bare list items (no sentence-ending punctuation of
            # their own) following a colon intro get that intro prepended
            # so they're independently checkable. A line that IS a complete
            # sentence on its own means we've moved past the list, clear
            # the intro so it doesn't leak into unrelated later lines, e.g.
            # a trailing "DTC(s) referenced: P0300" line incorrectly
            # inheriting an unrelated "causes are:" intro from several
            # lines earlier before this reset was added.
            if re.search(r"[.!?]$", sentence):
                intro_context = None
            elif intro_context and len(sentence.split()) <= 6:
                sentence = f"{intro_context}: {sentence}"

            statements.append(sentence)

    return statements


class FaithfulnessJudge:
    def __init__(self) -> None:
        self._llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX, temperature=0.0)

    def _check_statement(self, statement: str, context: str) -> StatementCheck:
        # Lexical fast path, checked before calling the LLM at all: if the
        # statement's actual content (stripping any prepended intro
        # context from split_into_statements) appears near-verbatim in the
        # source text, it's supported, no judgment call needed. Added after
        # finding llama3.2 3B unreliably said "NO" to short list items
        # (e.g. "Vacuum leak") even when shown context containing that
        # exact phrase verbatim ("Likely causes: ...; Vacuum leak; ..."),
        # consistently, across repeated tests, a genuine small-model
        # reliability ceiling, not a prompt-wording problem. Mirrors the
        # Phase 3 hybrid-retrieval philosophy: don't rely on one fuzzy
        # signal when a cheap, deterministic, complementary one exists.
        # This only short-circuits the easy, exact-overlap case; anything
        # that doesn't substring-match (paraphrases, and importantly,
        # fabricated specifics like a fake torque spec or success
        # percentage, which won't match anything) still goes to the LLM.
        core = statement.rsplit(": ", 1)[-1].strip().lower()
        if len(core) >= 6 and core in context.lower():
            return StatementCheck(statement=statement, supported=True, raw_response="(lexical match, no LLM call)")

        prompt = STATEMENT_CHECK_TEMPLATE.format(context=context or "(none)", statement=statement)
        response = self._llm.invoke(prompt).content.strip()

        # Fail-safe, not fail-open: only an explicit, unambiguous "YES"
        # counts as supported. Anything else, "NO", hedged, unparseable,
        # is treated as unsupported.
        supported = response.strip().upper().startswith("YES")

        return StatementCheck(statement=statement, supported=supported, raw_response=response)

    def score(self, answer: str, retrieved_docs: list[RetrievedDoc]) -> FaithfulnessVerdict:
        context = "\n\n".join(f"--- {d.doc_ref}: {d.title} ---\n{d.content}" for d in retrieved_docs)
        statements = split_into_statements(answer)

        checks = [self._check_statement(s, context) for s in statements]
        grounded = all(c.supported for c in checks) if checks else False

        unsupported = [c.statement for c in checks if not c.supported]
        reasoning = (
            "All statements supported by the retrieved reference documents."
            if grounded
            else "Unsupported statement(s): " + " | ".join(unsupported)
        )

        return FaithfulnessVerdict(grounded=grounded, statement_checks=checks, reasoning=reasoning)
