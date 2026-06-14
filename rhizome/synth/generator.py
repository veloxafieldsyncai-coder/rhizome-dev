"""Generators propose (question, answer) pairs for the RAFT synth.

The rhizome boundary: the model only proposes Q/A here; everything structural
(chunking, tagging, distractors, P split, QC, train/eval split) is deterministic
Python in core.py and pipeline.py.

Layer aware: research chunks yield grounded, verbatim cited answers; brainstorm
chunks yield attributed, provisional answers and an ungrounded flag; and cross
layer examples teach inference over a note while grounding facts in cited research.
Generators are swappable: a stub for offline testing, the Claude via MCP path on
the user's subscription for production, a local model later.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    answer: str             # CoT; research/cross MUST cite verbatim ##begin_quote##..##end_quote##
    golden_ids: list        # chunk ids the answer is grounded in (research and/or note)
    kind: str               # research | note | ungrounded | cross


# ── tiny text helpers ─────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?]) ", _norm(text)) if len(s.strip()) > 30]


def _first_sentence(text: str) -> str:
    s = _sentences(text)
    return s[0] if s else _norm(text)[:200]


def _short(text: str, n_words: int) -> str:
    return " ".join(_norm(text).split()[:n_words])


class Generator:
    """Override the three hooks. Default to emitting nothing."""
    def research_qa(self, chunk) -> list[QAPair]:
        return []

    def note_qa(self, chunk) -> list[QAPair]:
        return []

    def cross_qa(self, note_chunk, research_chunk) -> list[QAPair]:
        return []


# ── RAFT generation prompts (for the LLM backed generators) ───────────────────

RESEARCH_PROMPT = """This passage is VETTED RESEARCH (ground truth). Write {n} questions it \
answers, each with a chain-of-thought answer that quotes the exact supporting sentence(s) \
verbatim inside ##begin_quote## and ##end_quote##, then ends with "##Answer: ...". Assert the \
facts as established. Return STRICT JSON [{{"question","answer"}}]. PASSAGE:\n\"\"\"{text}\"\"\""""

NOTE_PROMPT = """This passage is the USER'S OWN PROVISIONAL NOTES (a hypothesis, not fact). \
Write {n} questions and chain-of-thought answers that ATTRIBUTE the idea to the user ("your \
notes propose ...") and never assert it as established fact. Do NOT quote it as truth. Include \
at least one answer that flags it as ungrounded if no research supports it. Return STRICT JSON \
[{{"question","answer"}}]. PASSAGE:\n\"\"\"{text}\"\"\""""

CROSS_PROMPT = """NOTE (user's provisional idea):\n\"\"\"{note}\"\"\"\n\nRESEARCH (ground truth):\
\n\"\"\"{research}\"\"\"\n\nWrite one question that evaluates the note against the research, and a \
chain-of-thought answer that: attributes the note to the user, grounds every factual claim by \
quoting the RESEARCH verbatim inside ##begin_quote##..##end_quote##, then reasons about whether \
the note holds. End with "##Answer: ...". Return STRICT JSON [{{"question","answer"}}]."""


class StubGenerator(Generator):
    """Deterministic, LLM free generator. Emits QC valid examples of every kind so
    the full layer aware pipeline can be exercised offline."""

    def __init__(self, n_questions: int = 2):
        self.n = n_questions

    def research_qa(self, chunk) -> list[QAPair]:
        out = []
        for i, quote in enumerate(_sentences(chunk.text)[: self.n]):
            q = f"What does the research on '{chunk.title}' establish ({i+1})?"
            a = (f"##Reason: The research states ##begin_quote## {quote} ##end_quote## and this "
                 f"is established fact. ##Answer: {quote}")
            out.append(QAPair(q, a, [chunk.id], "research"))
        return out

    def note_qa(self, chunk) -> list[QAPair]:
        idea = _short(chunk.text, 18)
        attributed = QAPair(
            f"What is my current thinking on '{chunk.title}'?",
            (f"##Reason: This is from your own notes, so I attribute it rather than assert it. "
             f"You propose: {idea}. That is a working hypothesis, not established fact. "
             f"##Answer: Your provisional idea is that {idea}"),
            [chunk.id], "note")
        ungrounded = QAPair(
            f"Is the idea in '{chunk.title}' established fact?",
            ("##Reason: This comes only from your notes with no research grounding in context, "
             "so I will not assert it as fact. ##Answer: No, it is an untested hypothesis from "
             "your notes."),
            [chunk.id], "ungrounded")
        return [attributed, ungrounded]

    def cross_qa(self, note_chunk, research_chunk) -> list[QAPair]:
        idea = _short(note_chunk.text, 16)
        rquote = _first_sentence(research_chunk.text)
        q = f"Does my idea in '{note_chunk.title}' hold up against the research?"
        a = (f"##Reason: Your note proposes: {idea} (your provisional idea, attributed to you). "
             f"The research establishes ##begin_quote## {rquote} ##end_quote##. Connecting them, "
             f"the research grounds part of your idea but the open point in your note still needs "
             f"verification. ##Answer: Plausible but not yet grounded; verify it against the cited research.")
        return [QAPair(q, a, [note_chunk.id, research_chunk.id], "cross")]


class ClaudeMCPGenerator(Generator):
    """Production generator: Claude on the user's subscription via the Agent SDK
    (CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`, NOT a metered API key).
    Wired but gated until that auth path is validated. Falls back to a clear error
    rather than silently using an API key. Uses the layer specific prompts above."""

    def __init__(self, n_questions: int = 3, model: str | None = None):
        self.n = n_questions
        self.model = model

    def _require(self):
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "ClaudeMCPGenerator needs CLAUDE_CODE_OAUTH_TOKEN (run `claude setup-token` to "
                "use your Claude subscription, not an API key). Use --generator stub to test "
                "the deterministic pipeline offline.")
        raise NotImplementedError(
            "Claude via MCP generator wiring is the next Phase 2 sub-step (drive the Agent SDK "
            "with RESEARCH_PROMPT / NOTE_PROMPT / CROSS_PROMPT, parse JSON into QAPair).")

    def research_qa(self, chunk):
        self._require()

    def note_qa(self, chunk):
        self._require()

    def cross_qa(self, note_chunk, research_chunk):
        self._require()
