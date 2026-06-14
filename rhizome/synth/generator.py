"""Generators propose (question, answer) pairs for a chunk.

The rhizome boundary: the model only *proposes* Q/A here; everything structural
(chunking, distractor sampling, the P-split, citation QC, train/eval split) is
deterministic Python elsewhere. Generators are swappable: a stub for testing, the
Claude-via-MCP-on-subscription path for production now, a local model later.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    answer: str            # CoT, must cite the source verbatim: ##begin_quote##..##end_quote## .. ##Answer:
    golden_chunk_id: str


class Generator:
    def generate(self, chunk) -> list[QAPair]:
        raise NotImplementedError


# ── RAFT generation prompt (shared by the LLM-backed generators) ──────────────
RAFT_GEN_PROMPT = """You are generating training data to teach a model to answer questions \
*grounded in a given passage*. From the passage below, write {n} diverse questions a teammate \
might ask that this passage answers, and for each, a chain-of-thought answer.

Rules for every answer:
- Reason step by step, then quote the EXACT supporting sentence(s) from the passage inside \
##begin_quote## and ##end_quote## (verbatim — copied character-for-character).
- End with "##Answer: <concise final answer>".
- The question must be answerable from THIS passage alone.

Return STRICT JSON: a list of objects with keys "question" and "answer". No prose outside the JSON.

PASSAGE (title: {title}{heading}):
\"\"\"
{text}
\"\"\""""


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?]) ", text) if len(s.strip()) > 30]


class StubGenerator(Generator):
    """Deterministic, LLM-free generator. Emits citation-valid answers (verbatim
    quote from the chunk) so the full pipeline + QC can be exercised offline."""

    def __init__(self, n_questions: int = 2):
        self.n = n_questions

    def generate(self, chunk) -> list[QAPair]:
        sents = _sentences(chunk.text)
        out = []
        for i in range(min(self.n, len(sents))):
            quote = sents[i]
            label = chunk.heading or chunk.title
            q = f"What does the '{label}' section state ({i+1})?"
            a = f"##Reason: The passage states ##begin_quote## {quote} ##end_quote##. ##Answer: {quote}"
            out.append(QAPair(q, a, chunk.id))
        return out


class ClaudeMCPGenerator(Generator):
    """Production generator: Claude on the user's subscription via the Agent SDK.

    Requires `claude-agent-sdk` and a subscription OAuth token
    (`claude setup-token` -> CLAUDE_CODE_OAUTH_TOKEN). NOT the metered API key.
    Wired but gated until that setup + the Agent-SDK-on-subscription path are
    validated on this box (see COMPUTE.md / basis-architecture). Falls back to a
    clear error rather than silently using an API key.
    """

    def __init__(self, n_questions: int = 3, model: str | None = None):
        self.n = n_questions
        self.model = model

    def generate(self, chunk) -> list[QAPair]:
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "ClaudeMCPGenerator needs CLAUDE_CODE_OAUTH_TOKEN (run `claude setup-token` "
                "to use your Claude subscription, not an API key). Use --generator stub to "
                "test the deterministic pipeline offline.")
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pip install claude-agent-sdk to use the Claude generator") from e
        # TODO(phase2-generator): drive query() with RAFT_GEN_PROMPT, parse JSON ->
        # [QAPair]. Kept behind the swappable interface so the deterministic engine
        # ships and is testable now; this lights up once subscription auth is set.
        raise NotImplementedError("Claude-via-MCP generator wiring is the next Phase 2 sub-step")
