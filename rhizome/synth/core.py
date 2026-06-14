"""Deterministic RAFT-synth stages: chunk, distractors, assemble (P-split), citation QC, split.

Pure stdlib. The model proposes Q/A (see generator.py); everything here is deterministic
and controls the structure of the training data.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── 1 · Chunking ──────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    id: str            # "<page-path>#<section-slug>"
    page_path: str
    title: str
    heading: str
    text: str
    category: str = ""


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            meta = {}
            for line in raw[3:end].splitlines():
                m = re.match(r"^(\w+):\s*(.*)$", line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
            return meta, raw[end + 4:]
    return {}, raw


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "section"


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split on ## (H2) headings. Content before the first H2 is the intro section."""
    sections, heading, cur = [], "", []
    for line in body.splitlines():
        if re.match(r"^##\s", line):
            if cur:
                sections.append((heading, "\n".join(cur)))
            heading, cur = re.sub(r"^#+\s*", "", line).strip(), []
        else:
            cur.append(line)
    if cur:
        sections.append((heading, "\n".join(cur)))
    return sections


def chunk_vault(vault_dir: str, min_chars: int = 120) -> list[Chunk]:
    vault = Path(vault_dir)
    chunks: list[Chunk] = []
    for p in sorted(vault.rglob("*.md")):
        if p.name in {"index.md", "README.md"} or p.name.startswith("_"):
            continue
        rel = str(p.relative_to(vault))
        meta, body = _parse_frontmatter(p.read_text())
        title = meta.get("title", p.stem)
        category = meta.get("category", p.parent.name)
        for heading, text in _split_sections(body):
            text = text.strip()
            if len(text) < min_chars:
                continue
            cid = f"{rel}#{_slug(heading) if heading else 'intro'}"
            chunks.append(Chunk(cid, rel, title, heading, text, category))
    return chunks


# ── 4 · Distractor sampling (hard negatives + random) ─────────────────────────


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def sample_distractors(golden: Chunk, pool: list[Chunk], k: int, rng, hard_frac: float = 0.5):
    """k distractors from OTHER pages: top lexical-overlap as hard negatives, rest random."""
    cands = [c for c in pool if c.page_path != golden.page_path]
    if not cands:
        return []
    gtok = _tokens(golden.text)
    ranked = sorted(cands, key=lambda c: len(gtok & _tokens(c.text)), reverse=True)
    n_hard = min(int(round(k * hard_frac)), len(ranked))
    hard, rest = ranked[:n_hard], ranked[n_hard:]
    rng.shuffle(rest)
    return (hard + rest)[:k]


# ── 5 · Assemble (P-split) ────────────────────────────────────────────────────


def assemble_example(qa, golden: Chunk, distractors, keep_golden: bool, rng) -> dict:
    docs = list(distractors) + ([golden] if keep_golden else [])
    rng.shuffle(docs)  # position must not leak the answer
    context = "\n".join(f"[DOC{i+1}] {d.text}" for i, d in enumerate(docs))
    return {
        "question": qa.question,
        "context": context,
        "answer": qa.answer,
        "golden_id": golden.id,
        "has_golden": keep_golden,
        "source_page": golden.page_path,
    }


# ── 6 · Citation QC ───────────────────────────────────────────────────────────

_QUOTE_RE = re.compile(r"##begin_quote##(.*?)##end_quote##", re.DOTALL)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def citation_ok(example: dict, golden_text: str) -> bool:
    """Every quoted span must appear verbatim in the golden chunk — kills hallucinated
    citations with a deterministic string match (checked for golden-included AND
    distractor-only examples, since the answer cites the real source either way)."""
    quotes = _QUOTE_RE.findall(example["answer"])
    if not quotes:
        return False
    gt = _norm(golden_text)
    return all(_norm(q) in gt for q in quotes)


# ── 7 · Split by source page (no leak) ────────────────────────────────────────


def split_by_page(examples: list[dict], eval_frac: float, rng) -> tuple[list, list]:
    pages = sorted({e["source_page"] for e in examples})
    if len(pages) <= 1:
        return examples, []  # too small to hold out without leaking
    rng.shuffle(pages)
    n_eval = max(1, round(len(pages) * eval_frac))
    eval_pages = set(pages[:n_eval])
    train = [e for e in examples if e["source_page"] not in eval_pages]
    ev = [e for e in examples if e["source_page"] in eval_pages]
    return train, ev
