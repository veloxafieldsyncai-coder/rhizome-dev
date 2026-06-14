"""Deterministic RAFT synth stages: chunk, distractors, cross layer retrieval,
assemble (tagged context), citation QC, page split.

Pure stdlib. The model proposes Q/A (see generator.py); everything here is deterministic
and controls the structure of the training data, including how the two substrates
(research vs brainstorm) are tagged and laid out.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

# layer value -> the tag the model reads in context. This tag is the input signal
# that lets the trained model tell ground truth from the user's provisional notes.
LAYER_TAG = {"research": "RESEARCH", "brainstorm": "NOTE"}

# ── 1 · Chunking ──────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    id: str            # "<page-path>#<section-slug>"
    page_path: str
    title: str
    heading: str
    text: str
    layer: str         # research | brainstorm
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
    """Split on H2 (##) headings. Content before the first H2 is the intro section."""
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


def _layer_of(meta: dict, rel_path: str) -> str:
    """Frontmatter layer wins; else pages under notes/ are brainstorm, else research."""
    lay = meta.get("layer", "").lower()
    if lay in ("research", "brainstorm"):
        return lay
    return "brainstorm" if rel_path.startswith("notes/") else "research"


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
        layer = _layer_of(meta, rel)
        for heading, text in _split_sections(body):
            text = text.strip()
            if len(text) < min_chars:
                continue
            cid = f"{rel}#{_slug(heading) if heading else 'intro'}"
            chunks.append(Chunk(cid, rel, title, heading, text, layer, category))
    return chunks


# ── token overlap, used for distractors and cross layer retrieval ─────────────


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def sample_distractors(golden: Chunk, pool: list[Chunk], k: int, rng, hard_frac: float = 0.5):
    """k distractors from OTHER pages: top lexical overlap as hard negatives, rest random."""
    cands = [c for c in pool if c.page_path != golden.page_path]
    if not cands:
        return []
    gtok = _tokens(golden.text)
    ranked = sorted(cands, key=lambda c: len(gtok & _tokens(c.text)), reverse=True)
    n_hard = min(int(round(k * hard_frac)), len(ranked))
    hard, rest = ranked[:n_hard], ranked[n_hard:]
    rng.shuffle(rest)
    return (hard + rest)[:k]


def retrieve_related(query: Chunk, pool: list[Chunk], top: int) -> list[Chunk]:
    """Top research chunks lexically related to a note, for cross layer pairing."""
    qtok = _tokens(query.text)
    scored = [(len(qtok & _tokens(c.text)), c) for c in pool if c.page_path != query.page_path]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for score, c in scored[:top] if score > 0]


# ── assemble (tagged context) ─────────────────────────────────────────────────


def assemble(qa, present_goldens: list[Chunk], distractors: list[Chunk],
            source_page: str, rng) -> dict:
    """Lay out present goldens + distractors, each prefixed with its layer tag
    [RESEARCH] or [NOTE], shuffled so position never leaks the answer."""
    docs = list(present_goldens) + list(distractors)
    rng.shuffle(docs)
    context = "\n".join(f"[{LAYER_TAG.get(d.layer, d.layer.upper())}] {d.text}" for d in docs)
    return {
        "question": qa.question,
        "context": context,
        "answer": qa.answer,
        "golden_ids": list(qa.golden_ids),
        "kind": qa.kind,
        "source_page": source_page,
    }


# ── citation QC ───────────────────────────────────────────────────────────────

_QUOTE_RE = re.compile(r"##begin_quote##(.*?)##end_quote##", re.DOTALL)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def citation_ok(example: dict, by_id: dict) -> bool:
    """Verbatim guarantee: any quoted span must appear verbatim in one of the
    example's golden chunks. Research examples MUST carry a quote (grounding);
    note examples may paraphrase with no quote, so verbatim is reserved for research."""
    quotes = _QUOTE_RE.findall(example["answer"])
    if example["kind"] == "research" and not quotes:
        return False
    if not quotes:
        return True
    gt = _norm(" ".join(by_id[g].text for g in example["golden_ids"] if g in by_id))
    return all(_norm(q) in gt for q in quotes)


# ── split by source page (no leak) ───────────────────────────────────────────


def split_by_page(examples: list[dict], eval_frac: float, rng) -> tuple[list, list]:
    pages = sorted({e["source_page"] for e in examples})
    if len(pages) <= 1:
        return examples, []
    rng.shuffle(pages)
    n_eval = max(1, round(len(pages) * eval_frac))
    eval_pages = set(pages[:n_eval])
    train = [e for e in examples if e["source_page"] not in eval_pages]
    ev = [e for e in examples if e["source_page"] in eval_pages]
    return train, ev
