"""RAFT synthesis: vault -> train.jsonl + eval.jsonl.

Two clearly separated phases so any source of Q/A flows through the same engine:

  generate_all(chunks, generator)  : a MODEL proposes Q/A (stub, or a frontier model
                                      via the MCP server's submit_qa, or an agent).
  assemble_dataset(chunks, qas)    : DETERMINISTIC. tags docs [RESEARCH]/[NOTE], applies
                                      the asymmetric P split, runs citation QC, splits by page.

run_synth = generate_all + assemble_dataset (the offline stub path). The MCP server calls
assemble_dataset directly over submitted Q/A. Either way the engine is identical.
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

from .core import (chunk_vault, sample_distractors, retrieve_related, assemble,
                   citation_ok, split_by_page)
from .generator import Generator, QAPair


# ── phase 1: a model proposes Q/A (used by the stub / offline path) ───────────


def generate_all(chunks: list, generator: Generator, cross_top: int = 1) -> list[QAPair]:
    research = [c for c in chunks if c.layer == "research"]
    notes = [c for c in chunks if c.layer == "brainstorm"]
    pairs: list[QAPair] = []
    for ch in research:
        pairs += generator.research_qa(ch)
    for ch in notes:
        pairs += generator.note_qa(ch)
    for note in notes:                       # cross layer: pair each note with related research
        for r in retrieve_related(note, research, cross_top):
            pairs += generator.cross_qa(note, r)
    return pairs


# ── phase 2: deterministic assembly (shared by stub and the MCP submit path) ──


def assemble_dataset(chunks: list, qa_pairs: list[QAPair], out_dir: str, *,
                     k_distractors: int = 4, p_golden: float = 0.8,
                     eval_frac: float = 0.1, seed: int = 42) -> dict:
    rng = random.Random(seed)
    by_id = {c.id: c for c in chunks}
    examples, dropped = [], 0

    for qa in qa_pairs:
        goldens = [by_id[g] for g in qa.golden_ids if g in by_id]
        if not goldens:
            dropped += 1
            continue
        primary = goldens[0]                 # golden_ids[0] sets the source page
        if qa.kind == "research":
            keep = rng.random() < p_golden   # 20% drop golden -> internalize the fact
            present = goldens if keep else []
            n_dist = k_distractors if keep else k_distractors + 1
        else:                                # note / ungrounded / cross: always kept in context
            present = goldens
            n_dist = max(0, k_distractors - (len(goldens) - 1))
        dists = sample_distractors(primary, chunks, n_dist, rng)
        ex = assemble(qa, present, dists, primary.page_path, rng)
        if citation_ok(ex, by_id):
            examples.append(ex)
        else:
            dropped += 1

    train, ev = split_by_page(examples, eval_frac, rng)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "eval.jsonl", ev)

    stats = {
        "chunks": len(chunks),
        "examples": len(examples),
        "by_kind": dict(Counter(e["kind"] for e in examples)),
        "dropped_failed_qc": dropped,
        "train": len(train),
        "eval": len(ev),
        "params": {"k_distractors": k_distractors, "p_golden": p_golden,
                   "eval_frac": eval_frac, "seed": seed},
    }
    (out / "synth_manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def run_synth(vault_dir: str, out_dir: str, generator: Generator, *,
              k_distractors: int = 4, p_golden: float = 0.8, eval_frac: float = 0.1,
              cross_top: int = 1, seed: int = 42) -> dict:
    chunks = chunk_vault(vault_dir)
    if not chunks:
        raise SystemExit(f"No chunks found in {vault_dir}. Is the vault empty?")
    qa_pairs = generate_all(chunks, generator, cross_top)
    stats = assemble_dataset(chunks, qa_pairs, out_dir, k_distractors=k_distractors,
                             p_golden=p_golden, eval_frac=eval_frac, seed=seed)
    stats["research_chunks"] = sum(c.layer == "research" for c in chunks)
    stats["brainstorm_chunks"] = sum(c.layer == "brainstorm" for c in chunks)
    stats["params"]["generator"] = type(generator).__name__
    return stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("question", "context", "answer")}) + "\n")
