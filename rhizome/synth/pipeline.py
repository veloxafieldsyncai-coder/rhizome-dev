"""Orchestrates the layer aware RAFT synthesis: vault -> train.jsonl + eval.jsonl.

Flow: chunk_vault -> split into research and brainstorm substrates ->
  research chunks  : generator.research_qa, asymmetric P split (20% drop golden to internalize facts)
  brainstorm chunks: generator.note_qa, note ALWAYS kept in context (never memorized as fact)
  cross layer      : for each note, retrieve_related research, generator.cross_qa (both kept)
-> assemble (tag each doc [RESEARCH]/[NOTE]) -> citation QC -> split_by_page.

The asymmetry is the structural guarantee: research can be recalled from weights;
a note can only ever be reasoned over when it is retrieved in front of the model.
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

from .core import (chunk_vault, sample_distractors, retrieve_related, assemble,
                   citation_ok, split_by_page)
from .generator import Generator


def run_synth(vault_dir: str, out_dir: str, generator: Generator, *,
              k_distractors: int = 4, p_golden: float = 0.8, eval_frac: float = 0.1,
              cross_top: int = 1, seed: int = 42) -> dict:
    rng = random.Random(seed)
    chunks = chunk_vault(vault_dir)
    if not chunks:
        raise SystemExit(f"No chunks found in {vault_dir}. Is the vault empty?")
    by_id = {c.id: c for c in chunks}
    research = [c for c in chunks if c.layer == "research"]
    notes = [c for c in chunks if c.layer == "brainstorm"]

    examples, dropped = [], 0

    def emit(qa, present_goldens, distractors, source_page):
        nonlocal dropped
        ex = assemble(qa, present_goldens, distractors, source_page, rng)
        if citation_ok(ex, by_id):
            examples.append(ex)
        else:
            dropped += 1

    # research factual: asymmetric P split (drop golden 20% of the time to internalize)
    for ch in research:
        for qa in generator.research_qa(ch):
            keep = rng.random() < p_golden
            present = [ch] if keep else []
            n_dist = k_distractors if keep else k_distractors + 1
            emit(qa, present, sample_distractors(ch, chunks, n_dist, rng), ch.page_path)

    # note attributed + ungrounded: the note is ALWAYS present (p_golden = 1.0 for notes)
    for ch in notes:
        for qa in generator.note_qa(ch):
            emit(qa, [ch], sample_distractors(ch, chunks, k_distractors, rng), ch.page_path)

    # cross layer inference: note + related research, both kept so the research citation is grounded
    for note in notes:
        for r in retrieve_related(note, research, cross_top):
            for qa in generator.cross_qa(note, r):
                dists = sample_distractors(note, chunks, max(0, k_distractors - 1), rng)
                emit(qa, [note, r], dists, note.page_path)

    train, ev = split_by_page(examples, eval_frac, rng)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "eval.jsonl", ev)

    stats = {
        "chunks": len(chunks),
        "research_chunks": len(research),
        "brainstorm_chunks": len(notes),
        "examples": len(examples),
        "by_kind": dict(Counter(e["kind"] for e in examples)),
        "dropped_failed_qc": dropped,
        "train": len(train),
        "eval": len(ev),
        "params": {"k_distractors": k_distractors, "p_golden": p_golden,
                   "eval_frac": eval_frac, "cross_top": cross_top, "seed": seed,
                   "generator": type(generator).__name__},
    }
    (out / "synth_manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("question", "context", "answer")}) + "\n")
