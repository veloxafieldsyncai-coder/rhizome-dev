"""Orchestrates the 7-stage RAFT synthesis: vault -> train.jsonl + eval.jsonl.

vault → chunk → (generator proposes Q/A) → distractors → assemble (P-split) → QC → split.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from .core import (chunk_vault, sample_distractors, assemble_example,
                   citation_ok, split_by_page)
from .generator import Generator


def run_synth(vault_dir: str, out_dir: str, generator: Generator, *,
              k_distractors: int = 4, p_golden: float = 0.8,
              eval_frac: float = 0.1, seed: int = 42) -> dict:
    rng = random.Random(seed)
    chunks = chunk_vault(vault_dir)
    if not chunks:
        raise SystemExit(f"No chunks found in {vault_dir} — is the vault empty?")

    examples, dropped = [], 0
    for ch in chunks:
        for qa in generator.generate(ch):
            keep = rng.random() < p_golden
            n_dist = k_distractors if keep else k_distractors + 1  # constant total docs
            dists = sample_distractors(ch, chunks, n_dist, rng)
            ex = assemble_example(qa, ch, dists, keep, rng)
            if citation_ok(ex, ch.text):
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
        "dropped_failed_qc": dropped,
        "train": len(train),
        "eval": len(ev),
        "params": {"k_distractors": k_distractors, "p_golden": p_golden,
                   "eval_frac": eval_frac, "seed": seed,
                   "generator": type(generator).__name__},
    }
    (out / "synth_manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("question", "context", "answer")}) + "\n")
