---
title: Rhizome Basis Architecture
category: concepts
layer: research
tags: [rhizome, architecture, raft, knowledge-base]
summary: The Rhizome basis is a markdown vault plus a local model RAFT-tuned on it; RAG-grounded, trained as knowledge grows.
sources: [docs/basis-architecture.eraser, docs/rhizome-fork-strategy.md]
lifecycle: active
created: 2026-06-14
updated: 2026-06-14
---

# Rhizome Basis Architecture

The Rhizome basis (V1) is two things bolted together: a **markdown vault that compiles
knowledge**, and a **local model that learns from it**. It runs entirely on one machine.

## The pipeline

```
VAULT  →  RAFT synth (Q + golden + distractors + cited answer)  →  bf16 LoRA fine-tune
       →  serve with RAG retrieval over the vault
```

- **Vault** — the corpus. Knowledge is compiled once into interconnected pages, not re-derived
  per query ("compile, don't retrieve"). Forked from the Karpathy LLM-Wiki pattern.
- **RAFT** — Retrieval-Augmented Fine-Tuning. Training data is `(question, retrieved docs incl.
  distractors, chain-of-thought answer citing the source verbatim)`. The model learns to use
  retrieved context, ignore distractors, and ground every answer. See [[raft]].
- **Local model** — Qwen3-30B-A3B (MoE), fine-tuned and served on the AMD Strix Halo box. See
  [[strix-halo-compute]].

## Key properties

- **Trained as knowledge grows.** Adding pages to the vault produces new RAFT training data; the
  model retrains (idle-triggered) with a keep-or-revert eval gate borrowed from Karpathy's
  autoresearch — a worse model is automatically discarded.
- **Grounded, not memorized.** RAFT's cite-verbatim + distractor-robustness is the antidote to
  hallucination — load-bearing for safety-critical reuse.
- **One base, per-domain adapters.** The pristine base model is shared read-only; each domain
  (company KB, mining-safety AI, …) is its own vault + its own LoRA adapter.

## What's deferred

Team collaboration UI, the full lifecycle state machine, and the cross-disciplinary
principle-matching engine (Rhizome Innovation, V2) are out of the basis.
