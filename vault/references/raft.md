---
title: RAFT — Retrieval-Augmented Fine-Tuning
category: references
layer: research
tags: [raft, fine-tuning, rag, training]
summary: RAFT fine-tunes a model to excel at RAG over a fixed domain by training on questions with golden + distractor docs and cited chain-of-thought answers.
sources: [papers/RAFT.pdf, https://arxiv.org/abs/2403.10131]
lifecycle: active
created: 2026-06-14
updated: 2026-06-14
---

# RAFT — Retrieval-Augmented Fine-Tuning

Zhang et al., UC Berkeley, 2024 (arXiv:2403.10131). The method Rhizome uses to "train" the
local model on the vault. It is the open-book-exam answer to *how do you adapt a model to a
private document collection*: study **with the book open**.

> [!tldr] Fine-tune on `(question, retrieved docs incl. distractors, CoT answer that cites the
> source verbatim)` so the model learns to use the right doc, ignore distractors, and ground +
> cite its answers. A RAFT-tuned 7B beat GPT-3.5+RAG on in-domain QA.

## The recipe (the knobs)

- Each training example = a question + **k documents** = one **golden** (holds the answer) +
  several **distractors** (irrelevant).
- **Answer is chain-of-thought and quotes the source verbatim** (`##begin_quote##…##end_quote##`).
  This was the single biggest gain and prevents overfitting to terse answers.
- **P ≈ 80%** of examples keep the golden doc; the other ~20% see **only distractors** (forces the
  model to internalize, not copy). 100%-golden is worse.
- Train **with distractors present** (paper: 1 golden + 4 distractors) → robustness to imperfect
  retrieval at query time.
- Retriever-independent; at test time it's standard RAG (top-k docs in context).

## Why Rhizome uses it

It unifies "RAG now / fine-tune later" into one mechanism, and its cite-verbatim + distractor
robustness directly counter model collapse / hallucination. See [[rhizome-basis]]. The synth
pipeline that builds RAFT data from the vault is documented in `docs/raft-synth-detail.eraser`.

## Related

- [[rhizome-basis]] — how RAFT fits the whole basis
