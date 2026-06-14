# Rhizome Vault — the knowledge corpus (basis schema)

This is the **vault**: the markdown knowledge base that is the contract of the Rhizome basis.
Knowledge is compiled into pages here; the RAFT synthesizer chunks these pages to generate
training data, and the local model is trained + served against them. Every change is a git
commit — that is the "GitHub-like, see what teammates added" model.

## Layout

```
vault/
├── index.md            # catalog of every page (one line each) — kept current
├── .manifest.json      # tracks ingested sources + delta (what changed since last train)
├── concepts/           # ideas, mental models, architectures
├── references/         # factual sources — papers, specs, distilled docs
├── skills/             # how-to knowledge, procedures
└── projects/           # per-project knowledge
```

## Page frontmatter (basis schema — trimmed)

```yaml
---
title: Page Title
category: concepts            # concepts | references | skills | projects
tags: [a, b]
summary: One sentence (≤200 chars) so a reader/skill previews without opening.
sources: [path-or-url]
lifecycle: active             # PRIMITIVE for now: draft | active. (Full state machine = later.)
created: 2026-06-14
updated: 2026-06-14
---
```

Body is plain markdown with `[[wikilinks]]` to related pages.

## Deferred (NOT in the basis)

Epistemic layers (brainstorm→hypothesis→frontier→principle), typed `relations:`, structural
matching, confidence formulas, tiering, visibility tags — these belong to **Rhizome Innovation
(V2)**, not the basis. The basis vault is a clean knowledge corpus for RAG + RAFT.
