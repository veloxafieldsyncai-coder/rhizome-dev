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
layer: research              # research | brainstorm  (see Two substrates below)
tags: [a, b]
summary: One sentence (<=200 chars) so a reader or skill previews without opening.
sources: [path-or-url]
lifecycle: active             # PRIMITIVE for now: draft | active. Full state machine is later.
created: 2026-06-14
updated: 2026-06-14
---
```

Body is plain markdown with `[[wikilinks]]` to related pages.

## Two substrates (layers)

Every page belongs to one of two substrates, set by the `layer:` field. Default when
absent: pages under `notes/` are `brainstorm`, everything else is `research`.

- **research** is vetted, external, authoritative knowledge: papers, specs, regulations,
  textbooks, distilled references. It is treated as ground truth. The model is trained to
  quote it verbatim and assert it as fact.
- **brainstorm** is the user's own provisional thinking: hypotheses, design notes, open
  questions, dead ends. It is never asserted as fact. The model is trained to attribute it
  ("your notes propose ...") and to reason over it, but only grounds factual claims in the
  research layer.

The two substrates are how the basis keeps the user free to speculate without polluting the
model's factual grounding. The training mechanism that produces this differential behavior
lives in `rhizome/synth` (see the generator example kinds and the asymmetric P split). This is
the primitive, two rung form of the epistemic ladder in `docs/rhizome-method.md`; the full
ladder belongs to Rhizome Innovation (V2).

## Deferred (NOT in the basis)

Epistemic layers (brainstorm→hypothesis→frontier→principle), typed `relations:`, structural
matching, confidence formulas, tiering, visibility tags — these belong to **Rhizome Innovation
(V2)**, not the basis. The basis vault is a clean knowledge corpus for RAG + RAFT.
