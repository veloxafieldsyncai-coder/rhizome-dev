# Rhizome: Fork Strategy and Build Boundary

A companion to the Rhizome method document. This records the decision to build on an existing open-source base, what we keep, what we replace, and exactly where the line sits between their code and ours. The short version: fork the loading dock, build the engine.

---

## The Decision in One Line

Fork an MIT-licensed implementation of Karpathy's LLM Wiki pattern for the knowledge store and the intake plumbing, and build the maturation-and-matching engine ourselves as deterministic Python writing to the same markdown vault. The vault is the interface between the two halves.

The base under evaluation is Ar9av/obsidian-wiki (MIT, ~1.5k stars, 57% Python). claude-obsidian (MIT) is a fallback with the same licensing posture.

---

## Why This Is Allowed (Licensing)

MIT is one of the most commercial-friendly licenses that exists. It is permissive, not copyleft. The distinction is the only thing that matters here:

- Permissive (MIT, BSD, Apache): you can take the code private, modify it, sell it, and keep your additions closed-source. The only duty is to retain the original copyright notice and license text in whatever you ship.
- Copyleft (GPL and relatives): a commercial derivative must itself be released under the same open terms. This is the viral case we are NOT in.

So a commercial Rhizome built on this base is permitted. The compliance cost is a single line in a NOTICES or THIRD_PARTY_LICENSES file. Not a lawyer, so a venture-stage product gets a real legal review, but MIT is about as unambiguous as licenses get.

### The three diligence items the top-level license does not cover

1. **The dependency tree.** The repo being MIT does not make everything it pulls in MIT. Companion skill packs, the optional semantic-search tool, and any bundled bits each carry their own license. Audit the transitive tree before committing. A copyleft surprise, if one exists, hides here, not in the top repo.
2. **Obsidian itself.** Obsidian the app is proprietary freeware, not open source. The clean escape: we do not need it. The vault is plain markdown and Obsidian is only a viewer. Our product is the engine plus our own UI over the same folder. Dropping Obsidian removes a proprietary dependency and sidesteps the question entirely. Since the engine is the whole value, this is the right architecture regardless.
3. **Trademark.** MIT grants no trademark rights, and "Obsidian" is a trademark. Ship nothing named obsidian-anything and imply no endorsement. We already have the right name.

### The quiet reason most of this is moot

Ideas and patterns are not copyrightable, only specific expression is. Because our engine is separate deterministic Python writing to a plain markdown vault, we may copy very little of their actual code. The more we reimplement the intake our own way rather than lifting their files, the lighter the obligation gets, toward zero. We do not need to lean on that, because MIT compliance is already trivial. The license is a green light.

---

## What We Keep From the Fork (the boring 70 percent)

This is real, working plumbing we do not want to rebuild. Each piece maps to a role in the Rhizome method.

| Kept from fork | What it does | Role in Rhizome |
| --- | --- | --- |
| `_raw/` staging directory | Drop rough notes, next ingest promotes them to pages | The brainstorm-layer intake, already built |
| Ingest / merge / dedup | Distill sources into pages, merge against existing, track sources in frontmatter | Gets knowledge into the vault so the engine has something to mature |
| Provenance tagging | Marks each claim extracted, inferred, or ambiguous; lint flags speculation drift | A primitive grounding signal we upgrade into real epistemic layers |
| Manifest / delta tracking | Only processes what is new or changed since last run | Keeps the vault current without full reprocessing |
| Cross-linker | Scans for unlinked mentions, inserts wikilinks | Demoted to a candidate generator only (see below) |
| Graph export | Vault to JSON, GraphML, Neo4j, HTML | Lets the engine and external tools read the graph |
| Lint / audit | Orphans, broken links, contradictions, missing frontmatter | Vault hygiene the engine can assume |

The value of keeping these is focus. The intake, dedup, linking, export, and provenance plumbing is done and free, so all of our effort goes to the part nobody has built.

---

## What We Build Ourselves (the 30 percent that is the whole point)

Everything the fork does operates at the associative English layer. It links pages by shared mentions and tags. That is the noisy bottom rung the Rhizome thesis exists to suppress. The engine is not there. We build it.

| We build | Replaces / upgrades | Why theirs is not enough |
| --- | --- | --- |
| Epistemic layers with promotion gates | Their flat provenance tag | They track "did the LLM guess this," not "what maturity is this thread at." We add brainstorm to hypothesis to frontier to principle, each gate costing evidence and explicit structure |
| Typed relational encoding | Their prose pages | Alignment needs wiring, not paragraphs. We require mechanism stated as typed relations |
| Deterministic structural matcher | Their cross-linker and surprising-connection scoring | Theirs counts shared nouns. We align relations, weighting causal and governing correspondence and depth over count |
| Nondimensionalization / equation-form matching | Nothing in the fork | The proof-grade top rung. Canonicalize governing equations to their dimensionless skeleton and match form |
| Principle condensation | Their synthesize skill | When two matured threads share a deep core, condense it into a principle-node one rung up, itself matchable |
| Graded-confidence connection | Their binary wikilink | A connection reports the layer it held at: English suggests, structural is strong, mathematical is proof |

### The trap to avoid

The fork's logic is prompt-driven: skills an AI agent reads and executes. That is acceptable for ingest and maintenance, where fuzzy is fine. It is the wrong tool for the hard parts. Structural matching, the gates that check real evidence, and the nondimensionalization that canonicalizes an equation must be deterministic Python, not a skill an LLM reinterprets every run. If the matcher is written as another markdown skill, we have rebuilt the noise machine. Keep the engine deterministic and out of the agent layer.

---

## The Build Boundary (where their code ends and ours begins)

The markdown file, body plus YAML frontmatter, is the contract. Their skills handle getting knowledge into the vault. Our engine owns what happens to it once it is there. Neither side reaches into the other's internals; both sides agree on the frontmatter schema.

```
  SOURCES                                          
    |  (their ingest / merge / provenance skills)  
    v                                              
  VAULT  ── plain markdown + YAML frontmatter ──   the contract
    ^                                              
    |  (our deterministic Python engine)           
    v                                              
  LAYERS · GATES · STRUCTURAL MATCH · PRINCIPLE-NODES
```

Their side runs as agent skills. Our side runs as a Python package that reads the vault, computes, and writes back new frontmatter and new nodes. Because the interface is files, the two can evolve independently and we copy almost none of their code.

### The frontmatter schema our engine reads and writes

A thread node:

```yaml
id: thrm-tpct-001
title: Passive two-phase closed thermosyphon
layer: principle            # brainstorm | hypothesis | frontier | principle
summary: Buoyancy-driven loop moves heat with no pump
provenance: extracted       # kept from fork: extracted | inferred | ambiguous
claim: A temperature difference alone can drive closed-loop heat transport
relations:                  # the wiring; this is what makes a thread matchable
  - {type: causes,     from: temp_difference, to: phase_change}
  - {type: causes,     from: phase_change,    to: density_difference}
  - {type: causes,     from: density_difference, to: buoyant_flow}
  - {type: yields,     from: buoyant_flow,    to: passive_transport}
  - {type: bounded_by, from: passive_transport, to: rayleigh_number}
governing_equations:        # the math layer, where it exists
  dimensionless_groups: [Ra, Nu, Pr]
  canonical_form: "Nu = f(Ra, Pr)"
evidence:                   # the grounding tax for promotion
  - {kind: hand_calc, ref: capstone, agreement: 0.993}
  - {kind: simulation, ref: solidworks_flow}
gates_passed: [falsifiable, experiment_designed, survived_generalized]
```

A principle-node, written by the engine when a match condenses:

```yaml
id: principle-natural-circulation
type: principle
invariant: Density gradient from energy input drives closed-loop transport without external work
instances: [thrm-tpct-001, bwrx-isolation-condenser-002]
match_layer: structural     # english | structural | mathematical
dimensionless_groups: [Ra]
status: grounded            # provisional | grounded
```

A connection edge, written by the matcher:

```yaml
between: [thrm-tpct-001, bwrx-isolation-condenser-002]
match_layer: structural
root_correspondences:       # relation-to-relation, not noun-to-noun
  - {a: "causes(temp_difference, phase_change)", b: "causes(decay_heat, phase_change)"}
  - {a: "causes(density_difference, buoyant_flow)", b: "causes(density_difference, natural_circulation)"}
confidence: 0.81
status: provisional         # stays provisional until grounded externally
```

This schema is the whole detachment. Their skills never need to understand `relations`, `gates_passed`, or `root_correspondences`; they only have to leave those fields alone. Our engine never edits their ingest logic; it only reads the body and writes these fields.

---

## How It Connects to the Whole Idea

The fork decision is not a detour from the Rhizome method, it is the method applied to its own construction. The three pillars line up directly:

- **Layers of ideas.** Their `_raw/` plus provenance is the raw associative bottom. Our epistemic layers and gates are the climb on top. The fork fills the brainstorm layer; we build the ladder.
- **The maturity cycle.** Their ingest gets an idea written down. Our gates make it pay evidence and structure to rise. Promotion stays our code because the gates are the defensible mechanism.
- **Connecting matured ideas across disciplines.** Their cross-linker generates cheap candidates at the English layer. Our structural matcher and nondimensionalization do the expensive job of confirming the few that earn it, and condensing the shared core into a principle-node. They propose, we verify.

And the strategic logic is the same one that drove the Field Sync pivot. There are now several MIT implementations of this pattern plus the original gist, so the substrate is commoditizing fast and is worth nothing to own because anyone can fork it by this afternoon. The entire value sits in the engine none of them have. Forking just stops us paying rent on the part that is already free.
