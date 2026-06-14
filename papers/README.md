# Engine Papers — Read & Extracted

The source material for Rhizome's **engine**: the deterministic core that matures threads and
matches them across disciplines. We read these so the engine architecture rests on prior formal
work rather than invention. Full text of each is extracted under `text/` (via `pdftotext`).

The method names four pillars (`../rhizome-method.md`, "People and Systems That Think This Way").
These are those pillars, read and distilled. Below: a role map, a per-paper distillation, then a
synthesis into Rhizome's own governing formula.

---

## Role map

| File | Authors / year | Rung of the engine |
| --- | --- | --- |
| `gentner-structure-mapping.pdf` | Gentner, 1983 | **The matcher's formal backing** — analogy = shared *relations*, not attributes. |
| `sindy-brunton-2016.pdf` | Brunton, Proctor, Kutz, 2016 | **Governing-equation discovery** — mints a thread's math block from data. |
| `ai-feynman-udrescu-tegmark-2019.pdf` | Udrescu & Tegmark, 2019 | **Nondimensionalization** — the canonical, comparable form of an equation. |
| `ai-feynman-2.0-udrescu-2020.pdf` | Udrescu et al., 2020 | **Pareto strength + modular sub-structure** — principle strength in bits; matchable modules. |
| `lie-symmetry-analysis-kunzinger.pdf` | Kunzinger (after Olver), 2015 | **Proof-grade "same equation"** — two systems match when they share a symmetry group. |

Each paper maps onto one of Rhizome's three matching layers — **english (suggests) · structural
(strong) · mathematical (proof)** — plus the maturation gate that costs evidence + structure.

---

## 1. Gentner — Structure-Mapping  *(the matcher)*

**Core.** Analogy and similarity are one process: alignment of *relational structure* between a
base and a target representation. Representations decompose into **objects, attributes** (one-place
predicates, e.g. `RED(ball)`), **functions** (dimensional info), and **relations** (multi-place
predicates). **Higher-order relations** (paradigmatically `CAUSE`) take lower-order relations as
arguments. *"Common relations are essential to analogy; common objects are not."* This is the
formal warrant for Rhizome's rule **a root is a relation, not a noun**.

**The systematicity principle (the heart).** *"A matching set of relations interconnected by
higher-order constraining relations makes a better analogical match than an equal number of
matching relations that are unconnected to each other."* Match quality is **not a count** — it is
the size of the largest *connected* system bound under a shared causal/governing relation. *"We are
not much interested in analogies that capture a series of coincidences, even if there are a great
many of them."*

**The hard constraints (what makes the matcher deterministic).**
- **One-to-one correspondence** — each element maps to at most one element in the other side.
- **Parallel connectivity** — if two relations correspond, their arguments must correspond too.
- **Relational focus** — score relations over attributes; ignore noun/surface overlap.
- **SME cascade evaluation** — pass evidence from higher-order relations down to the relations and
  arguments they bind, so the score *"favors deep systems over shallow systems, even if they have
  equal numbers of matches."*

**→ Rhizome's matcher must:** (a) represent each thread as typed predicates separating objects /
attributes / relations / higher-order relations; (b) seed correspondences by **concept-level
partial identity** of relations (not surface words — this is exactly why cross-domain matches
survive); (c) enforce one-to-one + parallel-connectivity as **hard filters**; (d) operationalize
"enough roots correspond" as the **largest connected, causally-rooted aligned subgraph**, rewarding
depth over count; (e) use cheap retrieval first, expensive alignment only on candidates (MAC/FAC);
(f) emit the base's unmatched-but-connected relations as **candidate inferences** — Rhizome's
maturation output, flagged provisional until grounded.

This is the **structural** matching layer.

---

## 2. SINDy — Sparse Identification of Nonlinear Dynamics  *(minting the math block)*

**Core.** Recovers governing ODEs directly from time-series data. From state samples and their
derivatives, build a **library** of candidate functions and ask which few actually drive the
dynamics via **sparse regression**. The load-bearing prior is **parsimony**: *"the only assumption
about the structure of the model is that there are only a few important terms that govern the
dynamics … this assumption holds for many physical systems."*

**The algorithm.** State matrix **X**, derivative matrix **Ẋ**, candidate library **Θ(X)**
(constants, polynomials, trig…); solve for sparse coefficients **Ξ** in

> **Ẋ = Θ(X) Ξ**

via sequential thresholded least-squares (STLSQ) or LASSO. One knob λ controls sparsity. Plotting
accuracy vs. number of nonzero terms traces a **Pareto front**; its *"elbow"* selects the
parsimonious law. *"The resulting models are parsimonious, balancing model complexity with
descriptive ability while avoiding overfitting."*

**Limits.** The library must contain the true terms; the right coordinates matter; derivatives must
be clean; dimension blows up combinatorially; demand **off-attractor / out-of-distribution data**
before trusting the law (the cylinder-wake lesson). Usefully, *"structure identification fails
before the coefficients become too inaccurate"* — the **form** can survive when constants are
uncertain.

**→ Rhizome:** SINDy is literally *"mint a `governing_equations` block."* A frontier thread with
data earns its math layer only if a **sparse** law reproduces it. **Parsimony becomes an epistemic
gate**: a 3-term law is a stronger principle than a 30-term fit. Cross-validation = the principle
must hold beyond the evidence that birthed it (Kepler's fit vs. Newton's generalizing law — the
exact layer metaphor). Where data is noisy, promote the **structural law** with constants flagged
uncertain.

---

## 3. AI Feynman — Nondimensionalization  *(the canonical comparable form)*

**Core.** Recovers a symbolic formula via a recursive cascade: **dimensional analysis →
polynomial fit → brute force → neural-net-assisted** symmetry/separability tests. Every step that
**removes a variable** is *"virtually guaranteed to be a step in the right direction."*

**Dimensional analysis (the part Rhizome needs).** A Buckingham-Pi reduction as pure linear
algebra: encode each variable's units as an **integer exponent vector** over a fixed fundamental
basis (m, s, kg, K, V). Build matrix **M** (columns = input dimensions), target **b** = output
dimensions; find a particular solution **p** and a null-space basis **U**. New **dimensionless**
variables `x′ᵢ = ∏ⱼ xⱼ^Uᵢⱼ` and `y′ = y / ∏ᵢ xᵢ^pᵢ`. The count of new variables = **dimension of
the null space** = the Buckingham-Pi number of independent dimensionless groups. *(Worked: Newton's
gravitation, 9 raw variables → 6 dimensionless.)*

**Canonicalization (the key trick).** The null-space basis is gauge-free; AI Feynman *"uses this
freedom to set as many elements as possible in p and U equal to zero,"* forcing **integer powers**
and a **canonical representative** — not one of infinitely many equivalent dimensionless forms.
This is exactly the canonical-form-for-comparison step Rhizome's match layer requires. And
equivalence is tested by CAS: a formula is correct iff *"algebraic simplification of f′ − f
produces the symbol 0"* — **match by symbolic equivalence after canonicalization, not by string
identity**.

**→ Rhizome's math-match pipeline:** (1) encode units as exponent vectors; (2) M/b/null-space →
dimensionless skeleton, gauge-fixed to a canonical integer-power form; (3) recursively quotient out
symmetry & separability to the minimal-variable skeleton; (4) normalize away pure numeric
constants; (5) match two equations by CAS-simplifying the difference of their canonical skeletons
to 0. The **dimensionless skeleton, not the dressed equation, is the matchable object.**

This is the **mathematical** matching layer (form level).

---

## 4. AI Feynman 2.0 — Pareto Strength + Modular Sub-structure  *(strength & partial match)*

**Core.** Two upgrades Rhizome needs. (a) Rank candidate laws on a **Pareto frontier of complexity
(bits) vs. inaccuracy (bits)** instead of arbitrary thresholds. (b) Discover **arbitrary modularity
in a formula's computational graph** from the gradients of a neural-net fit, recursively
decomposing an n-variable law into sub-functions.

**Pareto strength (principle strength in bits).** Complexity = **description length `Ld`** — and
*"both input variables and mathematical functions count toward"* it, so exotic operators cost more.
Accuracy = **MEDL**, mean error-description-length in bits, deliberately outlier-robust. Both axes
in bits → one information plane. A law is kept only if **non-dominated** (nothing simpler is as
accurate, nothing more accurate is as simple). *"Minimizing description length provably avoids the
overfitting problem … noise is unlikely to be predictable by a simple formula with small Ld."* The
**convex corners** of the frontier are the genuinely useful laws (Einstein's KE and `mv²/2` both
appear as distinct corners).

**Modular graph (matchable sub-structure).** **Generalized symmetry** `f(x′,x″) = g[h(x′), x″]` is
detected by a *"smoking-gun"*: the normalized gradient w.r.t. `x′` is **independent of `x″`** —
*"we can discover generalized symmetry without knowing h."* So the engine proves a decomposition
**exists** before paying to find it, then recurses. Two laws need not match as wholes — they can
**share a module** (an inverse-square kernel, a Lorentz factor) at any level of the graph.

**→ Rhizome:** (a) Model the **grounding tax / principle strength as bits** on the Pareto plane;
the strongest principles sit at **convex corners** where added complexity stops buying accuracy —
the maturation sweet spots. (b) Decompose each governing equation into its **modular graph** so two
laws can match on a shared **sub-structure** even when the wholes differ — partial structural
analogy, with a hierarchy of nested matchable modules.

This sharpens both the **gate** (strength in bits) and the **mathematical** layer (partial match).

---

## 5. Lie Symmetry Analysis — Proof-Grade "Same Equation"  *(the rigorous ceiling)*

**Core.** A **symmetry group** of a DE is a continuous group of transformations that **maps
solutions to solutions**. Worked infinitesimally: each one-parameter subgroup has a generator
(vector field) `v = Σ ξⁱ∂_{xⁱ} + Σ φ_α∂_{u^α}`; the set of all infinitesimal symmetries is closed
under the Lie bracket and forms the equation's **symmetry (Lie) algebra**. Derivatives are handled
by **prolongation** to jet space.

**The criterion (computable).** For a maximal-rank system `P_ν = 0`, `G` is a symmetry group iff
the prolonged generator annihilates the equation on its solution variety:

> **pr⁽ⁿ⁾v[P_ν] = 0  whenever  P = 0.**

Solving this yields an overdetermined linear system of **determining equations**; their solution
space *is* the symmetry algebra. *(Worked: the 1-D heat equation `u_t = u_xx` has a 6-dimensional
algebra — translations, scaling, dilation, Galilean boost, a conformal generator — plus an
infinite-dimensional ideal from linearity.)*

**"Same equation" (the part Rhizome lives on).** Thm. 3.14.7: an equation **admits `G` as a
symmetry group iff it is equivalent to one written purely in `G`'s differential invariants** — so
*"shares a symmetry group"* and *"is the same equation up to transformation"* are **formally
identical**. The symmetry algebra is **intrinsic** and carried along by any change of variables, so
it is a **transformation-invariant, domain-independent fingerprint**. It *"cannot know"* whether
`u_t = u_xx` came from heat, chemical diffusion, or option pricing — all yield the **same
6-dimensional algebra with the same brackets**.

**→ Rhizome:** compute each governing equation's **symmetry algebra** and compare its invariants —
**dimension, bracket structure (isomorphism class), presence of an infinite-dimensional ideal
(linearity), and the canonical form in invariants.** Isomorphic algebras = the proof-grade match.
**Caveats (must be honored):** determining equations are hard/sometimes non-algorithmic to solve;
cleanest results are for ODEs in one variable; **equal algebras are necessary but not sufficient**
— a clinching match still needs the explicit connecting transformation. So the fingerprint **gates
a heavier verification step; it does not stand alone.**

This is the **proof** ceiling of the mathematical matching layer.

---

## Synthesis — Rhizome's governing formula (first draft)

The five assemble into one engine with **two axes**: a vertical *maturation* axis (the gate) and a
horizontal *matching* axis (connection across threads). Both are deterministic.

### The maturation gate (vertical — how a thread climbs)

A thread's promotion cost is **evidence + structure**, made concrete:

```
brainstorm → hypothesis : a falsifiable claim exists                    (structure: a claim)
hypothesis → frontier   : an experiment/test is designed for the claim  (structure: typed relations)
frontier   → principle  : a sparse, generalizing law survives the test  (evidence: SINDy + Pareto)
```

- **Structure cost** = the thread exposes typed relations (Gentner): objects/attributes/relations
  with higher-order causal roots. No wiring, no climb.
- **Evidence cost** at the top = a governing law that is **sparse** (SINDy) and **Pareto-strong**
  (AI Feynman 2.0: low description length `Ld`, sitting at a convex corner), validated
  out-of-distribution. **Principle strength is measured in bits.**

### The matching axis (horizontal — how two threads connect)

A connection reports the **highest layer at which it holds** (the graded confidence from the method):

| Layer | Test | Source | Strength |
| --- | --- | --- | --- |
| **english** | shared nouns / embeddings | (the fork's cross-linker) | suggests — mostly noise |
| **structural** | aligned relational subgraph under shared causal roots; one-to-one + parallel connectivity; depth-weighted | **Gentner / SME** | strong |
| **mathematical (form)** | canonical **dimensionless skeletons** equal under CAS simplification; or a shared **modular sub-graph** | **AI Feynman 1.0 + 2.0** | near-proof |
| **mathematical (proof)** | isomorphic **symmetry algebras** + connecting transformation | **Lie / Olver** | proof |

**The governing relation of the engine, in one line:**

> *Two threads connect iff their **relational skeletons align** (Gentner), and the connection is
> **proof-grade** iff their **nondimensionalized governing equations** (AI Feynman) share a
> **symmetry group** (Lie). A thread **rises** iff it pays structure (typed relations) and evidence
> (a **sparse, Pareto-optimal** governing law — SINDy / AIF-2.0). The system **grows** when a
> confirmed match condenses its shared invariant into a new principle-node one rung up — itself a
> skeleton, hence matchable in turn.*

### The asymmetry the method insists on (kept)

A match with **structural** agreement but **no** governing equation is **not a failure** — it is the
**highest-value output**: a discovered frontier, a place where a strong principle exists with no math
behind it yet. The matching axis is allowed to top out at *structural* and still point somewhere
worth mining. That blind spot, named, is the compass to unsolved interdisciplinary work.

---

## Provenance

- SINDy, AI Feynman (both): arXiv 1509.03580, 1905.11481, 2006.10782.
- Gentner: MIT CSAIL course mirror (6.803).
- Lie symmetry: arXiv:1506.07131 (Kunzinger, after Olver, *Applications of Lie Groups to
  Differential Equations*).

Read-for-research references; not redistributed as part of any shipped product.
Per-paper full text under `text/`. Per-paper detailed extractions retained in the build log.
