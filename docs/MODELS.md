# Rhizome — Local Model Roster

The local agent runs on Ollama against open-weight models. Three specialized roles; Ollama swaps
the one you call (you don't need all three resident, though this box can keep two hot).

## Hardware

- **AMD Ryzen AI MAX+ 395** (Strix Halo), 32 cores, Radeon 8060S iGPU.
- **91 GB unified memory** (~96 GB allocatable to the iGPU).
- Best backend for this iGPU: **Vulkan (RADV)** — faster than ROCm/HIP for llama.cpp/Ollama.
- Excels at **Mixture-of-Experts** models (high tok/s for active-param count).

## Roster

| Role | Model | Tag | Why |
| --- | --- | --- | --- |
| **General / thinking** | Qwen3-30B-A3B (MoE) | `qwen3:30b-a3b` | The everyday conversational agent across all projects (incl. non-physics). MoE, ~3B active, ~50 tok/s, thinking mode. ~18GB. |
| **Physics / reasoning** | QwQ-32B | `qwq:32b` | The engine's heavy thinking — governing-equation derivation, symbolic math. Dense 32B reasoner, shows its work. No physics-*specialized* open model exists; this is a top reasoner. ~20GB. |
| **Coding** | Qwen3-Coder 30B | `qwen3-coder:30b` | Building the engine + future code. Apache-2.0 MoE, 256K context, agentic/tool use, ~50 tok/s. ~18GB. |

## Mapping to the engine

- **General (Qwen3)** drives dialogue + the brainstorm substrate — speculation lands safely here.
- **Physics (QwQ)** does the math layer: deriving/checking governing equations, nondimensionalization,
  symbolic reasoning the deterministic engine then verifies.
- **Coding (Qwen3-Coder)** is an optional local builder; Claude Code also builds the engine.

Note the boundary: these models **speculate and assist**; the **matcher / gates / nondimensionalization
stay deterministic Python**, not model calls (per `rhizome-fork-strategy.md`). Models propose,
the engine verifies.

## Choosing / changing models — not limited to the above

The roster is just a **default**. The agent talks to models by *role*, and the role → model map
lives in `config/models.json`. Any Ollama tag works, roles are free-form, and a `provider` field
leaves room for non-Ollama backends later. Manage it with the zero-dependency CLI:

```
python -m rhizome.models show            # current role -> model map (✓ installed, · not pulled)
python -m rhizome.models list            # every installed model + suggestions (hints only)
python -m rhizome.models choose          # interactive picker for all roles
python -m rhizome.models choose physics  # pick for one role — any installed model or any tag
python -m rhizome.models set physics deepseek-r1:32b   # assign directly (creates role if new)
python -m rhizome.models get physics     # print the tag (for the agent/scripts to consume)
python -m rhizome.models pull qwq:32b    # pull via ollama
```

The picker always offers: any installed model, the role's suggestions, or **type any custom tag**
(with an offer to pull it). Different projects can use different models — a physics-heavy project
might set `general` to a reasoner, a writing project might set it to something else entirely.

## Run

```
ollama run qwen3:30b-a3b      # general
ollama run qwq:32b            # physics/reasoning
ollama run qwen3-coder:30b    # coding
```

For best iGPU performance, ensure Ollama uses the Vulkan backend.
