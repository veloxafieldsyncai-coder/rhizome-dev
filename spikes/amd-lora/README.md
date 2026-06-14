# Phase 0a — AMD 30B LoRA smoke test (the basis go/no-go)

**Goal:** prove `Qwen3-30B-A3B` (MoE) trains a **bf16 LoRA** adapter end-to-end on this
Strix Halo box (Ryzen AI MAX+ 395, gfx1151), saves it, and the adapter reloads for
inference. This is throwaway code — its only job is to de-risk the single assumption the
whole basis rests on: **can we fine-tune the 30B locally on AMD?**

Hardware confirmed: RYZEN AI MAX+ 395, 91 GB unified, 830 GB free, amdgpu kernel driver
loaded. Blocker: the user is not yet in `render`/`video` groups, and no ROCm/torch yet.

---

## Step 0 — one-time privileged setup (USER runs; needs sudo)

```bash
sudo usermod -aG render,video ryder
```

Group changes don't apply to running sessions. Either re-login, **or** prefix the GPU
commands below with `sg render -c '...'` to pick up the group without relogging.

## Step 1 — Python env + gfx1151 nightly torch (no sudo)

```bash
cd /srv/project/code/rhizome-dev/spikes/amd-lora
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# AMD's gfx1151-specific PyTorch nightly (ships its own ROCm runtime libs)
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio

pip install transformers datasets accelerate peft "huggingface_hub[cli]"
# These crash on import on gfx1151 — make sure they are NOT present:
pip uninstall -y bitsandbytes torchao torchvision 2>/dev/null || true
```

Required env for every run on this stack:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HSA_ENABLE_SDMA=0
```

> Gotcha: pip will **silently swap torch for a CPU build** if a later package lists it as
> a dep. After installing everything, re-run Step 2 — if `hip runtime` is `None`, reinstall
> the nightly torch last.

## Step 2 — rung 1: does torch see the GPU? (cheapest gate)

```bash
python check_gpu.py        # or: sg render -c '.venv/bin/python check_gpu.py'
```

Must print `PASS: GPU usable for bf16 compute`. **If it fails, STOP** — toolchain/ROCm
runtime isn't ready (next move would be a system ROCm install, not more downloads).

## Step 3 — rung 2: tiny-model LoRA (proves PEFT+ROCm training works at all)

```bash
python train_lora.py --model Qwen/Qwen3-0.6B --out adapter-tiny --epochs 3
```

Cheap (~1.5 GB download). Confirms the training loop runs and saves an adapter before we
pull 60 GB.

## Step 4 — rung 3: the real gate — 30B MoE LoRA

```bash
python train_lora.py --model Qwen/Qwen3-30B-A3B --out adapter-30b --epochs 3
python infer_check.py --model Qwen/Qwen3-30B-A3B --adapter adapter-30b
```

First run downloads the 30B (~60 GB to `~/.cache/huggingface`). Watch GPU memory headroom
during training.

---

## Pass / fail criteria

| Rung | Pass means |
|---|---|
| 1 · `check_gpu` | torch sees the Radeon 8060S; bf16 matmul runs on GPU |
| 2 · tiny LoRA | training loop completes, loss decreases, adapter-tiny saved |
| 3 · 30B LoRA | **`print_trainable_parameters` shows MoE expert layers adapted**, training completes without OOM/kernel error, `adapter-30b` saved |
| final · infer | base + adapter-30b loads, generates a RAFT-style answer |

**GO** = all four pass → local 30B training is real; proceed to Phase 1 (vault).
**NO-GO** = rung 3 OOMs or hits a MoE/ROCm kernel error → fall back per `basis-architecture`
memory: (a) interim dense Qwen3-14B, or (b) lease NVIDIA for the training burst, serve 30B
locally. Record exactly where it broke.

## Knobs (from the working gfx1151 guide)

`r=32, alpha=64, dropout=0.01`, targets = attn + MLP/expert projections, seqlen 1024,
batch 1 + grad-accum 6, lr 1e-4, bf16, gradient checkpointing, `optim=adamw_torch`.
