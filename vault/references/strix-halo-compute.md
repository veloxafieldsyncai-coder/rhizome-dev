---
title: Strix Halo Compute — local 30B training on AMD
category: references
tags: [hardware, amd, rocm, training, strix-halo]
summary: How to train and serve a 30B model on the AMD Strix Halo box — GPU access, 80GB GTT, the streaming loader that beats the unified-memory load OOM.
sources: [COMPUTE.md]
lifecycle: active
created: 2026-06-14
updated: 2026-06-14
---

# Strix Halo Compute — local 30B training on AMD

The Rhizome dev/serving box is an AMD **RYZEN AI MAX+ 395** (Radeon 8060S, gfx1151), **91 GB
unified memory**. Full reference: `COMPUTE.md` at the repo root. Distilled lessons:

## Toolchain

- gfx1151 **nightly PyTorch** wheels bundle the ROCm runtime (no system ROCm install needed).
- **bf16 LoRA, not QLoRA** — bitsandbytes/torchao/Unsloth crash or don't detect ROCm; the 91 GB
  unified memory makes 4-bit unnecessary anyway.
- Required env: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `HSA_ENABLE_SDMA=0`,
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `HF_DEACTIVATE_ASYNC_LOAD=1`.

## Memory — the GTT and the load wall

- BIOS gives only 2 GB dedicated VRAM; the iGPU uses **GTT** (system RAM). amdgpu fixes GTT size
  at boot from `ttm.pages_limit` — default ½ RAM. Raised to **80 GB** via a modprobe option + reboot.
- **The 30B load OOM:** loading directly to GPU pins the mmap'd safetensors into GTT alongside the
  destination tensors → a 2× spike that blows the cap at ~57/60 GB.
- **The fix (streaming loader):** load to CPU RAM first, then migrate to GPU one tensor at a time,
  freeing each — peak stays ~1× model size. `spikes/amd-lora/streaming_load.py`.

## Proven

Qwen3-30B-A3B bf16 LoRA **trains and serves locally** (loss fell, adapter saved, generated a
grounded answer). This is the compute foundation for [[rhizome-basis]] and [[raft]].
