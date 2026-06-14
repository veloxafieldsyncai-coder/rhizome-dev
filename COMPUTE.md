# COMPUTE — the Rhizome dev machine (Strix Halo)

Everything we've learned about the compute on this box, so we don't relearn it. This is the
hardware/runtime reference for training and serving local models. Last verified 2026-06-14.

---

## 1. Hardware

| | |
|---|---|
| **APU** | AMD **RYZEN AI MAX+ 395** (Strix Halo), 32 CPU threads |
| **iGPU** | **Radeon 8060S** (RDNA 3.5), ROCm arch **gfx1151** |
| **RAM** | **91 GB** usable, **unified** (CPU+GPU share it; no VRAM/RAM boundary) |
| **Disk** | NVMe, ~915 GB (~830 GB free) |
| **OS / kernel** | Ubuntu, kernel 6.17, amdgpu driver loaded |

Memory bandwidth ~256 GB/s (vs ~960 on a discrete GPU): training is slower but big models fit
because there's no VRAM ceiling — within the GTT limits below.

## 2. Users & GPU access

Two users: **`fieldsync`** (uid 1000, primary, has sudo; sudo password held by the human) and
**`ryder`** (uid 1002, the agent's user). The GPU nodes (`/dev/kfd`, `/dev/dri/renderD128`) are
group `render`.

- `ryder` was added to `render`+`video` (`usermod -aG render,video ryder`) — **persists across
  reboot**, active on fresh login.
- Immediate (no relogin) grant if ever needed again: `setfacl -m u:ryder:rw /dev/kfd /dev/dri/renderD128`
  (resets on reboot).
- Privileged commands run via `fieldsync`: `su - fieldsync -c "sudo -S ..."`, password fed through a
  pty (never stored on disk or in argv).

## 3. ROCm + PyTorch toolchain (what actually works)

- **No system ROCm install needed** — AMD's gfx1151 **nightly PyTorch wheels bundle the ROCm
  runtime.** Kernel amdgpu driver + `render` group access is enough.
- Install (in a venv):
  ```bash
  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio
  pip install transformers datasets accelerate peft
  pip uninstall -y bitsandbytes torchao torchvision   # these CRASH on gfx1151
  ```
- Verified: `torch 2.12.0a0+rocm7.13`, `hip 7.13`, `transformers 5.12`, `peft 0.19`.
- **bf16 LoRA, NOT QLoRA.** bitsandbytes/torchao crash on import; Unsloth doesn't detect ROCm; Swift
  needs distributed ops that aren't present. The 91 GB unified memory makes 4-bit unnecessary anyway.

### Required env for every GPU run
```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True   # avoids allocator fragmentation
export HF_DEACTIVATE_ASYNC_LOAD=1                         # serial weight load (see §5)
export HF_HOME=/srv/project/cache/hf                      # shared cache (see §4)
```

## 4. Shared model cache (no per-account duplication)

One HuggingFace cache for the whole box: **`/srv/project/cache/hf`** (group `project`, setgid 2775,
`umask 002`). Set globally in `/etc/profile.d/rhizome-shared-cache.sh` (`HF_HOME`, `HF_HUB_CACHE`).
Models download **once**; `ryder`, `fieldsync`, and every project read the same files. **Base models
are treated read-only** — adapters/outputs live in each project's own tree, never in the cache.
(See `docs/` memory on model/project separation.)

## 5. The memory model — GTT, and the big-model load wall

This is the subtle part. The iGPU has only **2 GB dedicated VRAM** (BIOS); everything else is **GTT**
(system RAM the GPU can address).

- amdgpu fixes the **GTT size at boot** from `ttm.pages_limit`. Default = **½ RAM ≈ 46 GB** → that
  was the real GPU ceiling regardless of the 91 GB total.
- **Raising it requires a boot-time change** (runtime writes to `/sys/module/ttm/parameters/pages_limit`
  do NOT resize the live GTT). We persisted **80 GB** via:
  ```
  /etc/modprobe.d/amdgpu-gtt.conf:  options ttm pages_limit=20971520 page_pool_size=20971520
  sudo update-initramfs -u  &&  reboot
  ```
  After reboot: `mem_info_gtt_total` = 80 GB, torch sees ~86 GB. Check with
  `cat /sys/class/drm/card0/device/mem_info_gtt_total`.

### The 30B load wall — diagnosed and SOLVED ✅

**Symptom (now fixed):** loading Qwen3-30B-A3B (~60 GB bf16) directly with `device_map="cuda"` OOMs
at **~57 of 60 GB** against the 80 GB GTT cap. Root cause:

> HF loads weights from **mmap'd safetensors**, and ROCm **pins those host pages into GTT** for the
> H2D copy — source file (~57 GB) and destination tensors both consume GTT at once, blowing the cap.

Things that did **not** fix it: `expandable_segments:True` (fixed fragmentation only),
`HF_DEACTIVATE_ASYNC_LOAD=1` (moved wall 46 → 57 GB), a root page-cache dropper (pinned pages aren't
reclaimable cache).

**The fix (`spikes/amd-lora/streaming_load.py`):** load to **CPU RAM first** (plain anon memory — no
GTT pinning; `from_pretrained` also gets buffers/tying/init right), then **migrate to the GPU one
tensor at a time, freeing each CPU tensor as we go** (`t.data = t.data.to("cuda")`). Peak stays ~1x
model size, so 60 GB fits. During load GTT stays ~0 and CPU RAM fills to ~60 GB, then it shifts to GTT.

Gotchas also fixed:
- **`lora_dropout` MUST be 0** — Qwen3 MoE experts are fused params wrapped by PEFT's `ParamWrapper`,
  which rejects dropout != 0.
- Tokenize via render-to-string (`apply_chat_template(tokenize=False)` then `tok(text,
  add_special_tokens=False)`) — `tokenize=True` returns an `Encoding` object in this stack.
- **Adapter reload:** `PeftModel.from_pretrained` crashes here (peft 0.19 ↔ transformers 5.12 skew:
  `WeightConverter ... unexpected kwarg 'distributed_operation'`). Workaround in `infer_check.py`:
  re-wrap the base with the saved `LoraConfig` (`get_peft_model`), then `load_state_dict` the adapter
  weights directly (insert `.default` before `.weight`), bypassing peft's transformers conversion.

**Result:** Qwen3-30B-A3B bf16 LoRA **trains end-to-end on this box** — 995 M LoRA params (3.16%) over
31.5 B, loss 3.04 → 1.80, adapter saved. ~4 min for 3 epochs on 8 toy examples (MoE fwd/bwd is slow but
correct). Use `train_raft_lora.py` (streaming load + completion-only masking + cosine/warmup).

**Inference also via the streaming loader** (`infer_check.py`); Ollama/llama.cpp serve the 30B via GGUF
mmap (zero-copy) and never hit the wall regardless.

## 6. Loading a big model on this box — the procedure

For models that fit (≤ ~14B dense today; 30B once fix #1 lands):
```bash
source /srv/project/code/rhizome-dev/spikes/amd-lora/.venv/bin/activate
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True HF_DEACTIVATE_ASYNC_LOAD=1
export HF_HOME=/srv/project/cache/hf
umask 002
```
Watch live: `cat /sys/class/drm/card0/device/mem_info_gtt_used` and `free -g`.

## 7. Phase-0a smoke-test status (`spikes/amd-lora/`)

| Rung | Result |
|---|---|
| 1 · torch sees GPU, bf16 matmul | ✅ PASS |
| 2 · Qwen3-0.6B bf16 LoRA trains | ✅ PASS (loss 4.06→2.39, adapter saved) |
| 3 · Qwen3-30B-A3B bf16 LoRA trains | ✅ PASS via streaming loader (loss 3.04→1.80, 995M LoRA params, adapter saved) |
| 4 · 30B base + adapter inference | ✅ PASS — loaded + generated a correct grounded answer |

**Verdict:** local bf16 LoRA training on this AMD box is **proven, including the 30B MoE** — with the
streaming loader (§5) + the MoE gotchas. This is the green light for Phase 1 (the vault).

### The repeatable 30B recipe (`spikes/amd-lora/`)
```
source .venv/bin/activate
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True HF_DEACTIVATE_ASYNC_LOAD=1
export HF_HOME=/srv/project/cache/hf
python train_raft_lora.py --model Qwen/Qwen3-30B-A3B --data <raft>.jsonl --out <adapter>
python infer_check.py  --model Qwen/Qwen3-30B-A3B --adapter <adapter>
```
`streaming_load.py` is the reusable memory-safe loader (training and serving).
