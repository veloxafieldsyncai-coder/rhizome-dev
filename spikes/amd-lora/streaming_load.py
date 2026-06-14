"""Memory-safe base-model loader for unified-memory AMD (Strix Halo / gfx1151).

Why this exists (see COMPUTE.md §5): loading a big model directly with
`device_map="cuda"` OOMs, because ROCm pins the mmap'd safetensors into GTT *while*
the destination tensors also fill GTT — a ~2x peak that blows the 80GB GTT cap before
a 60GB model finishes loading.

Fix: load to CPU RAM first (plain anonymous memory, no GTT pinning — from_pretrained
also gets buffers / weight-tying / init correct), then migrate to the GPU one tensor at
a time, freeing each CPU tensor as we go. Peak stays ~1x model size, which fits.
"""
from __future__ import annotations
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def _migrate_to_cuda(model: torch.nn.Module) -> None:
    """Move params+buffers to cuda one tensor at a time; free CPU storage as we go.

    `t.data = t.data.to("cuda")` drops the last ref to the CPU storage on reassign, so
    at any instant memory is ~(remaining CPU) + (already-GPU) + one in-flight tensor —
    never a full 2x copy of the model.
    """
    for p in model.parameters(recurse=True):
        if p.device.type != "cuda":
            p.data = p.data.to("cuda", non_blocking=False)
    for b in model.buffers(recurse=True):
        if b.device.type != "cuda":
            b.data = b.data.to("cuda", non_blocking=False)
    gc.collect()
    torch.cuda.synchronize()


def load_base_to_gpu(model_id: str, dtype: torch.dtype = torch.bfloat16):
    """Load a base causal LM fully onto the GPU without the 2x GTT spike.

    Returns (model, tokenizer). Run with HF_DEACTIVATE_ASYNC_LOAD=1 so the CPU-side
    load stays serial and bounded to ~one shard of overhead.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )  # -> CPU RAM
    _migrate_to_cuda(model)  # -> GPU, streamed
    free, total = torch.cuda.mem_get_info()
    print(f"[streaming_load] base on GPU; GPU mem used "
          f"{(total - free) / 1e9:.1f} / {total / 1e9:.1f} GB")
    return model, tok
