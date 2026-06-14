"""Phase 0a — rung 1: does torch see the Radeon 8060S (gfx1151)?

Cheapest possible go/no-go on the whole AMD toolchain. If this fails, stop —
no point downloading a 60GB model. ROCm presents to torch through the CUDA API.
"""
import torch

print("torch:", torch.__version__)
print("hip runtime:", getattr(torch.version, "hip", None))

if not torch.cuda.is_available():
    raise SystemExit("FAIL: GPU not visible to torch — toolchain not ready. Stop here.")

print("device:", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"gpu mem free/total: {free/1e9:.1f} / {total/1e9:.1f} GB")

# prove a real bf16 compute kernel runs on the iGPU, not just device enumeration
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print("bf16 matmul on GPU OK (checksum finite):", y == y and abs(y) < float("inf"))
print("PASS: GPU usable for bf16 compute")
