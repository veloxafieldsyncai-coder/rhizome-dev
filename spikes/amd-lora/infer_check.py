"""Phase 0a final rung: load base (streaming) + trained adapter and generate.

Loads the LoRA adapter weights DIRECTLY (re-wrap with the saved LoraConfig, then
load_state_dict) to bypass a peft<->transformers version-skew bug in
PeftModel.from_pretrained (WeightConverter 'distributed_operation' kwarg). See COMPUTE.md.
"""
import argparse, re, torch
import safetensors.torch as st
from peft import get_peft_model, LoraConfig
from streaming_load import load_base_to_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
ap.add_argument("--adapter", default="adapter-30b")
args = ap.parse_args()

base, tok = load_base_to_gpu(args.model)
cfg = LoraConfig.from_pretrained(args.adapter)
model = get_peft_model(base, cfg)

# saved keys are '...lora_A.weight'; live module params are '...lora_A.default.weight'
sd = st.load_file(f"{args.adapter}/adapter_model.safetensors")
remap = {re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k): v for k, v in sd.items()}
res = model.load_state_dict(remap, strict=False)
unexpected_lora = [k for k in res.unexpected_keys if "lora_" in k]
print(f"[adapter] loaded {len(sd)} tensors | unexpected lora keys: {len(unexpected_lora)}")
assert not unexpected_lora, f"adapter keys didn't match: {unexpected_lora[:3]}"
model.eval()

q = "What drives flow in a passive two-phase closed thermosyphon with no pump?"
ctx = ("[DOC1] Redis caches sessions. [DOC2] A temperature difference drives a phase "
       "change, which drives a density difference, which drives buoyant flow with no "
       "moving parts. [DOC3] YAML frontmatter uses triple dashes.")
msgs = [{"role": "user", "content": f"Question: {q}\n\nContext:\n{ctx}"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
inp = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=200, do_sample=False)
print("\n--- generation ---")
print(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
print("\nPASS: base + adapter loaded and generated")
