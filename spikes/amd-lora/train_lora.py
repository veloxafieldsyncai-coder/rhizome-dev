"""Phase 0a — RAFT-style bf16 LoRA smoke test for the AMD Strix Halo box.

Goal: prove that Qwen3-30B-A3B (MoE) trains a LoRA adapter end-to-end on gfx1151,
saves it, and the adapter reloads for inference (run infer_check.py after).

NOT production: trains on the full prompt+answer sequence (no completion-only
masking) — that's a Phase 3 TODO. Here we only test the training MECHANICS.

Ladder: run first with --model Qwen/Qwen3-0.6B (cheap, proves PEFT+ROCm works at
all), then with the default 30B MoE (proves MoE + size on AMD — the real risk).
"""
import os, json, argparse
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--data", default="toy_raft_data.jsonl")
    ap.add_argument("--out", default="adapter-out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_jsonl(args.data)

    def to_text(ex):
        user = f"Question: {ex['question']}\n\nContext:\n{ex['context']}"
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": ex["answer"]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    ds = Dataset.from_dict({"text": [to_text(r) for r in rows]})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_seq_len),
                batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.config.use_cache = False

    # Suffix-name matching also catches every MoE expert's gate/up/down_proj but
    # NOT the router (named `gate`) — exactly what we want for MoE LoRA.
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.01, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()  # needed with gradient checkpointing + PEFT
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=6,
        num_train_epochs=args.epochs, learning_rate=1e-4, bf16=True, logging_steps=1,
        save_strategy="no", optim="adamw_torch", gradient_checkpointing=True,
        dataloader_num_workers=0, report_to=[])

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nPASS: adapter saved to {args.out}")


if __name__ == "__main__":
    main()
