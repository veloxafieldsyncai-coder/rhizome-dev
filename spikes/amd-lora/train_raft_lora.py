"""RAFT bf16 LoRA trainer for the local 30B (Strix Halo / gfx1151).

Best-practice training, done properly:
  - memory-safe streaming load of the base (streaming_load.py) — fits the 30B on this box
  - COMPLETION-ONLY loss masking (train on the answer tokens, not the prompt/context)
  - cosine LR schedule with warmup, weight-decay-free LoRA, grad clipping
  - gradient checkpointing + bf16, fixed seed for reproducibility
  - LoRA on attention + MLP/expert projections; base weights frozen (adapter only)

Run:
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HSA_ENABLE_SDMA=0
  export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True HF_DEACTIVATE_ASYNC_LOAD=1
  export HF_HOME=/srv/project/cache/hf
  python train_raft_lora.py --model Qwen/Qwen3-30B-A3B --data toy_raft_data.jsonl --out adapter-30b
"""
from __future__ import annotations
import argparse, json
import torch
from datasets import Dataset
from transformers import (TrainingArguments, Trainer, DataCollatorForSeq2Seq,
                          set_seed)
from peft import LoraConfig, get_peft_model

from streaming_load import load_base_to_gpu

SEED = 42


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_examples(rows, tok, max_len):
    """Format each RAFT row and mask everything but the answer (completion-only)."""
    def encode(ex):
        user = f"Question: {ex['question']}\n\nContext:\n{ex['context']}"
        kw = {}
        try:  # Qwen3: don't inject <think> scaffolding into training targets
            tok.apply_chat_template([{"role": "user", "content": "x"}],
                                    add_generation_prompt=True, enable_thinking=False)
            kw = {"enable_thinking": False}
        except TypeError:
            pass
        # Render to text then tokenize (robust: tokenize=True can return an
        # Encoding object in this stack). add_special_tokens=False — the template
        # already emits the special tokens as text.
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False, **kw)
        full_text = tok.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": ex["answer"]}],
            add_generation_prompt=False, tokenize=False, **kw)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"][:max_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(full_ids))):  # mask prompt+context
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}
    return Dataset.from_list([encode(r) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--data", default="toy_raft_data.jsonl")
    ap.add_argument("--out", default="adapter-30b")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    set_seed(SEED)

    model, tok = load_base_to_gpu(args.model, dtype=torch.bfloat16)
    model.config.use_cache = False  # required with gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA on attention + MLP/expert projections (suffix match also catches MoE
    # experts' gate/up/down_proj, but NOT the router `gate`). Base stays frozen.
    lora = LoraConfig(
        # dropout MUST be 0: Qwen3 MoE experts are fused params wrapped by PEFT's
        # ParamWrapper, which doesn't support lora_dropout != 0.
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_examples(load_jsonl(args.data), tok, args.max_seq_len)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="no",
        dataloader_num_workers=0,
        seed=SEED,
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                      return_tensors="pt")
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nPASS: RAFT LoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
