"""E1 · 垂直领域增量预训练（CPT）

在 Qwen2.5-0.5B 上做全参继续预训练，支持按 token 比例混入通用回放语料，
用于验证「领域适配 vs 灾难性遗忘 vs 回放缓解」三件事。

用法：
  # 纯领域 CPT
  python experiments/cpt_domain/train_cpt.py --tag pure_domain --max-steps 300
  # 领域 + 10% 通用回放
  python experiments/cpt_domain/train_cpt.py --tag replay10 --max-steps 300 --replay-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def token_count(tok, texts: list[str]) -> int:
    return sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)


def build_token_stream(tok, docs: list[str], eos_id: int) -> list[int]:
    """文档间插入 EOS，随后整体打包成定长块。"""
    stream: list[int] = []
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"]
        stream.extend(ids)
        stream.append(eos_id)
    return stream


def pack_blocks(stream: list[int], seq_len: int) -> list[dict]:
    blocks = []
    for i in range(0, len(stream) - seq_len, seq_len):
        chunk = stream[i:i + seq_len]
        blocks.append({"input_ids": chunk, "attention_mask": [1] * seq_len, "labels": list(chunk)})
    return blocks


class LossLogger(TrainerCallback):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.t0 = time.time()
        self.f = open(path, "w", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": state.global_step, "elapsed_s": round(time.time() - self.t0, 1)}
        for k in ("loss", "learning_rate", "grad_norm", "epoch"):
            if k in logs and logs[k] is not None:
                rec[k] = logs[k]
        loss = logs.get("loss")
        if loss is not None:
            rec["ppl_running"] = round(math.exp(min(loss, 20)), 2)
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        if loss is not None:
            print(f"[step {state.global_step}] loss={loss:.4f} "
                  f"ppl={rec.get('ppl_running')} elapsed={rec['elapsed_s']}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data-dir", default="D:/data/cpt_domain")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "outputs", "cpt"))
    ap.add_argument("--tag", default="pure_domain")
    ap.add_argument("--replay-ratio", type=float, default=0.0, help="回放语料 token 占比，如 0.1")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--optim", default="adamw_torch", choices=["adamw_torch", "adafactor"])
    ap.add_argument("--save-steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-domain-docs", type=int, default=0, help="0 = 全量")
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== CPT tag={args.tag} replay_ratio={args.replay_ratio} out={out_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(out_dir)  # 先落 tokenizer，保证任何 checkpoint 都能独立加载
    eos_id = tok.eos_token_id or tok.pad_token_id

    dom = read_lines(os.path.join(args.data_dir, "domain_train.txt"))
    rep = read_lines(os.path.join(args.data_dir, "replay_train.txt"))
    if args.max_domain_docs:
        dom = dom[: args.max_domain_docs]

    # ---- 按 token 比例混料 ----
    dom_tokens = token_count(tok, dom)
    target_replay = int(dom_tokens * args.replay_ratio / max(1e-9, (1 - args.replay_ratio))) if args.replay_ratio > 0 else 0
    if target_replay > 0:
        random.shuffle(rep)
        picked, acc = [], 0
        for d in rep:
            if acc >= target_replay:
                break
            picked.append(d)
            acc += len(tok(d, add_special_tokens=False)["input_ids"])
        rep = picked
        print(f"混料: 领域 {dom_tokens:,} tok + 回放 {acc:,} tok "
              f"(实际回放占比 {acc / (dom_tokens + acc):.1%}, {len(rep)} 篇)", flush=True)
    else:
        rep = []
        print(f"混料: 纯领域 {dom_tokens:,} tok（{len(dom)} 篇）", flush=True)

    docs = dom + rep
    random.shuffle(docs)
    stream = build_token_stream(tok, docs, eos_id)
    blocks = pack_blocks(stream, args.seq_len)
    print(f"打包: {len(stream):,} tokens → {len(blocks):,} blocks × {args.seq_len}", flush=True)

    # ---- 模型 ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数(全参): {n_train/1e6:.1f} M", flush=True)

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=blocks,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        callbacks=[LossLogger(os.path.join(out_dir, "metrics.jsonl"))],
    )

    t0 = time.time()
    trainer.train()
    dur = time.time() - t0

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    summary = {
        "tag": args.tag, "replay_ratio": args.replay_ratio,
        "train_tokens": len(stream), "blocks": len(blocks),
        "domain_docs": len(dom), "replay_docs": len(rep),
        "max_steps": args.max_steps, "seq_len": args.seq_len,
        "per_device_batch": args.per_device_batch, "grad_accum": args.grad_accum,
        "effective_batch_tokens": args.per_device_batch * args.grad_accum * args.seq_len,
        "trainable_params_m": round(n_train / 1e6, 1),
        "wall_clock_s": round(dur, 1),
        "s_per_step": round(dur / max(1, args.max_steps), 2),
        "tokens_per_s": round(len(stream) * min(1.0, args.max_steps * args.per_device_batch * args.grad_accum * args.seq_len / max(1, len(stream))) / max(dur, 1e-9), 1),
        "final_dir": final_dir,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== train_summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
