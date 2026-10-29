"""E2 · CPT 之后的 SFT 恢复实验（格式率与任务能力回收）

实验问题
--------
E1 观察到：领域 CPT 之后，模型在 GSM8K 上的「格式率」从 56% 掉到 46%
（不再稳定输出 <reasoning> 段与 \\boxed{}），这就是垂直领域 CPT 的典型副作用：
分布被拉向领域语料，指令跟随的"外壳"被冲掉了。

E2 要回答的是：这部分损失能不能用一小步 SFT 收回来？收回来要付多少代价？

单变量对照
----------
  arm = base ：原始 Qwen2.5-0.5B-Instruct  -> GSM8K SFT   （参考组）
  arm = cpt  ：E1 的 CPT 模型             -> GSM8K SFT   （实验组）

两个 arm 的数据、样本顺序、超参、步数完全相同，唯一差异是初始化权重。
因此 SFT 后（cpt - base）的差值，就是「先做 CPT」这件事对下游对齐的净影响。

用法
----
  python experiments/cpt_domain/train_sft_recover.py \
      --arm base --init D:/models/Qwen2.5-0.5B-Instruct --max-steps 200
  python experiments/cpt_domain/train_sft_recover.py \
      --arm cpt  --init D:/data/cpt_runs/replay10_strong/final --max-steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from posttrain_lab.eval_utils import parse_gsm8k_answer  # noqa: E402
from posttrain_lab.prompts import build_prompt, build_target  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class SFTPairDataset(Dataset):
    """prompt / target 拼接，只对 target 算 loss（prompt 部分 mask 成 -100）。

    与 scripts/02_train_sft.py 保持同一套构造逻辑，保证两边结论可比。
    """

    def __init__(self, rows: list[dict], tok, max_length: int = 768):
        self.rows = rows
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"]
        t_ids = t_ids + [self.tok.eos_token_id]

        budget = self.max_length - len(t_ids)
        if budget < 0:                       # completion 自己就超长：砍 completion
            t_ids = t_ids[: self.max_length]
            p_ids = []
        elif len(p_ids) > budget:            # 优先保住 completion，从左侧截 prompt
            p_ids = p_ids[-budget:]

        input_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + list(t_ids)
        return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class JsonlLogger(TrainerCallback):
    """逐步写 metrics.jsonl，长任务中断后也能看到损失曲线。"""

    def __init__(self, path: str):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_rows(parquet: str, tok, limit: int | None = None) -> list[dict]:
    df = pd.read_parquet(parquet)
    if limit:
        df = df.head(limit)
    rows = []
    for q, a_raw in zip(df["question"].tolist(), df["answer"].tolist()):
        sol = a_raw.split("####")[0].strip()
        ans = parse_gsm8k_answer(a_raw)
        rows.append({
            "prompt": build_prompt(q, tok),
            "target": build_target(sol, ans),
            "answer": ans,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "cpt"])
    ap.add_argument("--init", required=True, help="初始化权重目录")
    ap.add_argument("--data", default=os.path.join(
        ROOT, "experiments", "data_ablation", "mixes", "full", "train.parquet"))
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default="D:/data/sft_runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, f"e2_{args.arm}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== E2 arm={args.arm} init={args.init} steps={args.max_steps} lr={args.lr} -> {out_dir}", flush=True)

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.init, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.init, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True)
    model.config.use_cache = False

    rows = build_rows(args.data, tok, args.limit)
    print(f"[data] {len(rows)} 条 | 样例 target: {rows[0]['target'][:120]!r}", flush=True)

    targs = TrainingArguments(
        output_dir=out_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=args.max_steps,      # 只在最后存一次，省磁盘
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
    )

    log_path = os.path.join(out_dir, "metrics.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    trainer = Trainer(model=model, args=targs,
                      train_dataset=SFTPairDataset(rows, tok, args.max_length),
                      data_collator=Collator(tok.pad_token_id),
                      callbacks=[JsonlLogger(log_path)])

    torch.cuda.reset_peak_memory_stats()
    res = trainer.train()

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    summary = {
        "arm": args.arm,
        "init": args.init,
        "data": args.data,
        "n_train": len(rows),
        "max_steps": args.max_steps,
        "lr": args.lr,
        "bs": args.bs,
        "grad_accum": args.grad_accum,
        "effective_batch": args.bs * args.grad_accum,
        "train_loss": round(float(res.metrics.get("train_loss", float("nan"))), 4),
        "train_runtime_s": round(float(res.metrics.get("train_runtime", 0.0)), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[done]", final_dir, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
