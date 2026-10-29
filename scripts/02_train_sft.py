#!/usr/bin/env python
"""Step 2 —— 冷启动 SFT。

为什么不直接用 LLaMA-Factory：
  自己写 Trainer 才能讲清楚这三个面试高频点：
    - prompt token 是怎么被 mask 成 -100 的（只对 completion 算 loss）
    - 为什么这里不用 packing（要保持样本边界清晰，便于定位脏数据）
    - max_grad_norm / warmup / cosine 各自解决什么问题

用法：
    python scripts/02_train_sft.py --config configs/sft.yaml
    python scripts/02_train_sft.py --config configs/sft.yaml --smoke   # 只训 50 步
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from posttrain_lab.data_utils import read_jsonl  # noqa: E402


class SFTDataset(Dataset):
    """把 prompt+target 拼成一个序列，只对 target 部分计算 loss。"""

    def __init__(self, path: str, tok, max_length: int = 1024):
        self.rows = read_jsonl(path)
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"]
        t_ids = t_ids + [self.tok.eos_token_id]

        # 超长截断：优先保住 completion
        budget = self.max_length - len(t_ids)
        if budget < 0:
            t_ids = t_ids[: self.max_length]
            p_ids = []
        elif len(p_ids) > budget:
            p_ids = p_ids[-budget:]

        input_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + list(t_ids)
        return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--data", default="data/sft/train.jsonl")
    ap.add_argument("--smoke", action="store_true", help="只训 50 步 + 512 条样本")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    tcfg = cfg["training"]

    torch.manual_seed(cfg.get("seed", 42))

    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn = os.environ.get("PTL_ATTN") or cfg.get("attn_implementation", "flash_attention_2")
    dtype_name = os.environ.get("PTL_DTYPE") or cfg.get("torch_dtype", "bfloat16")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"],
            torch_dtype=getattr(torch, dtype_name),
            attn_implementation=attn,
            trust_remote_code=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {attn} 不可用: {e} -> eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"],
            torch_dtype=getattr(torch, dtype_name),
            attn_implementation="eager",
        )
    model.config.use_cache = False

    ds = SFTDataset(args.data, tok, cfg["data"].get("max_length", 1024))
    if args.smoke:
        ds.rows = ds.rows[:512]
        print(f"[smoke] 只用 {len(ds.rows)} 条样本")
    print(f"[data] {len(ds)} 条")

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        seed=cfg.get("seed", 42),
        num_train_epochs=1 if args.smoke else tcfg["num_train_epochs"],
        max_steps=50 if args.smoke else -1,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        save_steps=10 if args.smoke else tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        bf16=(dtype_name == "bfloat16") and tcfg.get("bf16", True),
        fp16=(dtype_name == "float16"),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        optim=tcfg.get("optim", "adamw_torch"),
        report_to=[],
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id))

    # ---- 免费平台：时间预算 + 断点续跑 ----
    max_min = float(os.environ.get("PTL_MAX_MINUTES", "0"))
    if max_min > 0:
        from posttrain_lab.callbacks import CheckpointSync, TimeBudgetCallback

        interrupt_dir = os.environ.get("PTL_INTERRUPT_DIR", os.path.join(cfg["output_dir"], "interrupt"))
        trainer.add_callback(TimeBudgetCallback(max_min, save_dir=interrupt_dir))
        print(f"[free-tier] 时间预算 {max_min/60:.2f} h -> {interrupt_dir}")
        if os.environ.get("PTL_SYNC_REPO"):
            trainer.add_callback(CheckpointSync(every_steps=int(tcfg["save_steps"]) * 2,
                                                local_dir=cfg["output_dir"],
                                                repo_id=os.environ["PTL_SYNC_REPO"]))

    resume_from = os.environ.get("PTL_RESUME", "") or None
    res = trainer.train(resume_from_checkpoint=resume_from)

    out = {}
    for k, v in res.metrics.items():
        out[k] = float(v) if isinstance(v, (int, float)) else str(v)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    trainer.save_model(os.path.join(cfg["output_dir"], "final"))
    tok.save_pretrained(os.path.join(cfg["output_dir"], "final"))
    print("[done]", cfg["output_dir"] + "/final")


if __name__ == "__main__":
    main()
