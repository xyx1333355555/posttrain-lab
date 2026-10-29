#!/usr/bin/env python
"""Step 0 —— 最小闭环 smoke test。

目的：在花任何真金白银之前，先用极小的资源确认以下五件事全部 OK：
  1. GPU / CUDA / torch 可用
  2. 模型和 tokenizer 能加载
  3. 数据 pipeline 能产出正确格式
  4. 奖励函数能返回 float 且维度正确
  5. 一次 forward+backward 能跑通，显存占用于预期范围

用法：
    python scripts/00_smoke_test.py --steps 20 --data-limit 256
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def check_offline() -> bool:
    """第 0 关：不需要 GPU 的逻辑自检。

    这一步拦截的是最贵的那类错误——等到真跑起来才发现，几小时配额就没了。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tp = os.path.join(here, "..", "tests", "test_offline.py")
    print("=" * 60)
    print("[0/5] 离线逻辑自检（无 GPU 消耗）")
    proc = subprocess.run([sys.executable, tp], check=False)
    if proc.returncode != 0:
        print("\n[fatal] 离线自检未通过。**先修上面的错，别急着烧 GPU。**")
        return False
    return True


def check_env() -> None:
    import torch

    print("=" * 60)
    print("[1/5] 环境检查")
    print(f"  torch      : {torch.__version__}")
    print(f"  cuda avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device     : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM       : {total:.1f} GB")
        if total < 20:
            print("  [warn] 显存 < 20GB，请确认使用 1.5B 模型 + gradient checkpointing")
    else:
        print("  [warn] 无 GPU，只能验证数据侧逻辑")
    print("  HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", "(默认，建议设为 https://hf-mirror.com)"))


def check_tokenizer_model(path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("=" * 60)
    print("[2/5] 模型 / tokenizer 加载")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  vocab size : {len(tok)}  eos={tok.eos_token}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    attn = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation=attn, trust_remote_code=True
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] flash_attention_2 不可用({e})，回退 eager")
        attn = "eager"
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation=attn)
    n = sum(p.numel() for p in model.parameters())
    print(f"  params     : {n/1e9:.3f} B   attn={attn}")
    return tok, model


def check_pipeline(tok) -> None:
    from posttrain_lab.prompts import build_prompt, build_target

    print("=" * 60)
    print("[3/5] 数据 pipeline")
    p = build_prompt("What is 2+3?", tok)
    t = build_target("Add 2 and 3.", "5")
    print("  --- prompt ---")
    print("  " + p.replace("\n", "\n  ")[:300])
    print("  --- target ---")
    print("  " + t.replace("\n", "\n  ")[:300])


def check_rewards() -> None:
    from posttrain_lab.rewards import (
        make_correctness_reward,
        make_format_reward,
        make_length_reward,
    )
    from posttrain_lab.eval_utils import extract_boxed_answer, is_math_equivalent

    print("=" * 60)
    print("[4/5] 奖励函数")
    good = "<reasoning>x</reasoning> The final answer is \\boxed{5}."
    bad = "I think maybe the answer is 5?"
    long_ok = "<reasoning>" + "y" * 3000 + "</reasoning> The final answer is \\boxed{5}."

    rc = make_correctness_reward()
    rf = make_format_reward()
    rl = make_length_reward(ref_tokens=400, mode="soft")

    print("  correctness(good/bad) :", rc([good, bad], answer=["5", "5"]))
    print("  format(good/bad)      :", rf([good, bad], answer=["5", "5"]))
    print("  length(good/long)     :", rl([good, long_ok], answer=["5", "5"]))

    ok = is_math_equivalent("0.5", "1/2") and is_math_equivalent("\\frac{1}{2}", "0.5")
    print("  math equiv 0.5==1/2   :", ok)
    print("  extract boxed         :", extract_boxed_answer(good))
    assert extract_boxed_answer(good) == "5", "答案抽取失败！"


def check_step(model, tok, steps: int) -> None:
    import torch

    if not torch.cuda.is_available():
        print("  [skip] 无 GPU，跳过训练步检查")
        return
    print("=" * 60)
    print("[5/5] 训练步 + 显存")
    model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

    t0 = time.time()
    for i in range(steps):
        batch = tok(["The capital of France is"] * 4, return_tensors="pt", padding=True).to("cuda")
        labels = batch["input_ids"].clone()
        labels[:, :1] = -100
        out = model(**batch, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i % 5 == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  step {i:>3}  loss={out.loss.item():.4f}  peak={peak:5.2f} GB")
    print(f"  {steps} 步耗时 {time.time()-t0:.1f}s")
    print("  ✔ smoke test 通过，可以开始正式跑 config 里的任务")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--data-limit", type=int, default=256, help="预留参数，便于后续接真实数据")
    ap.add_argument("--skip-offline", action="store_true", help="跳过第 0 关离线自检（不推荐）")
    args = ap.parse_args()

    if not args.skip_offline and not check_offline():
        sys.exit(1)

    check_env()
    tok, model = check_tokenizer_model(args.model)
    check_pipeline(tok)
    check_rewards()
    check_step(model, tok, args.steps)


if __name__ == "__main__":
    main()
