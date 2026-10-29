#!/usr/bin/env python
"""P1 —— 后训练数据配比消融（用 0.5B 做 proxy，能省 4 倍卡时）。

工业界真实做法：**小模型上做数据配方搜索，大模型上验证**。
写进简历时要突出这一点 —— 它体现的是成本意识，而不是"我不会用大模型"。

单一变量原则：每组实验的总 token 预算、step 数、seed、超参**完全一致**，只改数据配比。

用法：
    python scripts/05_data_mix_ablation.py --proxy Qwen/Qwen2.5-0.5B-Instruct --total 8000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from posttrain_lab.data_utils import read_jsonl, save_jsonl, mix_by_ratio  # noqa: E402

# 四组对照：不要一次变多个东西
ABLATIONS = {
    "A_math_only": {"math": 1.0, "gsm8k": 0.0},
    "B_gsm8k_only": {"math": 0.0, "gsm8k": 1.0},
    "C_3to1": {"math": 0.75, "gsm8k": 0.25},
    "D_1to3": {"math": 0.25, "gsm8k": 0.75},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-data", default="data/sft/train.jsonl")
    ap.add_argument("--total", type=int, default=8000)
    ap.add_argument("--out", default="outputs/ablation")
    ap.add_argument("--dry-run", action="store_true", help="只切数据不训练")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = read_jsonl(args.train_data)

    # 按 source 分池；若你的数据源命名不同，在这里改映射
    pools: dict = {}
    for r in rows:
        src = r.get("source", "unknown").lower()
        key = "gsm8k" if "gsm8k" in src or "openmathinstruct" in src and "gsm" in src else "math"
        pools.setdefault(key, []).append(r)
    print("[pools]", {k: len(v) for k, v in pools.items()})

    summary = {}
    for name, ratio in ABLATIONS.items():
        subset = mix_by_ratio(pools, ratio, args.total)
        data_path = os.path.join(args.out, f"{name}.jsonl")
        save_jsonl(subset, data_path)
        print(f"[{name}] {len(subset)} 条 -> {data_path}")

        if args.dry_run:
            continue

        out_dir = os.path.join(args.out, name)
        cmd = [
            sys.executable, "scripts/02_train_sft.py",
            "--config", "configs/sft.yaml",
            "--data", data_path,
        ]
        # 用环境变量把模型/输出覆盖掉，避免改主配置
        env = dict(os.environ)
        env["ABLATION_MODEL"] = args.proxy
        print("  $", " ".join(cmd))
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            summary[name] = {"status": "failed"}
            continue

        eval_cmd = [
            sys.executable, "scripts/04_eval.py",
            "--ckpt", os.path.join(out_dir, "final"),
            "--tasks", "gsm8k", "--limit", "300",
            "--out", out_dir,
        ]
        subprocess.run(eval_cmd, env=env)

        rp = os.path.join(out_dir, "eval_results.json")
        if os.path.exists(rp):
            summary[name] = json.load(open(rp, "r", encoding="utf-8"))["results"]["gsm8k"]
        else:
            summary[name] = {"status": "no eval"}

    with open(os.path.join(args.out, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== 数据配比消融结论 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n把这张表最长的那句结论，直接写进简历的 P1 项目里。")


if __name__ == "__main__":
    main()
