#!/usr/bin/env python
"""Step 1 —— 构建冷启动 SFT 数据。

产出：data/sft/train.jsonl，每行 {prompt, target, source, bucket}

工程纪律：
  - 这一步**不要开 GPU**，在租卡平台用无卡模式跑（省钱，这一步能占掉半天有效卡时）
  - 先 --limit 小样本看看数据长什么样，再全量
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from posttrain_lab.data_utils import (  # noqa: E402
    apply_filters,
    dedup,
    load_source,
    save_jsonl,
)
from posttrain_lab.prompts import build_prompt, build_target  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--limit", type=int, default=None, help="调试用：覆盖配置里的 limit")
    ap.add_argument("--out", default="data/sft/train.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    dcfg = cfg["data"]

    rows = []
    for src in dcfg["sources"]:
        if args.limit:
            src = dict(src)
            src["limit"] = min(args.limit, src.get("limit", args.limit))
        rows.extend(load_source(src))

    if not rows:
        raise SystemExit("没有任何数据源加载成功，请检查 logs/skip.log 与网络/HF 镜像")

    rows = apply_filters(rows, dcfg.get("filter", {}))

    if dcfg.get("dedup", {}).get("enable", False):
        rows = dedup(rows, key="problem", threshold=dcfg["dedup"].get("threshold", 0.9))

    # 构造 (prompt, target)
    out = []
    for r in rows:
        prompt = build_prompt(r["problem"])
        target = build_target(r["solution"], r["answer"])
        out.append(
            {
                "prompt": prompt,
                "target": target,
                "answer": r["answer"],
                "source": r["source"],
                "bucket": r["bucket"],
            }
        )

    save_jsonl(out, args.out)

    print("\n[统计]")
    print("  source 分布:", dict(Counter(r["source"] for r in out)))
    print("  难度分布  :", dict(Counter(r["bucket"] for r in out)))

    # 落盘配置快照，保证可复现
    os.makedirs("outputs/sft", exist_ok=True)
    with open("outputs/sft/data_pipeline_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print("\n[样例]")
    print(out[0]["prompt"][:300])
    print("---")
    print(out[0]["target"][:300])


if __name__ == "__main__":
    main()
