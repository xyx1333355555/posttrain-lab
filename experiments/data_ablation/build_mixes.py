#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后训练数据合成与配比消融 —— 数据飞轮与配比构造（可复现）
========================================================
从 GSM8K 数学推理语料出发，复现「质量过滤 → 去重 → 难度分桶 → 多源配比」的
数据飞轮，并构造固定总规模（N=1200）、不同难度配比的训练混合，供 GRPO 单变量
消融使用。

输出：
  experiments/data_ablation/mixes/<name>/train.parquet   （每行 question/answer，可直接被 03_train_grpo.py 读取）
  experiments/data_ablation/mixes/report.json            （去重/分桶/配比的真实统计，用于回填简历 L119）

难度代理：GSM8K 答案含 <<计算>> 标注，计算步数越多越难 → steps = answer.count("<<")
  easy: steps<=2   mid: 3-4   hard: >=5
"""
import os, re, json, random
from collections import Counter
import pandas as pd

SRC = "D:/data/gsm8k/train.parquet"
OUT = "experiments/data_ablation/mixes"
N = 1200
SEED = 42
random.seed(SEED)

# GSM8K 最终答案形如 '#### 72'
def parse_gsm8k_answer(sol):
    if sol is None:
        return None
    m = re.search(r"####\s*(.+?)\s*$", sol, re.S)
    if not m:
        return None
    return re.sub(r"[,\s]", "", m.group(1))


def norm(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def toks(s):
    return set(s.lower().split())


def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bucket(s):
    return "easy" if s <= 2 else ("mid" if s <= 4 else "hard")


def main():
    df = pd.read_parquet(SRC)
    print(f"[raw] GSM8K train 行数: {len(df)}")

    # 1) 质量过滤：答案可解析
    keep = []
    for _, r in df.iterrows():
        gold = parse_gsm8k_answer(r["answer"])
        if gold is None:
            continue
        steps = r["answer"].count("<<")
        keep.append({"question": r["question"], "answer": r["answer"],
                     "gold": gold, "steps": steps})
    print(f"[filter] 答案可解析: {len(keep)}")

    # 2) 去重：先精确去重（归一化问题），再近似去重（token Jaccard>=0.9）
    seen = set()
    dedup = []
    for r in keep:
        k = norm(r["question"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    print(f"[dedup] 精确去重后: {len(dedup)}（移除 {len(keep)-len(dedup)}）")

    # 近似去重：用 blocking（按 token 数 + 前 3 个排序 token 分桶）把比较限制在桶内，
    # 把 O(n^2) 降到近似线性，避免本地小机器跑超时。
    from collections import defaultdict
    block_accepted = defaultdict(list)
    near = []
    for r in dedup:
        t = toks(r["question"])
        key = (len(t), tuple(sorted(t)[:3]))
        dup = False
        for q in block_accepted[key]:
            if jac(t, q["_t"]) > 0.9:
                dup = True
                break
        if not dup:
            r["_t"] = t
            block_accepted[key].append(r)
            near.append(r)
    print(f"[dedup] 近似去重后: {len(near)}（再移除 {len(dedup)-len(near)}）", flush=True)

    # 3) 难度分桶
    for r in near:
        r["bucket"] = bucket(r["steps"])
    dist = Counter(r["bucket"] for r in near)
    print(f"[bucket] 难度分布: {dict(dist)}")

    def sample_mix(name, easy_p, hard_p, total=N):
        easy = [r for r in near if r["bucket"] == "easy"]
        mid = [r for r in near if r["bucket"] == "mid"]
        hard = [r for r in near if r["bucket"] == "hard"]
        ne = round(total * easy_p)
        nh = round(total * hard_p)
        nm = total - ne - nh

        def samp(pool, k):
            return random.sample(pool, min(k, len(pool)))

        sel = samp(easy, ne) + samp(mid, nm) + samp(hard, nh)
        random.shuffle(sel)
        sel = sel[:total]
        out = pd.DataFrame([{"question": r["question"], "answer": r["answer"]}
                            for r in sel])
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        out.to_parquet(os.path.join(d, "train.parquet"), index=False)
        comp = {"easy": len(samp(easy, ne)), "mid": len(samp(mid, nm)),
                "hard": len(samp(hard, nh))}
        print(f"  -> {name}: n={len(out)} composition={comp}")
        return len(out), comp

    report = {}
    # full：分层随机抽样（与总体分布近似）
    total_easy = dist.get("easy", 0)
    total_mid = dist.get("mid", 0)
    total_hard = dist.get("hard", 0)
    tot = total_easy + total_mid + total_hard
    p_e = total_easy / tot
    p_m = total_mid / tot
    p_h = total_hard / tot
    # 归一到 N
    ne_full = round(N * p_e); nh_full = round(N * p_h); nm_full = N - ne_full - nh_full

    for name, ne, nm, nh in [
        ("full", ne_full, nm_full, nh_full),
        ("easy_heavy", round(N * 0.70), N - round(N * 0.70) - round(N * 0.30), round(N * 0.30)),
        ("hard_heavy", round(N * 0.30), N - round(N * 0.30) - round(N * 0.70), round(N * 0.70)),
    ]:
        n, comp = sample_mix(name, ne / N, nh / N)
        report[name] = {"n": n, "composition": comp}

    report["_meta"] = {
        "src_rows": int(len(df)),
        "after_quality_filter": int(len(keep)),
        "after_exact_dedup": int(len(dedup)),
        "after_near_dedup": int(len(near)),
        "bucket_distribution": {k: int(v) for k, v in dist.items()},
        "N_per_mix": N,
        "seed": SEED,
    }
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[done] report ->", os.path.join(OUT, "report.json"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
