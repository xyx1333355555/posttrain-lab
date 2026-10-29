"""数据下载、清洗、去重、难度分桶。

注意两条工程纪律：
  1. 数据预处理不要用 GPU 机器 —— 在租卡平台上用「无卡模式」跑，能省掉一半预算
  2. 一定要先 limit 小样本跑通，再全量
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any, Optional

from .eval_utils import parse_gsm8k_answer, is_math_equivalent


def _safe_load(name, **kwargs):
    """下载失败时不要中断整条 pipeline，记录到 skip.log 后继续。"""
    from datasets import load_dataset

    try:
        return load_dataset(name, **kwargs)
    except Exception as e:  # noqa: BLE001
        os.makedirs("logs", exist_ok=True)
        with open("logs/skip.log", "a", encoding="utf-8") as f:
            f.write(f"[SKIP] {name} kwargs={kwargs} err={e}\n")
        print(f"[warn] 数据集加载失败，已跳过: {name} -> {e}")
        return None


# ---------------------------------------------------------------- 归一化
def normalize_text(s: str) -> str:
    """用于去重：抹掉数字、空白、标点，只保留语义骨架。"""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\d+(\.\d+)?", "#", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\u4e00-\u9fff#]", "", s)
    return s.strip()


def dedup(rows: List[Dict[str, Any]], key: str = "problem", threshold: float = 0.9) -> List[Dict[str, Any]]:
    """normalized 去重：用题目文本的归一化签名做精确去重。

    threshold < 1.0 时启用「前缀 shingle 近似」判断，用于识别同题改写。
    """
    seen = set()
    out = []
    for r in rows:
        sig = normalize_text(r.get(key, ""))
        if not sig:
            continue
        if threshold >= 1.0:
            token = sig
        else:
            # 取前 64 字符做近似键，能拦掉大部分同题微调改写
            token = sig[:64]
        if token in seen:
            continue
        seen.add(token)
        out.append(r)
    print(f"[dedup] {len(rows)} -> {len(out)}")
    return out


# ---------------------------------------------------------------- 难度分桶
def difficulty_bucket(sol: str) -> str:
    """用解的长度作为难度的廉价代理（无外部依赖，可解释）。

    easy   : < 400 字符
    medium : 400 - 1200
    hard   : > 1200
    """
    n = len(sol or "")
    if n < 400:
        return "easy"
    if n < 1200:
        return "medium"
    return "hard"


# ---------------------------------------------------------------- 加载器
def load_source(cfg_src: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按配置加载单个数据源，统一产出 {problem, solution, answer, source, bucket}。"""
    name = cfg_src["name"]
    kwargs = {}
    if cfg_src.get("config"):
        kwargs["name"] = cfg_src["config"]
    ds = _safe_load(name, split=cfg_src.get("split", "train"), **kwargs)
    if ds is None:
        return []

    limit = cfg_src.get("limit")
    rows = []
    pk = cfg_src.get("problem_key", "problem")
    sk = cfg_src.get("solution_key", "answer")
    ak = cfg_src.get("answer_key")

    for ex in ds:
        problem = ex.get(pk)
        solution = ex.get(sk)
        if not problem or not solution:
            continue

        if name.endswith("gsm8k"):
            answer = parse_gsm8k_answer(solution)
        else:
            answer = ex.get(ak) if ak else None
            # OpenMathInstruct-2 的 expected_answer 可能是 list
            if isinstance(answer, (list, tuple)):
                answer = answer[0] if answer else None
            if answer is None:
                # 从最后一个 \boxed / #### 里兜底抽
                from .eval_utils import extract_boxed_answer

                answer = extract_boxed_answer(solution)

        if answer is None or str(answer).strip() == "":
            continue

        rows.append(
            {
                "problem": str(problem),
                "solution": str(solution),
                "answer": str(answer),
                "source": name.split("/")[-1],
                "bucket": difficulty_bucket(solution),
            }
        )
        if limit and len(rows) >= limit:
            break

    print(f"[load] {name}: {len(rows)} 条")
    return rows


def apply_filters(rows: List[Dict[str, Any]], fcfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    lo = fcfg.get("min_solution_chars", 0)
    hi = fcfg.get("max_solution_chars", 10**9)
    out = [r for r in rows if lo <= len(r["solution"]) <= hi]
    print(f"[filter] length {lo}-{hi}: {len(rows)} -> {len(out)}")
    # 丢弃答案不可解析的样本，否则会给 RL 引入噪声
    good = [r for r in out if r.get("answer")]
    return good


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] {len(rows)} -> {path}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- 配比混合
def mix_by_ratio(pools: Dict[str, List[Dict[str, Any]]], ratio: Dict[str, float], total: int) -> List[Dict[str, Any]]:
    """按 (source -> 比例) 采样，用于 P1 的配比消融实验。"""
    import random

    rng = random.Random(42)
    out: List[Dict[str, Any]] = []
    s = sum(ratio.values()) or 1.0
    for src, w in ratio.items():
        need = int(total * w / s)
        pool = list(pools.get(src, []))
        rng.shuffle(pool)
        out.extend(pool[:need])
    rng.shuffle(out)
    return out
