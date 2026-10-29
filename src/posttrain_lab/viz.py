"""四件套曲线：loss / reward / length / (KL)。

这四张图是本项目最重要的交付物之一 —— 面试时你是把它们摊开来讲的。
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any


def read_metrics(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


KEYS = [
    ("loss", "Training Loss"),
    ("reward", "Mean Reward"),
    ("completions/mean_length", "Mean Completion Length"),
    ("kl", "KL"),
]


def plot_all(metrics_path: str, out_dir: str = "outputs/figs") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_metrics(metrics_path)
    if not rows:
        print(f"[warn] {metrics_path} 为空，跳过绘图")
        return
    os.makedirs(out_dir, exist_ok=True)

    steps = [r.get("step", i) for i, r in enumerate(rows)]
    for i, (key, title) in enumerate(KEYS):
        ys = [r.get(key) for r in rows]
        if all(y is None for y in ys):
            continue
        ys = [y if y is not None else float("nan") for y in ys]
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=140)
        ax.plot(steps, ys, lw=1.6, color="#2b6cb0")
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{key.replace('/', '_')}.png")
        fig.savefig(p)
        plt.close(fig)
        print(f"[fig] {p}")


def summary_table(metrics_path: str) -> Dict[str, Any]:
    rows = read_metrics(metrics_path)
    if not rows:
        return {}
    out = {}
    for key, _ in KEYS:
        ys = [r.get(key) for r in rows if r.get(key) is not None]
        if ys:
            out[key] = {"first": ys[0], "last": ys[-1], "min": min(ys), "max": max(ys)}
    return out
