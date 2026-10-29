"""E2 驱动脚本 · CPT 掉格式率 -> SFT 能否收回

一条命令跑完四点评测所需的全部训练与评测，产物全部落盘、可复现：

  [train] arm=base  : Qwen2.5-0.5B-Instruct      + GSM8K SFT(200 step)
  [train] arm=cpt   : E1 的 replay10_strong/final + GSM8K SFT(200 step, 超参完全一致)
  [eval ] 四个点，统一 n=200（E1 用的是 n=100，方差偏大，这里加大样本量）

四点为：
  1) base_orig        未 CPT 未 SFT（参考下界之上）
  2) cpt_replay10     只做 CPT（格式率受损点）
  3) sft_from_base    只做 SFT（参考组）
  4) sft_from_cpt     CPT 后再 SFT（实验组）

环境变量：
  E2_STEPS      SFT 步数（默认 200）
  E2_N          GSM8K 评测样本数（默认 200）
  E2_SKIP_TRAIN 置 1 时跳过训练，只重跑评测
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV = "D:/posttrain-lab-venv/.venv/Scripts/python.exe"
BASE_MODEL = "D:/models/Qwen2.5-0.5B-Instruct"
CPT_MODEL = "D:/data/cpt_runs/replay10_strong/final"
SFT_ROOT = "D:/data/sft_runs"
RES_DIR = os.path.join(ROOT, "experiments", "cpt_domain", "results_e2")

STEPS = int(os.environ.get("E2_STEPS", "200"))
N = int(os.environ.get("E2_N", "200"))
SKIP_TRAIN = os.environ.get("E2_SKIP_TRAIN", "0") == "1"

ENV = dict(os.environ)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    ENV.pop(k, None)
ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def run(cmd: list[str], tag: str) -> None:
    print(f"\n===== {tag} =====", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def train(arm: str, init: str) -> str:
    out = os.path.join(SFT_ROOT, f"e2_{arm}")
    if not SKIP_TRAIN:
        run([VENV, "-u", "experiments/cpt_domain/train_sft_recover.py",
             "--arm", arm, "--init", init, "--max-steps", str(STEPS),
             "--out-root", SFT_ROOT], f"TRAIN arm={arm}")
    return os.path.join(out, "final")


def evaluate(model: str, tag: str) -> dict:
    run([VENV, "-u", "experiments/cpt_domain/eval_cpt.py",
         "--model", model, "--tag", tag, "--n-gsm8k", str(N),
         "--out-root", RES_DIR], f"EVAL {tag}")
    with open(os.path.join(RES_DIR, f"eval_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    os.makedirs(RES_DIR, exist_ok=True)

    base_sft = train("base", BASE_MODEL)
    cpt_sft = train("cpt", CPT_MODEL)

    points = [
        (BASE_MODEL, "e2_base_orig"),
        (CPT_MODEL, "e2_cpt_replay10"),
        (base_sft, "e2_sft_from_base"),
        (cpt_sft, "e2_sft_from_cpt"),
    ]
    res = {tag: evaluate(m, tag) for m, tag in points}

    table = {tag: {
        "gsm8k_acc": r["gsm8k"]["accuracy"],
        "format_rate": r["gsm8k"]["format_rate"],
        "fmt_boxed": r["gsm8k"]["fmt_boxed"],
        "fmt_reasoning": r["gsm8k"]["fmt_reasoning"],
        "domain_ppl": r["domain"]["ppl"],
        "general_ppl": r["general"]["ppl"],
    } for tag, r in res.items()}

    out = os.path.join(RES_DIR, "e2_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": N, "steps": STEPS, "points": table}, f, ensure_ascii=False, indent=2)

    print("\n=== E2 SUMMARY ===", flush=True)
    hdr = f"{'point':<20}{'acc':>8}{'fmt':>8}{'boxed':>8}{'reason':>8}{'dom_ppl':>9}{'gen_ppl':>9}"
    print(hdr, flush=True)
    for tag, v in table.items():
        print(f"{tag:<20}{v['gsm8k_acc']:>8.3f}{v['format_rate']:>8.3f}"
              f"{v['fmt_boxed']:>8.3f}{v['fmt_reasoning']:>8.3f}"
              f"{v['domain_ppl']:>9.2f}{v['general_ppl']:>9.2f}", flush=True)
    print(f"\n[done] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
