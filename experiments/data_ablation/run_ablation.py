#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后训练数据合成与配比消融 —— 驱动脚本（可复现）
========================================================
对固定规模（N=1200）但不同难度配比的训练混合做 GRPO 单变量消融：
  easy_heavy : 840 简单 + 360 困难
  hard_heavy : 360 简单 + 840 困难
每个混合跑相同步数（默认 50 步，可由 PTL_AB_STEPS 覆盖），用 GSM8K 测试集测准确率，
给出「难度配比 → GSM8K 准确率」的真实结论，回填简历 L119/L121。

环境变量：
  PTL_AB_STEPS   每混合训练步数（默认 50）
  PTL_AB_LIMIT   评测样本数（默认 400）
  PTL_AB_MIXES   逗号分隔的混合名（默认 easy_heavy,hard_heavy）
  PTL_EVAL_LOCAL 评测用的 GSM8K 本地目录（需含 test.parquet）
"""
import os, sys, json, subprocess, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV = "D:/posttrain-lab-venv/.venv/Scripts/python.exe"
BASE_CFG = os.path.join(ROOT, "configs", "grpo_8gb.yaml")
MIX_BASE = os.path.join(ROOT, "experiments", "data_ablation", "mixes")
MODEL = "D:/models/Qwen2.5-0.5B-Instruct"
EVAL_LOCAL = os.environ.get("PTL_EVAL_LOCAL", "D:/data/gsm8k")

MAX_STEPS = int(os.environ.get("PTL_AB_STEPS", "50"))
LIMIT = int(os.environ.get("PTL_AB_LIMIT", "400"))
MIXES = os.environ.get("PTL_AB_MIXES", "easy_heavy,hard_heavy").split(",")


def main():
    base_cfg = yaml.safe_load(open(BASE_CFG, "r", encoding="utf-8"))
    results = {}
    for mix in MIXES:
        mix_dir = os.path.abspath(os.path.join(MIX_BASE, mix))
        out_dir = os.path.join(ROOT, "outputs", "ablation", mix, "main")
        cfg = dict(base_cfg)
        cfg["output_dir"] = f"outputs/ablation/{mix}"
        cfg["dataset"]["limit"] = 99999
        cfg["train"]["save_steps"] = MAX_STEPS
        tmp = os.path.join(ROOT, f"configs/_ablation_{mix}.yaml")
        yaml.safe_dump(cfg, open(tmp, "w", encoding="utf-8"), allow_unicode=True)

        # ---- 训练：GSM8K_LOCAL 指向该混合的 train.parquet ----
        train_env = dict(os.environ, GSM8K_LOCAL=mix_dir)
        print(f"\n===== TRAIN {mix} (steps={MAX_STEPS}) =====", flush=True)
        subprocess.run(
            [VENV, "scripts/03_train_grpo.py", "--config", tmp,
             "--max-steps", str(MAX_STEPS), "--model", MODEL],
            cwd=ROOT, env=train_env, check=True)

        # ---- 评测：GSM8K_LOCAL 指向含 test.parquet 的目录 ----
        ckpt = os.path.join(out_dir, f"checkpoint-{MAX_STEPS}")
        if not os.path.isdir(ckpt):
            ckpt = os.path.join(out_dir, "final")
        eval_out = os.path.join(out_dir, "eval")
        eval_env = dict(os.environ, GSM8K_LOCAL=EVAL_LOCAL)
        print(f"\n===== EVAL {mix} (ckpt={ckpt}, limit={LIMIT}) =====", flush=True)
        subprocess.run(
            [VENV, "scripts/04_eval.py", "--ckpt", ckpt,
             "--tasks", "gsm8k", "--limit", str(LIMIT), "--out", eval_out],
            cwd=ROOT, env=eval_env, check=True)

        res_path = os.path.join(eval_out, "eval_results.json")
        with open(res_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        results[mix] = {"ckpt": ckpt, "gsm8k": res["results"]["gsm8k"]}
        print(f"[result] {mix}: {results[mix]['gsm8k']}", flush=True)

    out_json = os.path.join(ROOT, "experiments", "data_ablation", "ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
