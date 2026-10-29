#!/usr/bin/env python
"""免费算力平台一键运行器（Kaggle / Colab / 魔搭 ModelScope 通用）。

它解决三件你在免费平台上一定会遇到的事：
  1. **自动识别环境**并选择合适的精度 / 注意力实现 / 模型规模
  2. **按时间预算优雅停止**，到点前自动存盘 —— 免费平台会话会被强制打断
  3. **支持跨会话续跑**：Session 1 跑 11 小时 → Session 2 从 checkpoint 接着跑

典型用法（Kaggle 一个 cell 就够）：
    !git clone https://github.com/<you>/posttrain-lab.git && cd posttrain-lab
    !pip install -r requirements.txt -q
    !python scripts/99_run_free_tier.py --stage grpo --hours 10.5

    # 下一周继续跑：
    !python scripts/99_run_free_tier.py --stage grpo --hours 10.5 --resume outputs/grpo_16gb/interrupt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from posttrain_lab import runtime  # noqa: E402


def banner(info: dict) -> None:
    print("=" * 68)
    print(f"  平台        : {info['platform']}")
    print(f"  GPU         : {info['gpu']['name']}  {info['gpu']['vram_gb']} GB")
    print(f"  自动精度    : {info['precision']}")
    print(f"  注意力实现  : {info['attn']}")
    print(f"  推荐模型    : {info['recommended_model']}")
    print(f"  工作目录    : {info['paths']['repo']}")
    print("=" * 68)


def ensure_repo(platform: str, p: dict) -> None:
    """本地/Colab 上没有仓库就拉一份；Kaggle 通常是你手动上传或用 dataset 挂载。"""
    if platform in ("kaggle", "colab") and not os.path.isdir(p["repo"]):
        print("[setup] 未检测到仓库，请先把 posttrain-lab 放到:", p["repo"])
    for d in (p["outputs"], p["data"]):
        os.makedirs(d, exist_ok=True)


def preflight(args, p: dict) -> None:
    """上机前的最后一道检查。

    这里拦的全部是「跑了两小时才发现白跑」的那类问题。
    """
    print("-" * 68)
    print("[preflight]")

    # 1) 持久化：这是免费平台的头号杀手
    if not os.environ.get("HF_TOKEN"):
        print("  [FATAL] 未检测到 HF_TOKEN —— 会话一断，所有 checkpoint 全部蒸发。")
        print("          → 到 https://huggingface.co/settings/tokens 申请 write token")
        print("          → 平台里设为名为 HF_TOKEN 的环境变量/密钥")
        print("          → 然后带上 --sync-repo <你的私有仓库id>")
        raise SystemExit(2)
    print("  [ok] HF_TOKEN 已就绪")

    if not args.sync_repo:
        print("  [FATAL] 未指定 --sync-repo，本次产出不会离开这台机器。")
        print("          → 加一句 --sync-repo <用户名>/<私有仓库名>")
        raise SystemExit(2)
    print(f"  [ok] 产出将同步到 {args.sync_repo}")

    # 2) 依赖是否真的装上
    missing = []
    for m in ("torch", "transformers", "datasets", "trl"):
        try:
            __import__(m)
        except Exception:  # noqa: BLE001
            missing.append(m)
    if missing:
        print(f"  [FATAL] 缺少依赖 {missing} —— 先 pip install -r requirements.txt")
        raise SystemExit(2)
    try:
        import trl

        print(f"  [ok] 依赖已安装 (trl {trl.__version__})")
    except Exception:
        print("  [ok] 依赖已安装")

    # 3) 磁盘配额（免费平台通常只有几十 GB，ckpt 很占地方）
    try:
        st = os.statvfs(p["base"])
        free_gb = st.f_bavail * st.f_frsize / 1024**3
        print(f"  [ok] 可用磁盘 {free_gb:.1f} GB")
        if free_gb < 8:
            print("  [warn] 磁盘不足 8GB，save_total_limit 建议调到 1")
    except Exception:  # noqa: BLE001
        pass

    # 4) 时间预算必须小于会话上限
    if args.hours >= 11.5:
        print(f"  [warn] --hours={args.hours} 已逼近 12h 强制断连上限，建议设 10.5")
    print("-" * 68)


def choose_config(platform: str, vram_gb: float, stage: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if stage == "sft":
        return "configs/sft.yaml" if vram_gb >= 22 else "configs/sft_16gb.yaml"
    if stage == "grpo":
        if vram_gb >= 22:
            return "configs/grpo_a10.yaml"
        if vram_gb >= 14:
            return "configs/grpo_16gb.yaml"
        # 8GB 级（笔记本 4060 / 3050 等）：0.5B + LoRA + 关 vLLM
        return "configs/grpo_8gb.yaml"
    return "configs/sft_16gb.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="grpo", choices=["probe", "sft", "grpo", "eval"])
    ap.add_argument("--hours", type=float, default=10.5,
                    help="本次会话的时间预算（务必小于平台会话上限，Kaggle/Colab 均为 12h）")
    ap.add_argument("--config", default=None, help="显式指定配置文件")
    ap.add_argument("--resume", default=None, help="从哪个 ckpt 续跑")
    ap.add_argument("--smoke", action="store_true", help="小样本快速验证，强烈建议第一次先跑它")
    ap.add_argument("--sync-repo", default=None, help="HF 私有仓库 id，用于跨会话持久化 ckpt")
    ap.add_argument("--probe", action="store_true", help="只探测环境就退出，不消耗任何配额")
    ap.add_argument("--no-preflight", action="store_true", help="跳过 HF_TOKEN / 依赖检查（调试用）")
    args = ap.parse_args()

    info = runtime.summary()
    banner(info)

    p = info["paths"]
    ensure_repo(info["platform"], p)

    if args.stage == "probe" or args.probe:
        print("[probe] 只做环境探测，到此结束（未消耗配额）")
        return

    if not args.no_preflight:
        preflight(args, p)

    cfg_path = choose_config(info["platform"], info["gpu"]["vram_gb"], args.stage, args.config)
    print(f"[config] {cfg_path}")

    # 把自动选择的精度/attn 通过环境变量透传给下游脚本
    os.environ["PTL_DTYPE"] = info["precision"]
    os.environ["PTL_ATTN"] = info["attn"]
    os.environ["PTL_MAX_MINUTES"] = str(args.hours * 60)
    os.environ["PTL_INTERRUPT_DIR"] = os.path.join(p["outputs"], f"{args.stage}_interrupt")
    os.environ["PTL_SYNC_REPO"] = args.sync_repo or ""
    if args.resume:
        os.environ["PTL_RESUME"] = args.resume

    SCRIPT = {
        "sft": "scripts/02_train_sft.py",
        "grpo": "scripts/03_train_grpo.py",
        "eval": "scripts/04_eval.py",
    }
    script = SCRIPT[args.stage]
    if not os.path.isfile(script):
        raise SystemExit(f"[fatal] 找不到脚本 {script}（当前目录 {os.getcwd()}）")

    t0 = time.time()
    cmd = [sys.executable, script, "--config", cfg_path]
    if args.smoke:
        cmd.append("--smoke")
    if args.resume:
        cmd += ["--ckpt", args.resume] if args.stage == "eval" else []
    print("[run]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)

    print(f"\n[done] 退出码 {proc.returncode}，本次会话用掉 {(time.time()-t0)/3600:.2f} 小时")

    # ---- 收尾：把产出推到持久化位置 ----
    if args.sync_repo:
        runtime.save_checkpoint_hub(os.path.join(p["outputs"], args.stage), args.sync_repo)
    else:
        tail = os.environ["PTL_INTERRUPT_DIR"]
        print(f"[提醒] 会话结束后 {p['base']} 下的数据不保证保留。")
        print(f"       请手动下载 {tail}，或下次带上 --sync-repo <你的HF私有仓库>。")

    json.dump(
        {"stage": args.stage, "hours_used": round((time.time() - t0) / 3600, 2), "config": cfg_path},
        open(os.path.join(p["base"], f"{args.stage}_session.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    main()
