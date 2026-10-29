#!/usr/bin/env python
"""Step 3 —— GRPO 强化。

这张脚本是整个项目的技术核心，也是面试时你被追问最多的地方。
跑之前请务必确认三件事：
  1. configs/grpo.yaml 里 model_name_or_path 指向的是 **Step 2 产出的 SFT ckpt**
  2. per_device_train_batch_size 能被 num_generations 整除（否则 TRL 直接报错）
  3. vllm colocate 的显存占比留够：训练侧 + vLLM 侧加起来 < 显存总量

用法：
    python scripts/03_train_grpo.py --config configs/grpo.yaml
    python scripts/03_train_grpo.py --config configs/grpo.yaml --ablate no_length
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from posttrain_lab.eval_utils import parse_gsm8k_answer  # noqa: E402
from posttrain_lab.prompts import build_prompt  # noqa: E402
from posttrain_lab.viz import plot_all, summary_table  # noqa: E402

# ---------------------------------------------------------------- Windows 多进程补丁
# 在 Windows + RTX 4060(8GB) 本地环境，TRL 的 GRPO 生成阶段会尝试 spawn 多进程
# DataLoader，但 Windows 句柄继承失败会疯狂报 [WinError 6] 句柄无效，把每步拖死。
# 小数据(SFT/GRPO 全在内存)本就不需要多 worker，这里强制 num_workers=0，从根上
# 消除 spawn，训练恢复干净且更快。仅作用于本进程。
import torch.utils.data as _tud


def _force_single_process_dataloader():
    _OrigDataLoader = _tud.DataLoader

    class _SingleProcDataLoader(_OrigDataLoader):
        def __init__(self, *args, **kwargs):
            # 显式关闭多进程 worker + 持久 worker，避免 Windows 句柄继承失败
            kwargs["num_workers"] = 0
            kwargs["persistent_workers"] = False
            super().__init__(*args, **kwargs)

    _tud.DataLoader = _SingleProcDataLoader
    # 同时关掉 tokenizers 的并行（它内部也会尝试 fork/spawn）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_force_single_process_dataloader()


# ---------------------------------------------------------------- 配置兼容
def build_grpo_config(cfg: dict, out_dir: str, smoke: bool):
    """不同 TRL 版本的 GRPOConfig 字段名有差异（例如 vllm_* 参数在部分版本已改名）。

    这里用 dataclass 反射做过滤：不支持的字段直接丢弃并打印，避免一上来就崩。
    """
    from trl import GRPOConfig

    g = cfg["grpo"]
    t = cfg["train"]
    v = cfg["vllm"]

    # ---- 硬约束前置校验：宁可在这里崩，也不要排在 GPU 队列里才发现 ----
    G, b = g["num_generations"], t["per_device_train_batch_size"]
    if b % G != 0:
        raise SystemExit(
            f"[fatal] per_device_train_batch_size={b} 必须能被 num_generations={G} 整除。"
            f"建议改成 {b - b % G} 或 {b + G - b % G}"
        )
    spg = g.get("steps_per_generation", t["gradient_accumulation_steps"])
    if t["gradient_accumulation_steps"] % spg != 0:
        raise SystemExit(
            f"[fatal] gradient_accumulation_steps={t['gradient_accumulation_steps']} "
            f"必须能被 steps_per_generation={spg} 整除"
        )
    print(f"[check] 每条设备 batch = {b // G} 个 prompt × {G} 次采样；"
          f"优化器每步 = {b // G * t['gradient_accumulation_steps']} 个 prompt")

    want = dict(
        output_dir=out_dir,
        seed=cfg.get("seed", 42),
        learning_rate=g["learning_rate"],
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_grad_norm=0.5,

        # --- GRPO 本体 ---
        num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"],
        max_prompt_length=g["max_prompt_length"],
        temperature=g["temperature"],
        top_p=g["top_p"],
        beta=g["beta"],
        epsilon=g["epsilon"],
        epsilon_high=g.get("epsilon_high"),
        scale_rewards=g["scale_rewards"],
        steps_per_generation=g.get("steps_per_generation", t["gradient_accumulation_steps"]),
        num_iterations=g.get("num_iterations", 1),

        # --- 训练 ---
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=1,
        optim=t["optim"],
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        # 非重入式检查点：reentrant=True 要求有输入张量 requires_grad，否则 policy 的
        # per_token_logps 被 detach，GRPO loss.backward 报 “does not require grad”。
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if t.get("gradient_checkpointing", True) else None
        ),
        logging_steps=5,
        save_strategy="steps",
        save_steps=10 if smoke else t["save_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=[],
        # Windows 下 DataLoader 多进程 worker 会因句柄继承失败（WinError 6 / Bad file descriptor）
        # 而反复崩溃重生，导致每个 batch 取数都卡住。直接关掉 worker 进程，主进程取数反而更快更稳。
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,

        # --- vLLM colocate（Windows / 无 vLLM 时关掉，改用模型自身 generate）---
        use_vllm=v["use_vllm"],
    )
    if v["use_vllm"]:
        want["vllm_mode"] = v["vllm_mode"]
        want["vllm_gpu_memory_utilization"] = v["vllm_gpu_memory_utilization"]
        want["vllm_tensor_parallel_size"] = v.get("vllm_tensor_parallel_size", 1)
    # sleep mode 在部分 TRL 版本里没有这个字段，交给下面的反射过滤
    if "vllm_enable_sleep_mode" in {f.name for f in fields(GRPOConfig)}:
        want["vllm_enable_sleep_mode"] = v.get("vllm_enable_sleep_mode", True)
    want = {k: v_ for k, v_ in want.items() if v_ is not None}

        # loss_type 只在较新版本存在
    if "loss_type" in {f.name for f in fields(GRPOConfig)}:
        want["loss_type"] = g.get("loss_type", "dapo")

    # 最大步数上限（可选）：免费平台 / 本地快速验证时用它来给训练「封顶」
    if "max_steps" in g and g["max_steps"]:
        want["max_steps"] = int(g["max_steps"])

    supported = {f.name for f in fields(GRPOConfig)}
    dropped = [k for k in want if k not in supported]
    if dropped:
        print(f"[compat] 当前 TRL 版本不支持以下参数，已自动忽略: {dropped}")
        for k in dropped:
            want.pop(k)

    if t["mode"] == "lora":
        lo = t["lora"]
        print(f"[lora] r={lo['r']} alpha={lo['alpha']}（PEFT 配置在 Trainer 侧注入）")

    return GRPOConfig(**want), supported


# ---------------------------------------------------------------- 数据
def build_dataset(cfg: dict, tok, smoke: bool):
    d = cfg["dataset"]
    # 本地数据优先：设置 GSM8K_LOCAL 指向含 train.parquet/test.parquet 的目录，
    # 可绕过 hf-mirror 在部分网络下不稳定 / 超时的问题。
    local = os.environ.get("GSM8K_LOCAL")
    if local and d["name"] == "gsm8k":
        import pandas as pd
        fp = os.path.join(local, f"{d.get('split', 'train')}.parquet")
        ds = pd.read_parquet(fp).to_dict("records")
        print(f"[data] 使用本地 gsm8k: {fp} ({len(ds)} rows)")
    else:
        ds = load_dataset(d["name"], d.get("config"), split=d.get("split", "train"))
    rows = []
    for ex in ds:
        ans = parse_gsm8k_answer(ex.get("answer", ""))
        if ans is None:
            continue
        rows.append({"prompt": build_prompt(ex["question"], tok), "answer": ans})
        if d.get("limit") and len(rows) >= d["limit"]:
            break
    if smoke:
        rows = rows[:256]
    print(f"[data] GRPO prompt 数: {len(rows)}")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--ablate", default=None,
                    choices=[None, "no_length", "no_format", "no_scale", "high_kl"],
                    help="消融开关")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None,
                    help="运行时覆盖 model_name_or_path（SFT→GRPO 串联时用）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖 grpo.max_steps（快速验证用小步数，正式跑去掉）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.max_steps is not None:
        cfg.setdefault("grpo", {})["max_steps"] = args.max_steps
    if args.model:
        cfg["model_name_or_path"] = args.model
        # 强制同步 base_model_for_tokenizer，避免仍指向 HF 原名去联网（镜像不稳定时卡死）
        cfg["base_model_for_tokenizer"] = args.model
        print(f"[override] model_name_or_path -> {args.model}")

    # ---- 消融处理（每条改一个变量，其余保持不动，这是消融的唯一法则）----
    tag = "main"
    if args.ablate == "no_length":
        cfg["rewards"]["length_penalty"]["mode"] = "none"
        tag = "ablate_no_length"
    elif args.ablate == "no_format":
        cfg["rewards"]["weights"]["format"] = 0.0
        tag = "ablate_no_format"
    elif args.ablate == "no_scale":
        cfg["grpo"]["scale_rewards"] = True
        tag = "ablate_scale_on"
    elif args.ablate == "high_kl":
        cfg["grpo"]["beta"] = 0.04
        tag = "ablate_beta_0.04"

    out_dir = os.path.join(cfg["output_dir"], tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] tag={tag} -> {out_dir}")

    torch.manual_seed(cfg.get("seed", 42))

    base = cfg.get("base_model_for_tokenizer", cfg["model_name_or_path"])
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 生成用左 padding，避免右补 pad 污染生成

    grpo_cfg, _ = build_grpo_config(cfg, out_dir, args.smoke)
    dataset = build_dataset(cfg, tok, args.smoke)

    from posttrain_lab.rewards import build_reward_funcs
    cfg["_tokenizer"] = tok  # 让长度奖励用真实 token 数而非字符代理
    rewards = build_reward_funcs(cfg)
    cfg.pop("_tokenizer", None)  # 仅在构建奖励时临时用；pop 掉以免 yaml.dump 序列化 tokenizer 对象失败

    from trl import GRPOTrainer

    kwargs = dict(
        model=cfg["model_name_or_path"],
        reward_funcs=rewards,
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tok,
    )
    if cfg["train"]["mode"] == "lora":
        from peft import LoraConfig

        kwargs["peft_config"] = LoraConfig(
            r=cfg["train"]["lora"]["r"],
            lora_alpha=cfg["train"]["lora"]["alpha"],
            lora_dropout=cfg["train"]["lora"]["dropout"],
            target_modules=cfg["train"]["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(**kwargs)

    # ---- 关键性能修复：TRL 在 *训练模式* 下调用 model.generate()，
    # 触发 Qwen2 forward 的「gradient_checkpointing + training + use_cache -> use_cache=False」
    # 分支，导致自回归生成退化为 O(n^2)（单次 step 卡 30min+ 不动）。
    # 修复：仅在 generate() 调用期间临时关闭梯度检查点，让 KV cache 生效；
    # 生成结束后立刻恢复，训练前向（model(input_ids)）仍走梯度检查点省显存。
    _orig_generate = trainer.model.generate

    def _generate_with_kv_cache(self, *args, **kwargs):
        was_gc = self.is_gradient_checkpointing
        self.gradient_checkpointing_disable()
        try:
            return _orig_generate(*args, **kwargs)
        finally:
            if was_gc:
                self.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

    import types
    trainer.model.generate = types.MethodType(_generate_with_kv_cache, trainer.model)
    _gen_cfg = getattr(trainer, "generation_config", None)
    if _gen_cfg is not None:
        _gen_cfg.use_cache = True
    print("[perf] 已对 generation 阶段临时关闭梯度检查点以启用 KV cache（规避 O(n^2) 生成）")

    # ---- 免费平台三件套：时间预算 / 断点续跑 / 跨会话持久化 ----
    from posttrain_lab.callbacks import CheckpointSync, TimeBudgetCallback

    max_min = float(os.environ.get("PTL_MAX_MINUTES", "0"))
    interrupt_dir = os.environ.get("PTL_INTERRUPT_DIR", os.path.join(out_dir, "interrupt"))
    if max_min > 0:
        trainer.add_callback(TimeBudgetCallback(max_min, save_dir=interrupt_dir))
        print(f"[free-tier] 时间预算 {max_min/60:.2f} h -> 到点存于 {interrupt_dir}")
    sync_repo = os.environ.get("PTL_SYNC_REPO", "")
    if sync_repo:
        every = int(cfg["train"].get("save_steps", 50)) * 2
        trainer.add_callback(CheckpointSync(every_steps=every,
                                            local_dir=out_dir, repo_id=sync_repo))
        print(f"[free-tier] ckpt 每 {every} 步同步到 {sync_repo}")

    resume_from = os.environ.get("PTL_RESUME", "") or None
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(os.path.join(out_dir, "final"))

    # ---- 日志 -> 曲线 ----
    hist = trainer.state.log_history
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for h in hist:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    plot_all(metrics_path, out_dir=os.path.join(out_dir, "figs"))
    summ = summary_table(metrics_path)
    print("\n[summary]", json.dumps(summ, indent=2)[:1200])

    # 配置快照（可复现的凭证）
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print("[done]", out_dir)


if __name__ == "__main__":
    main()
