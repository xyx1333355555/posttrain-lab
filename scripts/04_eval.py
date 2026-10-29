#!/usr/bin/env python
"""Step 4 —— 评测闭环。

评测必须和训练用**同一套 prompt 模板**（见 posttrain_lab/prompts.py），
否则训练 A 格式、评测 B 格式，分数再低都不知道是不是自己的锅。

指标：
  - accuracy       : 答案正确率（math-verify / 数值等价判定）
  - format_rate    : 输出结构合规率（能抽出 \boxed{} 的比例）
  - avg_tokens     : 平均生成长度（长度膨胀的观测口）
  - throughput     : tokens/s（Infra 岗最爱问的效率指标）

实现说明：默认用 transformers 原生 generate（无需 vLLM，Windows / 8GB 卡也能跑）。
若环境里有 vLLM 会自动改用它加速（见 VLLM_AVAILABLE）。

用法：
    python scripts/04_eval.py --ckpt outputs/grpo_8gb/main/final --tasks gsm8k --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from posttrain_lab.eval_utils import extract_boxed_answer, is_math_equivalent, parse_gsm8k_answer  # noqa: E402
from posttrain_lab.prompts import build_prompt  # noqa: E402

try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_AVAILABLE = True
except Exception:  # noqa: BLE001
    VLLM_AVAILABLE = False


def load_task(name: str, limit: int | None = None) -> List[Tuple[str, str]]:
    """返回 [(question, gold_answer)]。"""
    rows = []
    if name == "gsm8k":
        local = os.environ.get("GSM8K_LOCAL")
        if local:
            import pandas as pd
            fp = os.path.join(local, "test.parquet")
            ds = pd.read_parquet(fp).to_dict("records")
            print(f"[eval] 使用本地 gsm8k: {fp} ({len(ds)} rows)")
        else:
            ds = load_dataset("openai/gsm8k", "main", split="test")
        for ex in ds:
            gold = parse_gsm8k_answer(ex["answer"])
            if gold is None:
                continue
            rows.append((ex["question"], gold))
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for ex in ds:
            rows.append((ex["problem"], ex["answer"]))
    else:
        raise ValueError(f"未知任务: {name}")
    if limit:
        rows = rows[:limit]
    return rows


def reformat_gold(ans: str) -> str:
    """MATH-500 的答案是 LaTeX，取最后一个 \\boxed 的内容做 gold。"""
    b = extract_boxed_answer(ans)
    return b if b is not None else ans


def _resolve_base(ckpt: str) -> str:
    """若 ckpt 是 LoRA adapter 目录，返回其基座模型名；否则返回 ckpt 本身。"""
    peft_cfg = os.path.join(ckpt, "adapter_config.json")
    if os.path.exists(peft_cfg):
        with open(peft_cfg, "r", encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path", ckpt)
    return ckpt


def _safe_load_tokenizer(ckpt: str, base: str):
    from transformers import AutoTokenizer

    for cand in (ckpt, base):
        try:
            tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
            if tok is not None:
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                # decoder-only 生成必须用左 padding，否则右边补 pad 会污染生成结果
                tok.padding_side = "left"
                return tok
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(f"[fatal] 找不到 tokenizer：{ckpt} / {base}")


def _load_policy(ckpt: str, attn: str = "sdpa", dtype=torch.float16):
    """加载策略模型，兼容两种产物：
      - 完整模型目录（AutoModelForCausalLM 可直接加载）
      - LoRA adapter 目录（trainer.save_model 只存了 adapter）
    后者：基座 + PeftModel + merge_and_unload，得到可独立评测的权重。
    """
    from transformers import AutoModelForCausalLM

    base = _resolve_base(ckpt)
    if base != ckpt:
        print(f"[eval] 检测到 LoRA adapter，基座 = {base}")

    tok = _safe_load_tokenizer(ckpt, base)

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, attn_implementation=attn, trust_remote_code=True
    )
    if base != ckpt:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ckpt)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def generate_with_transformers(ckpt: str, prompts: List[str], max_new_tokens: int = 768):
    tok, model = _load_policy(ckpt)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    out_texts = []
    bs = 8
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        in_len = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0
            )
        for j in range(len(batch)):
            text = tok.decode(gen[j][in_len:], skip_special_tokens=True)
            out_texts.append(text)
    return out_texts


def generate_with_vllm(ckpt: str, prompts: List[str], max_new_tokens: int = 768):
    tok, model = _load_policy(ckpt)
    merged = os.path.join(ckpt, "_merged_vllm")
    model.save_pretrained(merged)
    tok.save_pretrained(merged)
    llm = LLM(model=merged, gpu_memory_utilization=0.75, max_model_len=2048, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, top_p=1.0)
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outputs]


def run(ckpt: str, tasks: List[str], limit: int | None, out: str) -> None:
    print(f"[eval] engine = {'vLLM' if VLLM_AVAILABLE else 'transformers(generate)'}  ckpt={ckpt}")
    results = {}
    for task in tasks:
        rows = load_task(task, limit)
        # 同一套 tokenizer / chat template 构造 prompt（训练 == 评测）
        tok = _safe_load_tokenizer(ckpt, _resolve_base(ckpt))
        prompts = [build_prompt(q, tok) for q, _ in rows]
        golds = [reformat_gold(a) for _, a in rows]

        t0 = time.time()
        if VLLM_AVAILABLE:
            texts = generate_with_vllm(ckpt, prompts)
        else:
            texts = generate_with_transformers(ckpt, prompts)
        dt = time.time() - t0

        n_ok = 0
        n_fmt = 0
        total_tokens = 0
        for text, gold in zip(texts, golds):
            # 粗略 token 估计（不依赖 tokenizer 精确计数）
            total_tokens += len(text) // 2
            pred = extract_boxed_answer(text)
            if pred is not None:
                n_fmt += 1
                if is_math_equivalent(pred, gold):
                    n_ok += 1

        n = len(rows)
        results[task] = {
            "n": n,
            "accuracy": round(100.0 * n_ok / n, 2),
            "format_rate": round(100.0 * n_fmt / n, 2),
            "wallclock_s": round(dt, 1),
            "throughput_tok_s": round(total_tokens / max(dt, 1e-6), 1),
        }
        print(f"[{task}] {results[task]}")

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "eval_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckpt, "engine": "vllm" if VLLM_AVAILABLE else "transformers", "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[save] {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="gsm8k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run(args.ckpt, tasks, args.limit, args.out)


if __name__ == "__main__":
    main()
