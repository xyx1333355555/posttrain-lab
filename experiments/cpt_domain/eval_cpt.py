"""E1 · CPT 评测：领域困惑度 / 通用困惑度 / GSM8K 遗忘检测

三个指标一起看，才叫「垂直领域 CPT」的完整结论：
  ppl_domain  ↓ 说明领域适配成功
  ppl_general ↑ 说明通用语言能力受损
  gsm8k_acc   ↓ 说明出现灾难性遗忘（能力回归）

用法：
  python experiments/cpt_domain/eval_cpt.py --model D:/models/Qwen2.5-0.5B-Instruct --tag base_orig
  python experiments/cpt_domain/eval_cpt.py --model outputs/cpt/pure_domain/final --tag pure_domain
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from posttrain_lab.eval_utils import (  # noqa: E402
    extract_boxed_answer,
    is_math_equivalent,
    parse_gsm8k_answer,
)
from posttrain_lab.prompts import build_prompt, strip_thinking  # noqa: E402


def read_lines(p: str) -> list[str]:
    with open(p, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


@torch.no_grad()
def perplexity(model, tok, docs: list[str], seq_len: int, max_blocks: int, dev: str) -> dict:
    """按定长块计算 token 级平均负对数似然（= 困惑度）。"""
    stream: list[int] = []
    eos = tok.eos_token_id or tok.pad_token_id
    need = max_blocks * seq_len
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"]
        stream.extend(ids)
        stream.append(eos)
        if len(stream) >= need:
            break
    n_blocks = min(max_blocks, len(stream) // seq_len)
    total_nll, total_tok = 0.0, 0
    for b in range(n_blocks):
        chunk = torch.tensor([stream[b * seq_len:(b + 1) * seq_len]], device=dev)
        out = model(input_ids=chunk, labels=chunk)
        n_tok = chunk.numel() - 1
        total_nll += out.loss.item() * n_tok
        total_tok += n_tok
    avg_nll = total_nll / max(1, total_tok)
    return {"ppl": round(math.exp(min(avg_nll, 20)), 4), "avg_nll": round(avg_nll, 4),
            "blocks": n_blocks, "tokens": total_tok}


@torch.no_grad()
def gsm8k_acc(model, tok, n: int, max_new_tokens: int, bs: int, dev: str) -> dict:
    df = pd.read_parquet("D:/data/gsm8k/test.parquet").head(n)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [build_prompt(q, tok) for q in df["question"].tolist()]
    correct, fmt, n_box, n_reason, outs = 0, 0, 0, 0, []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for txt, gold_raw in zip(texts, df["answer"].tolist()[i:i + bs]):
            txt = strip_thinking(txt)
            gold = parse_gsm8k_answer(gold_raw)
            pred = extract_boxed_answer(txt)
            ok = bool(is_math_equivalent(pred, gold))
            correct += int(ok)
            has_box = "\\boxed" in txt
            has_reason = "<reasoning>" in txt
            fmt += int(has_box or has_reason)
            n_box += int(has_box)
            n_reason += int(has_reason)
            outs.append({"pred_ok": ok, "text": txt[:300]})
        print(f"  gsm8k {min(i+bs, len(prompts))}/{len(prompts)}  acc={correct/max(1,min(i+bs,len(prompts))):.3f}", flush=True)
    return {"accuracy": round(correct / len(prompts), 4), "format_rate": round(fmt / len(prompts), 4),
            "fmt_boxed": round(n_box / len(prompts), 4), "fmt_reasoning": round(n_reason / len(prompts), 4),
            "n": len(prompts), "samples": outs[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", default="D:/data/cpt_domain")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "outputs", "cpt"))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ppl-blocks", type=int, default=120)
    ap.add_argument("--n-gsm8k", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--skip-gsm8k", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    print(f"=== eval {args.tag} | model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    model.config.use_cache = True

    res: dict = {"tag": args.tag, "model": args.model}
    dom_val = read_lines(os.path.join(args.data_dir, "domain_val.txt"))
    rep_val = read_lines(os.path.join(args.data_dir, "replay_val.txt"))

    print("[1/3] 领域困惑度 …", flush=True)
    res["domain"] = perplexity(model, tok, dom_val, args.seq_len, args.ppl_blocks, dev)
    print("   ", res["domain"], flush=True)

    print("[2/3] 通用困惑度 …", flush=True)
    res["general"] = perplexity(model, tok, rep_val, args.seq_len, args.ppl_blocks, dev)
    print("   ", res["general"], flush=True)

    if not args.skip_gsm8k:
        print("[3/3] GSM8K（能力遗忘检测）…", flush=True)
        res["gsm8k"] = gsm8k_acc(model, tok, args.n_gsm8k, args.max_new_tokens, args.bs, dev)
        print("   ", {k: v for k, v in res["gsm8k"].items() if k != "samples"}, flush=True)

    if dev == "cuda":
        res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

    os.makedirs(args.out_root, exist_ok=True)
    p = os.path.join(args.out_root, f"eval_{args.tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n写了 {p}")
    print(json.dumps({k: v for k, v in res.items() if k != "gsm8k"}, ensure_ascii=False, indent=2))
    if "gsm8k" in res:
        print("gsm8k:", {k: v for k, v in res["gsm8k"].items() if k != "samples"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
