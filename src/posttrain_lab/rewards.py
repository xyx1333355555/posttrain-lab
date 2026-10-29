"""三路可验证奖励。

设计目标（面试要能讲出来的部分）：
  1. correctness —— 唯一真正重要的信号，用 math-verify 做数学等价判定而不是字符串相等
  2. format      —— 防止「格式崩溃」：一旦模型不再输出 \boxed{}，后续评测全挂且毫无信号
  3. length      —— 抑制 GRPO 中期的思维链长度膨胀，但不能一刀切砍短该长的地方

关于长度膨胀，社区公认的两个手段：
  - Dr. GRPO：scale_rewards=False（不按响应长度归一化 advantage）
  - DAPO：对超长序列做惩罚
本项目两者都用，并且把 length 项做成**可开关**的 ablation 变量。
"""
from __future__ import annotations

import re
from typing import Callable, List

from .eval_utils import extract_boxed_answer, is_math_equivalent

_BOXED_RE = re.compile(r"\\boxed\{")
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.S)


# ---------------------------------------------------------------- 工具
def _get_completion_text(completion) -> str:
    """TRL 在对话模式下传的是 messages list，在纯文本模式下传的是 str。两种都兼容。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _char_token_proxy(text: str) -> float:
    """粗略的 token 代理：4 字符 ≈ 1 token（英文技术文本的经验值）。

    只在**没有 tokenizer** 时用。用它做长度惩罚是可以的，但要注意：
    一旦你有 tokenizer，就用真的 token 数（见 make_length_reward 的 tokenizer 参数），
    否则中英混排时估算会偏，这也是一个会被面试官追问到第二层的点。
    """
    return len(text) / 4.0


# ---------------------------------------------------------------- 奖励 1：正确性
def _pick_completions(prompts, completions):
    """兼容 TRL 不同版本的奖励函数调用约定：
    - 新版按关键字传 prompts= / completions=
    - 旧版按位置传 (prompts, completions)
    - 个别版本只传单个参数
    """
    if completions is not None:
        return completions
    return prompts


def make_correctness_reward(answer_key: str = "answer") -> Callable:
    """可验证奖励的核心：比对模型输出里的 \\boxed{} 与 ground truth。"""

    def _fn(prompts=None, completions=None, **kwargs):
        golds = kwargs[answer_key]
        comps = _pick_completions(prompts, completions)
        out: List[float] = []
        for cmp_, gold in zip(comps, golds):
            text = _get_completion_text(cmp_)
            pred = extract_boxed_answer(text)
            if pred is None:
                out.append(0.0)
                continue
            out.append(1.0 if is_math_equivalent(pred, str(gold)) else 0.0)
        return out

    _fn.__name__ = "reward_correctness"
    return _fn


# ---------------------------------------------------------------- 奖励 2：格式
def make_format_reward() -> Callable:
    """结构合规：<reasoning> 块存在且 \boxed{} 存在且非空。"""

    def _fn(prompts=None, completions=None, **kwargs):
        comps = _pick_completions(prompts, completions)
        out = []
        for cmp_ in comps:
            text = _get_completion_text(cmp_)
            has_reasoning = bool(_REASONING_RE.search(text))
            boxed = extract_boxed_answer(text)
            has_box = boxed is not None and len(boxed.strip()) > 0
            out.append(1.0 if (has_reasoning and has_box) else 0.0)
        out = [float(v) for v in out]
        return out

    _fn.__name__ = "reward_format"
    return _fn


# ---------------------------------------------------------------- 奖励 3：长度（惩罚）
def make_length_reward(ref_tokens: float = 400.0, mode: str = "soft", tokenizer=None) -> Callable:
    """soft 惩罚：只有超过 ref_tokens 之后的部分才 penalty，避免"该长的地方被砍短"。

    返回负值（惩罚）。mode='none' 时返回全 0，方便做 ablation。

    面试要点：为什么不直接用 `-len`？
      因为那样梯度会诱导模型把**难题**也砍短，最终表现是难题正确率掉、易题长度降。
      soft 惩罚的梯度只在超长时才出现，等于告诉模型"别啰嗦，但别少写"。

    为什么用 tanh 而不是 min(1, excess/ref)？
      早期的实现做过硬截断版本，单元测试 `tests/test_offline.py::t_reward_length_monotone`
      暴露出它在 **2 倍 ref** 之后就饱和了 —— 也就是说，一旦模型啰嗦超过 2 倍，
      长度惩罚就**再也给不出任何梯度**，长度膨胀从此彻底失控。
      这正是社区里"加了长度惩罚照样爆长度"最常见的原因。
      换成 tanh 后：有界（∈(-1,0)，不会因超长样本炸掉训练），
      但**梯度处处非零**，多啰嗦一点就多罚一点。
    """
    from math import tanh

    def _count(text: str) -> float:
        if tokenizer is not None:
            try:
                return float(len(tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        return _char_token_proxy(text)

    def _fn(prompts=None, completions=None, **kwargs):
        comps = _pick_completions(prompts, completions)
        if mode == "none":
            return [0.0 for _ in comps]
        out = []
        for cmp_ in comps:
            text = _get_completion_text(cmp_)
            n = _count(text)
            if n <= ref_tokens:
                out.append(0.0)
            else:
                excess = (n - ref_tokens) / ref_tokens
                # 有界且处处可导
                out.append(-tanh(excess))
        return out

    _fn.__name__ = "reward_length"
    return _fn


# ---------------------------------------------------------------- 组装
def build_reward_funcs(cfg) -> List[Callable]:
    """按 configs/grpo*.yaml 的 rewards 段组装奖励函数列表。

    关键点：TRL 把多个 reward 函数的返回值**直接相加**得到单条样本的总奖励，
    并不会替你做加权。所以权重必须在这里**烘焙进每个函数的输出**，否则
    config 里的 weights 只是一串没被使用的死数字（这正是早期版本踩过的坑）。
    """
    w = cfg["rewards"]["weights"]
    lp = cfg["rewards"]["length_penalty"]

    wc = float(w["correctness"])
    wf = float(w["format"])
    wl = float(w["length"])  # 长度项是惩罚（负值），权重为正表示"惩罚强度"

    base_correct = make_correctness_reward()
    base_format = make_format_reward()
    base_len = make_length_reward(
        ref_tokens=lp.get("ref_tokens", 400.0),
        mode=lp.get("mode", "soft"),
        tokenizer=cfg.get("_tokenizer"),
    )

    def weighted(fn, weight, name, neg=False):
        def _w(prompts=None, completions=None, **kwargs):
            raw = fn(prompts=prompts, completions=completions, **kwargs)
            return [r * weight * (-1.0 if neg else 1.0) for r in raw]

        _w.__name__ = name
        return _w

    r_correct = weighted(base_correct, wc, "reward_correctness")
    r_format = weighted(base_format, wf, "reward_format")
    r_len = weighted(base_len, wl, "reward_length", neg=True)

    r_correct.__doc__ = f"weight={wc}"
    r_format.__doc__ = f"weight={wf}"
    r_len.__doc__ = f"weight=-{wl}"

    return [r_correct, r_format, r_len]
