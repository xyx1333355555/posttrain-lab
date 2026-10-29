#!/usr/bin/env python
"""离线自检：不需要 GPU、不需要 torch，笔记本上 python 直接跑。

为什么把这一步放在最前面
--------------------------
后训练跑一次要烧掉几小时 GPU。而实践中最贵的错误，往往不需要 GPU 就能发现：

  * 答案抽取漏掉一种写法 -> 奖励全 0 -> loss 平着趟了几小时你才发现
  * 组内 advantage 符号搞反 -> 模型在学"答错" -> 曲线看起来还在动
  * batch_size 不能被 num_generations 整除 -> 训练脚本启动 3 秒才崩，但你已经排了 20 分钟队

这些都在这份测试里被钉死了。**上机前的第一件事就是 `make test`。**

用法：
    python tests/test_offline.py           # 全量
    python tests/test_offline.py -v        # 打印每条用例
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import yaml  # noqa: E402

from posttrain_lab import grpo_reference as gr  # noqa: E402
from posttrain_lab.eval_utils import (  # noqa: E402
    extract_boxed_answer,
    is_math_equivalent,
    parse_gsm8k_answer,
)
from posttrain_lab.prompts import build_prompt, build_target, strip_thinking  # noqa: E402
from posttrain_lab.rewards import (  # noqa: E402
    build_reward_funcs,
    make_correctness_reward,
    make_format_reward,
    make_length_reward,
)

PASS, FAIL = [], []
VERBOSE = "-v" in sys.argv


def case(name: str, fn):
    try:
        fn()
        PASS.append(name)
        if VERBOSE:
            print(f"  [ok] {name}")
    except AssertionError as e:
        FAIL.append((name, str(e) or "assertion failed"))
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")


# ============================================================ 1. 答案抽取
def t_boxed_simple():
    assert extract_boxed_answer("The answer is \\boxed{42}.") == "42"


def t_boxed_nested_braces():
    assert extract_boxed_answer("x = \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def t_boxed_takes_last():
    s = "first \\boxed{1} then finally \\boxed{7}"
    assert extract_boxed_answer(s) == "7"


def t_boxed_unterminated_returns_none():
    assert extract_boxed_answer("the answer is \\boxed{7") is None


def t_boxed_text_wrapper():
    assert extract_boxed_answer("\\boxed{\\text{yes}}") == "yes"


def t_answer_tag_fallback():
    assert extract_boxed_answer("<answer>13</answer>") == "13"


def t_boxed_none_input():
    assert extract_boxed_answer(None) is None


def t_boxed_strips_trailing_period_and_space():
    assert extract_boxed_answer("\\boxed{ 18. }") == "18"


# ============================================================ 2. 数学等价
def t_equiv_decimal_fraction():
    assert is_math_equivalent("0.5", "1/2")
    assert is_math_equivalent("1/2", "0.5")


def t_equiv_latex_frac():
    assert is_math_equivalent("\\frac{1}{2}", "0.5")
    assert is_math_equivalent("\\dfrac{3}{4}", "0.75")


def t_equiv_thousands_separator():
    assert is_math_equivalent("1,234", "1234")


def t_equiv_percent():
    assert is_math_equivalent("12%", "0.12")


def t_equiv_negative():
    assert is_math_equivalent("-3", "-3.0")


def t_equiv_rejects_wrong():
    assert not is_math_equivalent("5", "6")
    assert not is_math_equivalent("0.33", "1/3")


def t_equiv_none_is_false():
    assert not is_math_equivalent(None, "5")


def t_equiv_string_form():
    assert is_math_equivalent("yes", "yes")


# ============================================================ 3. GSM8K
def t_gsm8k_hash():
    assert parse_gsm8k_answer("blah blah\n#### 18") == "18"


def t_gsm8k_dollar():
    sol = "He has 3 apples.\n#### $18"
    assert parse_gsm8k_answer(sol) == "$18".replace("$", "")


def t_gsm8k_missing():
    assert parse_gsm8k_answer("no final marker") is None


# ============================================================ 4. 提示与模板
def t_train_eval_format_consistent():
    """训练用的 target 必须能被评测的抽取器抽出来，否则奖励恒为 0。"""
    target = build_target("Step one. Step two.", "42")
    assert extract_boxed_answer(target) == "42"


def t_prompt_contains_requirements():
    p = build_prompt("What is 2+3?")
    assert "boxed" in p and "<reasoning>" in p


def t_prompt_no_tokenizer_is_str():
    assert isinstance(build_prompt("q"), str)


def t_strip_thinking():
    assert strip_thinking("<think>hmm</think>real answer") == "real answer"


# ============================================================ 5. 奖励函数
GOOD = "<reasoning>2+3=5</reasoning> The final answer is \\boxed{5}."
BAD_TEXT = "I guess maybe the answer is 5?"
COLLAPSED = "the answer is 5"          # 格式崩溃：没有 reasoning 也没有 boxed
LONG = "<reasoning>" + "y" * 4000 + "</reasoning> The final answer is \\boxed{5}."


def t_reward_correctness_basic():
    rc = make_correctness_reward()
    got = rc([GOOD, BAD_TEXT], answer=["5", "5"])
    assert got == [1.0, 0.0], got


def t_reward_correctness_chat_messages():
    """TRL 对话模式下 completion 是 messages list，必须兼容。"""
    rc = make_correctness_reward()
    got = rc([[{"role": "assistant", "content": GOOD}]], answer=["5"])
    assert got == [1.0], got


def t_reward_correctness_frac_gold():
    rc = make_correctness_reward()
    got = rc(["... so it is \\boxed{0.5}"], answer=["1/2"])
    assert got == [1.0], got


def t_reward_format_detects_collapse():
    rf = make_format_reward()
    got = rf([GOOD, COLLAPSED], answer=["5", "5"])
    assert got == [1.0, 0.0], got


def t_reward_length_below_ref_is_zero():
    rl = make_length_reward(ref_tokens=400, mode="soft")
    got = rl([GOOD], answer=["5"])
    assert abs(got[0] - 0.0) < 1e-9, got


def t_reward_length_penalty_bounded_but_nonzero():
    """长度惩罚必须有界，但**不能饱和**：超长区域仍要有梯度。

    这条用例曾经挂过一次 —— 老实现在 2 倍 ref 之后 min(1,·) 饱和成常数，
    导致模型啰嗦过头时长度惩罚彻底失效。改动前后这条用例就是回归护栏。
    """
    rl = make_length_reward(ref_tokens=400, mode="soft")
    got = rl([LONG], answer=["5"])
    assert -1.0 < got[0] < 0, got          # 有界
    # 严格落在开区间内：tanh 不会取到 -1
    assert got[0] > -1.0, got


def t_reward_length_monotone():
    rl = make_length_reward(ref_tokens=100, mode="soft")
    short = "a" * 400      # ~100 tok -> 无惩罚
    mid = "a" * 800        # ~200 tok
    longer = "a" * 1600    # ~400 tok
    longest = "a" * 3200   # ~800 tok
    got = rl([short, mid, longer, longest], answer=["5"] * 4)
    assert got[0] == 0.0
    assert got[1] < 0, got
    assert got[2] < got[1], got            # 越啰嗦罚得越狠
    assert got[3] < got[2], got            # ★ 不可以在中段就饱和


def t_reward_length_gradient_never_vanishes():
    """关键在于**实用区间**（1.5x–4x ref）内梯度不能消失。

    老实现用 min(1, excess) 时，2x ref 之后差值恒为 0；tanh 版本在 4x ref
    仍有约 1e-2 量级的差异，足以产生有效梯度。
    （注：10x ref 以上 tanh 数值上趋近饱和，但那已远超实际生成长度。）
    """
    rl = make_length_reward(ref_tokens=400, mode="soft")
    a = rl(["a" * 2400], answer=["5"])[0]   # ~600 tok = 1.5x
    b = rl(["a" * 3200], answer=["5"])[0]   # ~800 tok = 2.0x
    c = rl(["a" * 6400], answer=["5"])[0]   # ~1600 tok = 4.0x
    assert b < a - 1e-3, (a, b)
    assert c < b - 1e-3, (b, c)


def t_reward_length_off_mode():
    rl = make_length_reward(ref_tokens=400, mode="none")
    assert rl([LONG], answer=["5"]) == [0.0]


def t_reward_length_with_tokenizer_proxy():
    """给一个假 tokenizer，验证会走真实 token 计数分支而不是字符代理。"""

    class FakeTok:
        def encode(self, s, add_special_tokens=False):
            return list(range(len(s.split())))

    rl = make_length_reward(ref_tokens=2, mode="soft", tokenizer=FakeTok())
    got = rl(["a b c d e f g h"], answer=["5"])   # 8 tokens >> 2
    assert got[0] < 0, got


def t_build_reward_funcs_from_config():
    cfg = yaml.safe_load(open(_cfg("configs/grpo.yaml"), "r", encoding="utf-8"))
    fns = build_reward_funcs(cfg)
    assert len(fns) == 3
    assert all(callable(f) for f in fns)


def t_total_reward_is_finite():
    cfg = yaml.safe_load(open(_cfg("configs/grpo.yaml"), "r", encoding="utf-8"))
    fns = build_reward_funcs(cfg)
    for f in fns:
        out = f([GOOD, COLLAPSED, LONG], answer=["5", "5", "5"])
        assert len(out) == 3
        assert all(isinstance(v, float) for v in out), f.__name__


# ============================================================ 6. 配置体检
def _cfg(rel: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)


def _grpo_configs():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.startswith("grpo")]


def t_all_configs_parse():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    files = [f for f in os.listdir(d) if f.endswith(".yaml")]
    assert len(files) >= 4, files
    for f in files:
        cfg = yaml.safe_load(open(os.path.join(d, f), "r", encoding="utf-8"))
        assert isinstance(cfg, dict), f


def t_grpo_batch_divisible():
    """TRL 硬约束：batch 必须能被 num_generations 整除，否则启动即崩。"""
    for path in _grpo_configs():
        cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
        G = cfg["grpo"]["num_generations"]
        b = cfg["train"]["per_device_train_batch_size"]
        ga = cfg["train"]["gradient_accumulation_steps"]
        assert b % G == 0, f"{os.path.basename(path)}: {b} 不能被 {G} 整除"
        spg = cfg["grpo"].get("steps_per_generation", ga)
        assert ga % spg == 0, f"{os.path.basename(path)}: steps_per_generation 不整除 grad_accum"


def t_grpo_generations_reasonable():
    for path in _grpo_configs():
        cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
        assert cfg["grpo"]["num_generations"] >= 4, "G 太小，advantage 方差会爆炸"


def t_t4_config_uses_fp16():
    cfg = yaml.safe_load(open(_cfg("configs/grpo_16gb.yaml"), "r", encoding="utf-8"))
    assert cfg["torch_dtype"] == "float16", "T4 不支持 bf16，必须是 fp16"
    assert cfg["train"]["bf16"] is False
    assert cfg["attn_implementation"] == "sdpa"


def t_a10_config_uses_bf16():
    cfg = yaml.safe_load(open(_cfg("configs/grpo_a10.yaml"), "r", encoding="utf-8"))
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["train"]["bf16"] is True


def t_dtype_and_bf16_flag_consistent():
    """torch_dtype 与 train.bf16 必须自洽。

    在 T4 上设 float16 却开着 bf16=true → 直接 NaN，而且你会误判成数据问题。
    """
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    for f in os.listdir(d):
        if not f.endswith(".yaml"):
            continue
        cfg = yaml.safe_load(open(os.path.join(d, f), "r", encoding="utf-8"))
        dt = cfg.get("torch_dtype")
        train = cfg.get("train") or cfg.get("training") or {}
        if "bf16" not in train:
            continue
        if dt == "float16":
            assert train["bf16"] is False, f"{f}: fp16 却开了 bf16"
        if dt == "bfloat16":
            assert train["bf16"] is True, f"{f}: bf16 却关了 train.bf16"


def t_save_steps_set_for_free_tier():
    """免费平台会被强制断连，save_steps 太大等于白跑。"""
    for path in _grpo_configs():
        cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
        assert cfg["train"]["save_steps"] <= 100, f"{os.path.basename(path)}: save_steps 过大"


# ============================================================ 7. GRPO 数学
def t_adv_all_same_is_zero():
    adv = gr.group_advantages([1.0] * 8, 8, scale_rewards=True)
    assert abs(adv).max() < 1e-12, adv


def t_adv_all_zero_is_zero():
    adv = gr.group_advantages([0.0] * 8, 8, scale_rewards=True)
    assert abs(adv).max() < 1e-12, adv


def t_adv_zero_mean():
    adv = gr.group_advantages([1, 0, 1, 0, 1, 0, 1, 0], 8, scale_rewards=False)
    assert abs(float(adv.mean())) < 1e-12, adv


def t_adv_monotone_with_reward():
    r = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    adv = gr.group_advantages(r, 8, scale_rewards=False)
    order = sorted(range(8), key=lambda i: adv[i])
    assert [r[i] for i in order] == sorted(r), (r, adv)


def t_adv_scale_changes_magnitude_not_sign():
    r = [1, 0, 1, 0, 1, 0, 1, 1]
    a = gr.group_advantages(r, 8, scale_rewards=False)
    b = gr.group_advantages(r, 8, scale_rewards=True)
    assert abs(float(a.mean())) < 1e-9 and abs(float(b.mean())) < 1e-2
    assert (a >= 0).tolist() == (b >= 0).tolist(), (a, b)


def t_adv_ddof1_matches_torch_std_semantics():
    """手算校验：G=4, rewards=[1,0,0,1]，mean=0.5，无偏 std=sqrt((0.25*4)/3)=0.5774"""
    r = [1.0, 0.0, 0.0, 1.0]
    a = gr.group_advantages(r, 4, scale_rewards=True, ddof=1, eps=0.0)
    expected = 0.5 / (0.5773502691896257)
    assert abs(abs(a[0]) - expected) < 1e-6, (a, expected)


def t_adv_bad_group_size_raises():
    try:
        gr.group_advantages([1.0, 0.0, 1.0], 8)
    except ValueError:
        return
    raise AssertionError("不完整的组应当抛错，否则会静默算错")


def t_token_level_advantages_shape():
    out = gr.token_level_advantages([5, 3], [1.0, 0.0], 2, scale_rewards=False)
    assert len(out) == 2 and out[0].shape == (5,) and out[1].shape == (3,)
    assert out[0][0] == out[0][4], "同一条回答内所有 token 共享 advantage"


def t_worked_example_runs():
    ex = gr.worked_example()
    assert "dr_grpo" in ex and len(ex["dr_grpo"]) == 8
    assert abs(ex["all_same_group_adv_should_be_zero"][0]) < 1e-12


def main() -> int:
    # 分组规则：给每组一组前缀。[("", None)] 兜底，保证新增用例不会被静默漏掉
    GROUPS = [
        ("答案抽取", ("t_boxed", "t_answer")),
        ("数学等价", ("t_equiv",)),
        ("GSM8K", ("t_gsm8k",)),
        ("提示模板", ("t_prompt", "t_train_eval", "t_strip")),
        ("奖励函数", ("t_reward", "t_build", "t_total")),
        ("配置体检", ("t_all_configs", "t_grpo_", "t_t4", "t_a10", "t_save", "t_dtype")),
        ("GRPO 数学", ("t_adv", "t_token", "t_worked")),
    ]
    all_tests = {k: v for k, v in globals().items() if k.startswith("t_") and callable(v)}
    groups, assigned = [], set()
    for gname, prefixes in GROUPS:
        fns = [v for k, v in sorted(all_tests.items()) if k.startswith(prefixes)]
        groups.append((gname, fns))
        assigned |= {k for k, v in all_tests.items() if k.startswith(prefixes)}

    missing = sorted(set(all_tests) - assigned)
    if missing:
        # 新写的用例忘了归类 —— 让它炸出来，而不是悄悄变成"没跑"
        raise SystemExit(f"[fatal] 以下用例未被任何分组收录，会变成永远不执行的死代码: {missing}")

    total = 0
    for gname, fns in groups:
        print(f"\n── {gname} ({len(fns)}) ──")
        for fn in fns:
            case(fn.__name__, fn)
            total += 1

    print("\n" + "=" * 60)
    print(f"  通过 {len(PASS)} / {total}")
    if FAIL:
        print(f"  失败 {len(FAIL)}：")
        for n, e in FAIL:
            print(f"    - {n}: {e}")
        print("=" * 60)
        return 1
    print("  ✔ 离线自检全部通过 —— 可以上机烧 GPU 了")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
