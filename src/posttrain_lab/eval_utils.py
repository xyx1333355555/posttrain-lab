"""答案抽取与数学等价判定。

关键点：不能用字符串相等来判断数学答案。
  "0.5" / "1/2" / "\\frac{1}{2}" 是同一个答案。
优先使用 math-verify（工业界标准），失败时回退到手写归一化。
"""
from __future__ import annotations

import re
from typing import Optional

# 匹配最后一个 \boxed{...}（模型常常在中间也写 \boxed，取最后一个）
_BOXED_ALL = re.compile(r"\\boxed\{")


def extract_boxed_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    matches = list(_BOXED_ALL.finditer(text))
    if not matches:
        # 兼容 <answer>...</answer> 形式
        m = re.search(r"<answer>(.*?)</answer>", text, re.S)
        return m.group(1).strip() if m else None

    start = matches[-1].end()  # '{' 之后
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    inner = text[start : i - 1]
    # 去掉 \text{...} 外壳
    inner = re.sub(r"^\s*\\text\{(.*)\}\s*$", r"\1", inner, flags=re.S)
    return _normalize(inner)


def _normalize(s: str) -> str:
    s = s.strip()
    s = s.rstrip(".").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("$", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\s+", "", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.rstrip("\\").strip()
    return s


def _try_float(s: str) -> Optional[float]:
    """尽力把字符串转成 float。这里是「可验证奖励」的正确性地基：

    判错一次 = 给模型一条错误的策略梯度。所以下面这些形式都必须覆盖到：
        1234        -0.5         1,234.5
        1/2         \\frac{1}{2}  \\dfrac{1}{2}
        12%         2^3 (不做，交给字符串归一化)
    """
    if s is None:
        return None
    try:
        cleaned = str(s).strip().replace(",", "").replace("$", "").replace("\\!", "")

        # 百分号：12% -> 0.12（注意 gold 与 pred 要同一处理，单侧出现时不影响结论）
        if cleaned.endswith("%"):
            base = _to_number(cleaned[:-1])
            return None if base is None else base / 100.0

        # \frac{a}{b} / \dfrac{a}{b} / \tfrac{a}{b}
        m = re.fullmatch(r"-?\\?(?:frac|dfrac|tfrac)\{(-?[\d.]+)\}\{(-?[\d.]+)\}", cleaned)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return None if b == 0 else a / b

        # 纯分数 a/b（MATH-500 的 gold 经常长这样，漏掉会把 1/2 判成不等于 0.5）
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", cleaned)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return None if b == 0 else a / b

        return _to_number(cleaned)
    except Exception:
        return None


def _to_number(cleaned: str) -> Optional[float]:
    neg = 1.0
    if cleaned.startswith("-"):
        neg, cleaned = -1.0, cleaned[1:]
    if not cleaned or not re.fullmatch(r"[\d.]+", cleaned):
        return None
    try:
        return neg * float(cleaned)
    except Exception:
        return None


def is_math_equivalent(pred: Optional[str], gold: str) -> bool:
    """三重判定：math-verify → 数值比较 → 字符串归一化比较。"""
    if pred is None:
        return False
    gold = _normalize(str(gold))

    # 1) math-verify（最权威，但安装失败时不可用）
    try:
        from math_verify import parse, verify  # type: ignore

        try:
            g = parse(f"${gold}$")
            p = parse(f"${pred}$")
            if verify(g, p):
                return True
        except Exception:
            pass
    except Exception:
        pass

    # 2) 数值等价
    fp, fg = _try_float(pred), _try_float(gold)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-6

    # 3) 归一化字符串
    return pred == gold


# ---------------------------------------------------------------- GSM8K
def parse_gsm8k_answer(sol: str) -> Optional[str]:
    """gsm8k 的最终答案形如 '#### 18'。"""
    if sol is None:
        return None
    m = re.search(r"####\s*(.+?)\s*$", sol, re.S)
    ans = _normalize(m.group(1)) if m else None
    return ans
