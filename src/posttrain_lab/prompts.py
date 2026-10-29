"""提示模板。

统一放在这里，是为了保证「训练格式 == 评测格式」。
这是新手最容易踩的坑：训练时用了带 System Prompt 的模板，评测时忘了加，分数直接掉一半。
"""

SYSTEM = (
    "You are a careful mathematical reasoner. Think step by step, then give the final answer."
)

COT_TEMPLATE = (
    "<reasoning>\n{reasoning}\n</reasoning>\n"
    "The final answer is \\boxed{{{answer}}}.\n"
)


def build_prompt(problem: str, tokenizer=None) -> str:
    """构造训练/推理用的 user prompt。

    若传入 tokenizer，则使用官方 chat template；否则用朴素拼接（兼容性更好）。
    """
    instruction = (
        f"Solve the following problem.\n\n{problem}\n\n"
        "Requirements:\n"
        "1. Put your full reasoning inside <reasoning>...</reasoning>.\n"
        "2. Put the final answer inside \\boxed{...}.\n"
    )
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{SYSTEM}\n\n{instruction}\n"


def build_target(reasoning: str, answer: str) -> str:
    """把 (推理过程, 答案) 拼成标准 completion。"""
    return COT_TEMPLATE.format(reasoning=reasoning.strip(), answer=str(answer).strip())


def strip_thinking(text: str) -> str:
    """去掉 Qwen 系列可能自带的思考标签，避免污染格式奖励。"""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
