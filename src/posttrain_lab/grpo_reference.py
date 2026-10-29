"""GRPO advantage 的numpy 参考实现（不依赖 torch，笔记本上就能跑）。

为什么要有这个文件
------------------
GRPO 最容易被忽略的一点是：**advantage 怎么算，决定了这个 run 到底是不是你要的那个算法。**
而这一步在 TRL 里是藏在 trainer 内部的黑箱，跑出来 loss 往下掉你也不知道它算对了没有。

这个模块把 shields TRL `_get_per_token_logps_and_advantages` 的逻辑用 numpy 重写一遍，
并且写成可以被单元测试验证的形式。于是你有两样东西：

  1. **正确性护栏** —— `tests/test_offline.py` 验证它满足 GRPO 应有的一切性质；
     将来你在 TRL 里跑真实训练时，可以把中间 advantage 拉出来跟这个参考值对一遍。
  2. **面试武器** —— 面试官问"GRPO 和 PPO 到底差在哪"的时候，你可以直接把公式写出来，
     并且说"我把它实现了一遍，跟 TRL 的算过一致性"，这跟"我看过论文"是两种分量。

数学
----
对同一个 prompt 采样 G 条回答，奖励 r_1..r_G：

    DR-GRPO / DAPO 推荐（scale_rewards=False）：
        A_i = r_i - mean(r)                      ← 不除以标准差

    原始 GRPO（scale_rewards=True）：
        A_i = (r_i - mean(r)) / (std(r) + 1e-4)

差别不是小事：除以 std 会让**组内奖励差异很小的一组**获得被放大的 advantage，
而这些组恰恰是"题目太简单/太难、大部分回答都同分"的组——等于给噪声样本加了权重。
Dr.GRPO 那篇论文指出，这还会与"回答长度"产生耦合，是思维链无脑变长的元凶之一。

关于 std 的自由度
-----------------
torch.std 默认 correction=1（无偏估计，除以 G-1），numpy 默认 ddof=0。
TRL 用的是 torch.std()，所以这里默认 **ddof=1**。这个细节面试可以提一句，
说明你是真的读过实现而不是只看公式。
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def group_advantages(
    rewards: Sequence[float],
    num_generations: int = 8,
    scale_rewards: bool = False,
    ddof: int = 1,
    eps: float = 1e-4,
) -> np.ndarray:
    """按 prompt 分组算 GRPO advantage。

    rewards 的长度必须是 num_generations 的整数倍（每个 prompt 一组 G 个采样）。
    """
    r = np.asarray(list(rewards), dtype=np.float64)
    if r.size % num_generations != 0:
        raise ValueError(
            f"rewards 长度 {r.size} 不是 num_generations={num_generations} 的整数倍；"
            "这通常意味着 batch 里混进了不完整的组"
        )
    G = num_generations
    grouped = r.reshape(-1, G)

    mean = grouped.mean(axis=1, keepdims=True)
    adv = (grouped - mean).reshape(-1)

    if scale_rewards:
        # ddof=1 与 torch.std 默认行为一致；G=1 时 std 为 0，靠 eps 兜底
        std = grouped.std(axis=1, ddof=ddof, keepdims=True) if G > 1 else np.zeros_like(mean)
        adv = (adv.reshape(-1, G) / (std + eps)).reshape(-1)
    return adv


def token_level_advantages(
    completion_lengths: Sequence[int],
    rewards: Sequence[float],
    num_generations: int = 8,
    scale_rewards: bool = False,
) -> List[np.ndarray]:
    """把 sequence-level advantage 摊到每个 token 上（loss 里实际用的是这个）。

    GRPO 没有 critic / GAE：同一条回答内部**所有 token 共享同一个 advantage**，
    这也是它比 PPO 省一大块显存的原因（省掉了 value model）。
    """
    seq_adv = group_advantages(rewards, num_generations, scale_rewards)
    out = []
    for adv, L in zip(seq_adv, completion_lengths):
        out.append(np.full(int(L), float(adv), dtype=np.float64))
    return out


def dapo_token_normalizer(total_tokens: int) -> float:
    """DAPO 的 token-level loss 归一化：按 batch 内总 token 数而非样本数求平均。

    原始 GRPO 对每个样本内 token 求平均再对样本求平均，会让长回答里的每个 token
    权重被稀释。DAPO 改成对整个 batch 的 token 求一次全局平均，等价于
    「每个 token 一视同仁」，这也是抑制长度膨胀的手段之一。
    """
    return float(max(int(total_tokens), 1))


def worked_example() -> dict:
    """一个能用手算验证的最小例子 —— 面试时可以直接写在白板上。"""
    rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # G=8，四对四
    g8 = group_advantages(rewards, 8, scale_rewards=False)
    g8_scaled = group_advantages(rewards, 8, scale_rewards=True)
    allsame = group_advantages([1.0] * 8, 8, scale_rewards=True)
    return {
        "rewards": rewards,
        "dr_grpo": g8.tolist(),
        "grpo_scaled": g8_scaled.tolist(),
        "all_same_group_adv_should_be_zero": allsame.tolist(),
    }
