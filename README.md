# TinyThinker — 单卡可复现的推理模型后训练流水线（SFT → GRPO）

> 目标：在一张消费级 GPU（RTX 4090 / 24GB）上跑通「**冷启动 SFT → GRPO 可验证奖励强化 → 自动评测**」的完整后训练闭环。
> 这是一份**面向实习申请**的工程型项目：重点不是刷榜，而是 **可复现、可归因、可讲解**。

---

## 为什么做这个

2025–2026 的大模型后训练岗位，核心语言是 **verifiable-reward RL**（GRPO / DAPO / Dr.GRPO）。
但在校生最常见的困境是：没有多卡集群，只有一张 24GB 的卡，跑不动 7B 的 RLHF。

本项目的答案是：**把规模降到能本人在单卡上跑一遍 RL 全过程的量级（1.5B），把工程做扎实**。
面试官更在意你是否懂这四件事，而不是你训了多大的模型：

1. reward 怎么设计、为什么这么设计
2. loss / reward / KL / 生成长度四条曲线怎么读
3. 显存是怎么算出来的
4. 消融实验排除了哪些变量

---

## 三项目产出（对应简历三条）

| 编号 | 内容 | 主要脚本 |
|---|---|---|
| **P0** | SFT → GRPO 主链路 + 三路可验证奖励 + 完整追踪 | `scripts/01` ~ `04` |
| **P1** | 后训练数据合成与配比消融（0.5B proxy → 1.5B 验证） | `scripts/05` |
| **P2** | 安全对齐与越狱攻防双指标评测 | `scripts/06` |

---

## 快速开始（本地 / 租卡）

### ⭐ 第 0 步：先跑离线自检（今天就能做，不需要 GPU）

```bash
pip install numpy pyyaml           # 只依赖这两个
python tests/test_offline.py       # 或 make test
```

51 条用例，覆盖答案抽取、数学等价判定、奖励函数、配置一致性、GRPO advantage 数学。
**这些都是错了要烧掉几小时 GPU 才能发现的 bug**，而它们全部可以在笔记本上被拦住。

它已经替你抓到过一个真实缺陷：长度惩罚原本用 `min(1, excess)` 截断，
导致模型啰嗦超过 2 倍 ref 之后**惩罚再无梯度**，长度膨胀彻底失控 ——
现已改为 `tanh(excess)`：有界但处处可导。
回归护栏就是 `t_reward_length_monotone` 与 `t_reward_length_gradient_never_vanishes` 两条用例。

### 上机之后

```bash
git clone <your-repo-url> posttrain-lab && cd posttrain-lab
pip install -r requirements.txt
# 可选：vLLM 让采样快 5-10 倍；A10/4090 再装 flash-attn
pip install vllm
pip install flash-attn --no-build-isolation

# 【国内加速】建议全程开启 HF 镜像，否则下载会很慢
export HF_ENDPOINT=https://hf-mirror.com

# Step 0 —— 先跑最小闭环（约 15 分钟，务必先跑这步；它会自动先跑离线自检）
python scripts/00_smoke_test.py --steps 20 --data-limit 256

# Step 1 —— 准备冷启动 SFT 数据
python scripts/01_prepare_sft_data.py --config configs/sft.yaml

# Step 2 —— 冷启动 SFT
python scripts/02_train_sft.py --config configs/sft.yaml

# Step 3 —— GRPO 强化
python scripts/03_train_grpo.py --config configs/grpo.yaml

# Step 4 —— 评测（真模型产出数字填进简历）
python scripts/04_eval.py --ckpt outputs/grpo/final --tasks gsm8k,math500
```

也可以直接用 `make`：`make test` / `make smoke` / `make sft` / `make grpo` / `make eval`。

> **按天排期、命令清单、故障速查** → 见 [`docs/runbook.md`](docs/runbook.md)
> **免费算力（魔搭 A10 / Kaggle T4）怎么用** → 见根目录 `05-免费算力作战指南.md`

---

## 免费算力平台（Kaggle / Colab / 魔搭）

没有预算也能跑完整套。代码已经内置环境自适应，**一条命令即可**：

```bash
python scripts/99_run_free_tier.py --probe                  # 先看分到什么卡（不耗配额）
python scripts/99_run_free_tier.py --stage grpo --hours 10.5
```

自动处理三件事：

| 问题 | 处理 |
|---|---|
| T4 / P100 **不支持 bf16** | 自动切 `fp16` + `sdpa` |
| 显存只有 16GB | 自动切 `configs/grpo_16gb.yaml`（0.5B + 短生成 + 小 vLLM 显存占比） |
| 会话到点被强制断连 | `TimeBudgetCallback` 在预算用尽前主动停止并存盘；下次 `--resume` 续跑 |

配套：`notebooks/kaggle_run.ipynb`（Kaggle 可直接上传）、`configs/grpo_a10.yaml`（魔搭 A10 24GB 主实验）。
详细平台配额对比与排班表见 `../05-免费算力作战指南.md`。

> 💡 **主线推荐**：主实验用魔搭 A10（24GB + bf16，能跑 1.5B，免费 36 小时）；
> 消融这类多组小实验丢给 Kaggle T4（每周 30 小时）。这是工业界的真实分工。

---

## 显存账怎么算（面试必问，请把这张表背下来）

以 `Qwen2.5-Math-1.5B` 全参训练、bf16、batch=4、seq=1024 为例：

| 项 | 大小 | 说明 |
|---|---|---|
| 权重（bf16） | 1.5B × 2 ≈ **3.0 GB** | 训练时权重通常用 bf16 主拷贝 |
| 梯度（bf16） | ≈ **3.0 GB** | 与权重同精度同尺寸 |
| 优化器 AdamW fp32 (m, v) | 1.5B × 8 ≈ **12.0 GB** | **最贵的一项**，用 8-bit Adam 可降到 ≈3 GB |
| AdamW 8-bit（本项目默认） | ≈ **3.0 GB** | `optim="adamw_bnb_8bit"` |
| vLLM colocate 显存池 | 0.30 × 24 ≈ **7.2 GB** | 采样与训练共用一张卡 |
| 激活值（开 gradient checkpointing） | ≈ **1.5–3 GB** | 与 batch × seq 成正比 |
| **合计（默认配置）** | **≈ 18–20 GB** | 留有余量，24GB 可跑 |

> 如果你换成 LoRA：`optim` 只优化适配器参数，优化器状态几乎为 0，能再省 6–8 GB，但要注意 vLLM 侧的 adapter 同步开销。

---

## 三路可验证奖励

```python
r = 1.0 * correctness + 0.2 * format_ok - 0.05 * length_penalty
```

| 奖励 | 目的 | 设计要点 |
|---|---|---|
| `correctness` | 答案正确与否 | 用 `math-verify` 做**数学等价判定**而不是字符串相等（`0.5` ≡ `1/2` ≡ `\frac{1}{2}`） |
| `format_ok` | 输出结构合规 | 要求 `<reasoning>…</reasoning>` 且答案出现在 `\boxed{}` 中；防止「格式崩溃」 |
| `length_penalty` | 抑制思维链膨胀 | **软惩罚**：只在超过对照组中位数长度后才生效，避免一刀切把该长的地方砍短 |

配套使用了两个社区公认技巧：
- `scale_rewards=False`（Dr. GRPO）：去掉按长度归一化带来的偏置，直接治理长度膨胀
- `loss_type="dr_grpo"`（若 TRL 版本不支持会自动回退到默认）

---

## 目录结构

```
posttrain-lab/
├── configs/          训练配置（yaml）
├── src/posttrain_lab/
│   ├── prompts.py     提示模板（CoT / 结构约束）
│   ├── rewards.py     三路可验证奖励
│   ├── data_utils.py  数据下载、清洗、去重、难度分桶
│   ├── eval_utils.py  答案抽取与 math-verify 判定
│   └── viz.py         四条曲线的产出
├── scripts/
│   ├── 00_smoke_test.py        最小闭环
│   ├── 01_prepare_sft_data.py  SFT 数据构建
│   ├── 02_train_sft.py         冷启动 SFT
│   ├── 03_train_grpo.py        GRPO 强化
│   ├── 04_eval.py              GSM8K / MATH-500 评测
│   ├── 05_data_mix_ablation.py 数据配比消融（P1）
│   └── 06_safety_eval.py       安全双指标评测（P2）
└── docs/cost_budget.md
```

---

## 复现性保证

- 每个阶段固定 `seed=42`，并将完整配置落盘到 `outputs/<stage>/config_snapshot.yaml`
- 所有指标写入 `outputs/<stage>/metrics.jsonl`，可用 `viz.py` 一键出图
- 环境版本记录在 `requirements.txt`，建议开跑前先 `pip freeze > env.txt`
