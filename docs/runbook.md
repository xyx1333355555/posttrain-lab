# TinyThinker 执行手册

> 这份文档回答一个问题：**从现在到简历上有真数字，每天具体干什么。**  
> 不是方法论，是命令清单。每条命令后面都写了「该看到什么」和「不对怎么办」。

---

## 0. 先理解这条链路在干什么

```
模型冷启动(SFT)  →  GRPO 强化  →  评测对比  →  消融实验  →  数字回填简历
   让模型学会      让模型学会        拿到         解释         拿到面试
   "按格式答题"    "答对 + 不啰嗦"    准确率      为什么      的弹药
```

**为什么必须先 SFT 再 GRPO？**（面试官必问）  
直接上 **GRPO，模型连 `\boxed{}` 都输出不出来，奖励恒为 0，没有任何梯度信号。  
SFT 的作用是把策略推进「能拿到非零奖励」的区域，这是冷启动的唯一目的。**

**为什么用 GRPO 而不是 PPO？**（面试官必问）  
PPO 要额外维护一个和策略同规模的 critic/value model，显存直接翻倍。  
GRPO 对同一个 prompt 采样 G 条回答，用**组内奖励的均值**当 baseline，  
不需要 critic —— 这是它能在单卡上跑起来的根本原因。  
代价：每条 prompt 要采样多次，采样成本高。

---

## Day 0（今天，0 成本，纯本地）

不用任何 GPU，在笔记本上完成。

### 0.1 跑离线自检

```bash
cd posttrain-lab
python tests/test_offline.py          # 或 make test
```

**该看到**：`通过 51 / 51`。

这一步不需要 torch，只要有 numpy + pyyaml。它验证的是答案抽取、数学等价判定、  
奖励函数、配置文件一致性、以及 GRPO advantage 的数学性质。  
**这些都是错了要烧掉几小时 GPU 才能发现的东西。**

> 每次改动奖励或相关系数之后都重跑一次 —— 几十秒的事，替你省几小时 GPU。

### 0.2 同步真人结语的三件小事

| 事                 | 为什么                     | 怎么做                                          |
| ----------------- | ----------------------- | -------------------------------------------- |
| **问学院/导师要 GPU**   | 一张校内 A800 胜过所有免费平台      | 直接问："能不能给我一个账号跑大模型微调实验"                      |
| **申请魔搭 36h 免费额度** | A10 24GB，能跑 1.5B，是主实验场地 | <https://modelscope.cn/my/mynotebook> 领新用户算力 |
| **注册 Kaggle**     | T4 16GB × 30h/周，用来跑消融   | 需手机验证；左侧 Settings → Accelerator 选 GPU T4     |

**顺序建议**：先问 GPU → 不行就魔搭 → 消融丢 Kaggle。

### 0.3 把仓库推上去（Kaggle/魔搭都要用）

```bash
cd posttrain-lab
git init && git add . && git commit -m "init TinyThinker post-training lab"
git remote add origin https://github.com/xyx1333355555/posttrain-lab.git
git push -u origin main
```

**同时开两个 Secret**（免费平台会话结束会清空磁盘，没这个等于白跑）：

- HuggingFace **write token**：<https://huggingface.co/settings/tokens>  
  新建一个权限为 write 的 token，并**新建一个私有仓库**（比如 `xyx/tinythinker-ckpt`）。
- 平台里设为环境变量 `HF_TOKEN`。

> ⚠️ **这句话请记住：checkpoint 不推出会话之外，等于这几小时白烧。**

---

## Day 1：上机，只做环境，不训练

**目标**：确认 GPU、精度、注意力实现被正确识别。这步烧不了 1 小时。

```bash
git clone https://github.com/xyx1333355555/posttrain-lab.git
cd posttrain-lab
pip install -r requirements.txt -q
# 可选但推荐（vLLM 能让采样快 5-10 倍）
pip install vllm
# A10/4090 才装；T4 跳过
pip install flash-attn --no-build-isolation

python scripts/99_run_free_tier.py --probe
```

**该看到**（以魔搭 A10 为例）：

```
  平台        : modelscope
  GPU         : NVIDIA A10  24.0 GB
  自动精度    : bfloat16
  注意力实现  : flash_attention_2
  推荐模型    : Qwen/Qwen2.5-Math-1.5B-Instruct
```

**关键核对**：精度必须是 `bfloat16`。  
如果是 `float16` 而你明明在 A10 上 —— 说明 `runtime.py` 没识别到型号，**先别往下跑**，  
在 T4 上强行用 bf16 会直接产出 NaN，而且你会以为是自己数据的问题。

```bash
python scripts/00_smoke_test.py --steps 20
```

**该看到**：离线自检过关 → 环境 → 模型加载 → 数据样例 → 奖励输出 → 20 步 training loss。

`[4/5] 奖励函数` 这一段务必肉眼看一眼：

```
  correctness(good/bad) : [1.0, 0.0]
  format(good/bad)      : [1.0, 0.0]
  length(good/long)     : [0.0, -0.99...]
```

**如果 correctness 是 `[0.0, 0.0]`** → 答案抽取挂了，停下来修，别继续。

---

## Day 2：数据准备（CPU，不烧 GPU 额度）

```bash
python scripts/01_prepare_sft_data.py --config configs/sft.yaml
```

**该看到**：`data/sft/train.jsonl`，几千到几万条。

抽 3 条出来**人工看一眼**（这一步 90% 的人跳过，然后跑了几小时才发现数据全错）：

```bash
head -3 data/sft/train.jsonl
```

每条必须包含完整 reasoning + `\boxed{答案}`。

**立刻验证一遍能被抽取**：答案抽取失败率必须接近 0，否则 SFT 白做。

---

## Day 3：SFT 冷启动

```bash
python scripts/99_run_free_tier.py --stage sft --hours 3
```

**该看到**：`loss` 稳定下降，最后落在 0.3–0.8 区间。  
如果 loss 不动 → 大概率是 learning rate 太大或太小；如果在 0.01 以下 → 可能过拟合到模板了，别高兴太早。

**验收最重要的一条**（这决定了整个项目的成败）：

> 用 SFT 后的模型生成一条回答，**必须输出 `\boxed{}`**。

产出在 `outputs/sft`，这个路径会被 Day 4 的 GRPO 读取。

---

## Day 4–8：GRPO 主跑（跨会话）

**这是真正烧钱的部分，一次会话 = 10.5 小时，务必看清再启动。**

先把 `configs/grpo_a10.yaml` 里的 `model_name_or_path` 改成 `outputs/sft`。

```bash
# Session 1
python scripts/99_run_free_tier.py --stage grpo --hours 10.5 \
       --sync-repo xyx/tinythinker-ckpt
```

**该看到**（`logging_steps=5`，每 5 步一行）：

| 指标                        | 健康               | 不健康意味着               |
| ------------------------- | ---------------- | -------------------- |
| `reward`                  | 缓慢上升（0.3 → 0.6+） | 平的 = 没有信号；跳变 = LR 太大 |
| `rewards/correctness`     | 上升               | 平 = 答案抽取或等价判定有问题     |
| `rewards/format`          | 接近 1.0           | 掉下去 = **格式崩溃**，必须停   |
| `completions/mean_length` | 先涨后稳，或缓慢降        | 单调暴涨 = 长度膨胀失控        |
| `kl`                      | 小且稳定             | 爆掉 = beta 太小且 LR 太大  |
| `clip_ratio`              | 0.01–0.3         | >0.5 = 策略更新过猛        |

**跨会话续跑**（Session 2，配额刷新后）：

```bash
python scripts/99_run_free_tier.py --stage grpo --hours 10.5 \
       --resume outputs/grpo/interrupt \
       --sync-repo xyx/tinythinker-ckpt
```

> `TimeBudgetCallback` 会在预算用尽前 8 分钟**主动停下来存盘**。  
> 这是专门为「平台到 12 小时强制断连且不给任何预警」设计的 ——  
> 没有它，你大概率会崩在两次 `save_steps` 之间，一晚上全白烧。

---

## Day 9：评测拿数字

```bash
python scripts/04_eval.py --ckpt outputs/grpo/main/final --tasks gsm8k,math500
```

**必须产出三行数**（这三个数就是简历上的那组）：

```
               base    sft     grpo
GSM8K         xx.x%  → yy.y% → zz.z%
MATH-500      xx.x%  → yy.y% → zz.z%
平均生成长度  xxx tok → yyy → zzz
```

**三个模型都要测**（基座 / SFT / GRPO）。只有 GRPO 一个数 = 没有对照 = 说明不了任何事。

---

## Day 10–12：消融（丢给 Kaggle T4）

```bash
make ablate-length     # 去掉长度惩罚 → 期望长度暴涨  ← 证明你的长度项有用
make ablate-format     # 去掉格式奖励 → 期望出现格式崩溃
make ablate-scale      # scale_rewards=True → 对照组，验证 Dr.GRPO 选取理由
make ablate-kl         # beta=0.04 → 看 KL 约束的代价
```

**每条只改一个变量**，这是消融的唯一法则，也是面试官会检查的地方。

---

## Day 13–15：收尾

1. 回填 `resume/徐迎新-简历-大模型后训练方向.html` 里所有标红 `【待填】`
2. 写仓库 README：实验卡片（配置、曲线、结论）
3. 写技术博客（中文，1000–2000 字）
4. 开始投 `01-投递作战地图.md` 里的 T0 名单

---

## 故障速查表

| 现象                     | 90% 的原因                    | 处理                                                      |
| ---------------------- | -------------------------- | ------------------------------------------------------- |
| 一启动就 OOM               | vLLM 占比太高                  | `vllm_gpu_memory_utilization` 0.30 → 0.22               |
| 跑几步后 OOM               | `max_completion_length` 过大 | 512 → 384 / 320；它是显存大头                                  |
| loss 变 NaN             | T4 上用了 bf16                | 确认 `--probe` 输出是 float16；检查 `train.bf16: false`         |
| reward 恒为 0            | 答案抽取失败                     | 先跑 `python tests/test_offline.py`，再人工看模型是否漏了 `\boxed{}` |
| reward 不涨              | 学习率不对 / 题目太难               | 1e-6 → 2e-6 试；或换更小数据时先看 convergence                     |
| 生成长度暴涨                 | 长度惩罚失效                     | 确认 `mode: soft`（tanh 版）；检查 `ref_tokens` 是否设得过大          |
| 会话被中断后产物没了             | 没设 `--sync-repo`           | 无法挽回，重跑。**第一次跑就带上它**                                    |
| TRL 报 unknown argument | 版本差异                       | `03_train_grpo.py` 的反射过滤已处理；确认打了 `[compat]` 日志          |
| `pip install -r` 直接失败  | flash-attn 行格式非法           | 已修复；现在 core 依赖里没有 flash-attn                            |
| HF 下载卡死                | 网络                         | `export HF_ENDPOINT=https://hf-mirror.com`              |

---

## 「够用了」的标准

不要追求跑完美。满足下面四条就可以开始投简历了：

- [ ] GSM8K 有 **base → SFT → GRPO 三个对照数**
- [ ] 生成长度有下降，且你能解释是**哪个奖励项**起了作用
- [ ] 至少跑完 2 组消融，能说清"为什么这么选超参"
- [ ] loss / reward / length 三条曲线能当场打开给面试官看

> 最后一句：**不要等准备好了再投。** AI Lab 是滚动招聘，早一周投 = 早一周占 HC。  
> 项目可以边投边跑。
