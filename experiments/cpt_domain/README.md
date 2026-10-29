# E1 · 垂直领域增量预训练（CPT）与灾难性遗忘

对应 JD：**「参与垂直领域大模型的全流程训练工作，包括增量预训练（CPT）、SFT、强化学习对齐（RLHF/DPO/GRPO）」** 中的第一环 —— CPT。

单卡 RTX 4060 8GB 可跑，全程 0 元算力。

---

## 1. 实验设计

**垂直领域选择：密码学与 AI 安全**（与本人导师研究方向、Argus 提示词窃取防御课题同源）

| 项 | 内容 | 规模 |
|---|---|---|
| 领域语料 | arXiv `cs.CR`（Cryptography and Security）论文摘要 | 1,900 篇 / **56.9 万 tokens** |
| 领域验证集 | 同分布留出文档（文档级切分，避免同文泄漏） | 100 篇 / 3.05 万 tokens |
| 通用回放语料 | WikiText-2（英文维基正文） | 1,425 篇 / 25.1 万 tokens |
| 通用验证集 | WikiText-2 留出 | 75 篇 / 1.12 万 tokens |
| 基座模型 | Qwen2.5-0.5B-Instruct | 494 M 参数（全参训练） |
| 训练配置 | seq 512 / bs 1 / 梯度累积 8 = **4096 tokens/step**；bf16；梯度检查点；cosine | — |

**三组对照（单变量）**

| 组 | 步数 | lr | 回放比例 | 目的 |
|---|---|---|---|---|
| base | — | — | — | 基线锚点 |
| pure_domain（温和） | 200 | 2e-5 | 0 | 常规 CPT 是否伤通用能力 |
| pure_strong（加强） | 300 | 5e-5 | 0 | 加大强度诱发灾难性遗忘 |
| replay10_strong（回放） | 300 | 5e-5 | **10%（按 token 计）** | 通用回放能否救回通用能力 |

**三个指标同时看**（只报一个好数字不算结论）
- `ppl_domain`：领域留出集困惑度 → 领域适配是否成功
- `ppl_general`：通用留出集困惑度 → 通用语言能力是否受损
- `gsm8k_acc`：GSM8K 前 100 题（贪心解码，`\boxed{}` 抽取 + 数学等价判定）→ 下游任务能力是否回归

---

## 2. 结果（真实数字，全部来自 `results/*.json`）

| 组 | 领域 PPL | Δ vs base | 通用 PPL | Δ vs base | GSM8K (n=100) | 格式率 |
|---|---|---|---|---|---|---|
| base（Qwen2.5-0.5B-Instruct） | 46.65 | — | **24.87** | — | 25.0% | 56% |
| CPT 温和（200步 / 2e-5 / 无回放） | 31.02 | **−33.5%** | 25.17 | +1.2% | 29.0% | 60% |
| CPT 加强（300步 / 5e-5 / 无回放） | **29.08** | **−37.7%** | 30.77 | **+23.7%** | 27.0% | 53% |
| CPT 加强 + 10% 回放 | 29.25 | −37.3% | **24.59** | **−1.1%** | 26.0% | 46% |

**训练开销（4060 8GB，0 元）**

| 组 | 可训练参数 | 训练 tokens | 步耗时 | 峰值显存 | 总耗时 |
|---|---|---|---|---|---|
| pure_domain | 494 M（全参） | 57.1 万 | 1.68 s/step | **4.65 GB** | 5.6 min |
| replay10_strong | 494 M（全参） | 63.4 万（领域 56.9 万 + 回放 6.3 万） | 1.60 s/step | **4.65 GB** | 8.0 min |

---

## 3. 结论（写进简历的三句话）

1. **领域适配有效且显著**：仅 200 步 / 57 万 tokens 的轻量 CPT，就把领域留出困惑度从 46.65 压到 31.02（**−33.5%**）；加强到 300 步 / lr 5e-5 后进一步降到 29.08（**−37.7%**）。

2. **灾难性遗忘真实存在，但「困惑度」比「任务准确率」更敏感**：加强版 CPT 使通用困惑度上涨 **+23.7%**（24.87→30.77），而 GSM8K 准确率只从 25.0% 变到 27.0%（在 n=100 的采样噪声 ±4.4pt 内）。**只看下游任务指标会严重低估遗忘程度** —— 这是评测设计上的一个实际教训。

3. **10% 通用回放几乎零成本地消除遗忘**：混入 10%（按 token）通用语料后，通用困惑度回到 **24.59（比基线还低 1.1%）**，而领域困惑度仅从 29.08 微升到 29.25（**代价 0.6%**）。即「用 10% 的数据预算买回全部通用能力」。

4. **额外发现（引出下一环 SFT）**：CPT 会削弱指令遵循格式，`\boxed{}` 格式合规率从 56% 掉到 46% —— 说明 CPT 之后**必须接 SFT 才能恢复对话/输出格式**，这正是全流程里 CPT→SFT 的动机，而不是照本宣科。

---

## 4. 复现命令

```bash
# 1) 构建语料（需外网；arXiv API + HF 镜像）
python experiments/cpt_domain/prepare_corpus.py --wiki-en-gen 1500 --general-source wikitext

# 2) 基线评测
python experiments/cpt_domain/eval_cpt.py \
  --model D:/models/Qwen2.5-0.5B-Instruct --tag base_orig --n-gsm8k 100

# 3) 纯领域 CPT（温和 / 加强）
python experiments/cpt_domain/train_cpt.py --tag pure_domain  --max-steps 200 --lr 2e-5 --replay-ratio 0
python experiments/cpt_domain/train_cpt.py --tag pure_strong  --max-steps 300 --lr 5e-5 --replay-ratio 0

# 4) 加强 + 10% 通用回放
python experiments/cpt_domain/train_cpt.py --tag replay10_strong --max-steps 300 --lr 5e-5 --replay-ratio 0.1

# 5) 各组评测
python experiments/cpt_domain/eval_cpt.py --model outputs/cpt/pure_domain/final    --tag pure_domain
python experiments/cpt_domain/eval_cpt.py --model outputs/cpt/pure_strong/final    --tag pure_strong
python experiments/cpt_domain/eval_cpt.py --model <replay10_strong>/final          --tag replay10_strong
```

## 5. 复现踩坑记录（Windows + 8GB）

- **代理坑**：环境变量 `http(s)_proxy` 指向已失效的本地代理会返回 502，抓语料需 `env -u HTTP_PROXY -u HTTPS_PROXY ...` 直连。
- **arXiv 分页**：`max_results=2000` 会被 403，改为 100/次 + 1s 间隔。
- **语料源选择**：中文维基按标题抓取会触发 429、按 allpages 抓命中率极低（大量短条目），最终通用语料改用 WikiText-2（HF 国内镜像）。
- **显存**：0.5B 全参 + AdamW + 梯度检查点峰值仅 4.65 GB，8GB 卡有余量；真正的显存瓶颈是 GRPO 的 4 路 rollout（7.7 GB），不是 CPT。
- **保存**：yaml/json 里别塞 tokenizer 对象；trainer 结束时的 final `on_log` 可能不含 `loss` 字段，日志回调需判空，否则训练跑完却在最后一步崩、模型没存。
- **磁盘**：全参 checkpoint 含 optimizer 约 2.8 GB/个，输出目录别放 C 盘。
