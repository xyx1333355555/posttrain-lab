# 本地 8GB 跑通指引（RTX 4060 / 3050 等笔记本显卡）

> 这台机器自带一张 **RTX 4060 8GB**。不用消耗你的 ModelScope / Kaggle 免费额度，
> 直接在这里把 **P0 TinyThinker 的 0.5B 验证版** 跑出真实数字，作为简历「在研项目」的硬证据。
> 1.5B 主实验仍放到 **魔搭 A10（24GB）** 上跑（见 `05-免费算力作战指南.md`）。

## 0. 解释器

本机 `python` 是 3.13，但 ML 栈装在 **Python 3.11**，统一用这个：

```bat
set PY311=C:\Users\17856\AppData\Local\Programs\Python\Python311\python.exe
```

之后所有命令都用 `%PY311%` 代替 `python`。

## 1. 环境探测（确认 GPU 到账）

```bat
%PY311% -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出类似 `2.5.1+cu121 True NVIDIA GeForce RTX 4060 Laptop GPU`。

## 2. 离线自检（不烧 GPU，2 秒）

```bat
%PY311% tests/test_offline.py
```

期望 `通过 51 / 51`。

## 3. 最小闭环 smoke test（约 5–10 分钟，下载 0.5B 模型）

```bat
cd 上海实习冲刺\posttrain-lab
set HF_ENDPOINT=https://hf-mirror.com
%PY311% scripts\00_smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 20
```

会依次验证：环境 / 模型加载 / 数据 pipeline / 奖励函数 / 一次 forward+backward。
最后看到 `✔ smoke test 通过` 再往下。

## 4. 正式 GRPO（0.5B，本地 8GB 配置，自动关 vLLM）

```bat
set HF_ENDPOINT=https://hf-mirror.com
%PY311% scripts\99_run_free_tier.py --stage grpo --config configs/grpo_8gb.yaml --no-preflight
```

- 自动走 `grpo_8gb.yaml`：0.5B + LoRA(r16) + fp16 + sdpa + 关 vLLM
- 训练日志写进 `outputs/grpo_8gb/main/metrics.jsonl`
- 跑完自动出 `figs/` 曲线和 `config_snapshot.yaml`（可复现凭证）

> 本地机器不会被强制断连，所以不必设 `--hours`。想试中断续跑可加
> `--hours 2` 触发 TimeBudgetCallback，下次 `--resume outputs/grpo_8gb/main/interrupt`。

## 5. 评测（拿简历要写的数字）

```bat
%PY311% scripts\04_eval.py --ckpt outputs/grpo_8gb/main/final --tasks gsm8k --limit 300 --out outputs/eval
```

## 6. 把真实数字回填简历

GRPO 跑完后，打开 `outputs/grpo_8gb/main/metrics.jsonl`，取：
- `reward` 首步 vs 末步（应明显上升）
- `reward_format`（格式合规率，应 → 1.0）
- `kl`（应平稳，不爆炸）
- 评测 `gsm8k` 准确率

回填到 `resume/徐迎新-简历-大模型后训练方向.html` 的「TinyThinker」条目（替换所有标红【】）。

## 失败速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | batch/序列过长 | 调小 `grpo_8gb.yaml` 的 `per_device_train_batch_size` 或 `max_completion_length` |
| `flash_attention_2` 报错 | Windows 不支持 | 配置已用 `sdpa`，若手动改过请改回 |
| 下载卡住 | 没走镜像 | 确认 `set HF_ENDPOINT=https://hf-mirror.com` |
| `reward` 恒为 0 | 答案抽取失败 | 看 `tests/test_offline.py` 的 boxed 用例；确认 gsm8k `####` 解析 |
| TRL 版本报错 `vllm_*` 字段 | 版本差异 | 已用反射过滤 + `use_vllm:false`，理论上不会 |
