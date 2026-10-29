# 安全对齐越狱攻防评测（jailbreak_eval）

自动化越狱评测 pipeline：量化**越狱攻击成功率 ASR** 与**过度拒答率 OverRefusal**，
用「安全性 ↑」与「能力保持 ↑」双指标共同裁决，输出「安全—能力」帕累托前沿实测曲线。

## 攻击方法
| 方法 | 说明 | 是否需梯度 |
|------|------|-----------|
| `base` | 直接请求（baseline ASR） | 否 |
| `dan` | DAN 风格角色扮演越狱 | 否 |
| `prefix` | 前缀注入 "Ignore previous instructions..." | 否 |
| `base64` | 请求 base64 编码后要求解码执行 | 否 |
| `gcg` | GCG 白盒对抗后缀（默认关，`--enable-gcg` 预留） | 是 |

## 评测集
- 有害行为：JailbreakBench 100 behaviors（联网 `--online` 从 HF 拉取；离线回退内置 12 条 demo）
- 良性集（测过度拒答）：XSTest 风格「看似有害实则安全」prompt

## 判定
- 拒绝判定：关键词法（离线稳定可用）
- 有害判定：默认「未拒绝即越狱成功」（保守上界）；`--judge-model` 可接 LLM judge 降假阳性

## 运行
```bash
# 本地冒烟（离线，0.5B，内置 demo）
python run_eval.py --model D:/models/Qwen2.5-0.5B-Instruct --max-behaviors 12 --out outputs/jb_smoke.json

# 免费 GPU（Kaggle T4 / 魔搭 A10，联网，完整 100 behaviors + LLM judge）
python run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --online --judge-model Qwen/Qwen2.5-7B-Instruct --out outputs/jb_full.json
```

## 复用建议
- 把本目录整体作为 Kaggle Notebook / 魔搭 Notebook 运行，免费 GPU 上跑完整 100 behaviors + LLM judge 即可得到可回填简历的真实数字。
- 结果 JSON 含每条 behavior 的 prompt/response/refused/jailbroken，可回溯定位「哪类攻击最易突破」。
