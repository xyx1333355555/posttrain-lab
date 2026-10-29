# 训练命令的一层薄封装 —— 记不住 python 命令时就用它

PY ?= python
export HF_ENDPOINT ?= https://hf-mirror.com

.PHONY: env test smoke data sft grpo eval figs ablation safety all

env:
	pip install -r requirements.txt

# ★ 不烧 GPU 的第一道关卡：本地笔记本就能跑，上机前必跑
test:
	$(PY) tests/test_offline.py

smoke: test
	$(PY) scripts/00_smoke_test.py --steps 20

# 免费平台专用（Kaggle / Colab / 魔搭）
probe:
	$(PY) scripts/99_run_free_tier.py --probe

free-smoke:
	$(PY) scripts/99_run_free_tier.py --stage grpo --smoke --hours 1.0

free-grpo:
	$(PY) scripts/99_run_free_tier.py --stage grpo --hours 10.5

free-resume:
	$(PY) scripts/99_run_free_tier.py --stage grpo --hours 10.5 --resume outputs/grpo/interrupt

data:
	$(PY) scripts/01_prepare_sft_data.py --config configs/sft.yaml

data-debug:
	$(PY) scripts/01_prepare_sft_data.py --config configs/sft.yaml --limit 2000 --out data/sft/debug.jsonl

sft:
	$(PY) scripts/02_train_sft.py --config configs/sft.yaml

sft-smoke:
	$(PY) scripts/02_train_sft.py --config configs/sft.yaml --smoke

grpo:
	$(PY) scripts/03_train_grpo.py --config configs/grpo.yaml

grpo-smoke:
	$(PY) scripts/03_train_grpo.py --config configs/grpo.yaml --smoke

# 四个消融（每条只改一个变量）
ablate-length:
	$(PY) scripts/03_train_grpo.py --config configs/grpo.yaml --ablate no_length

ablate-format:
	$(PY) scripts/03_train_grpo.py --config configs/grpo.yaml --ablate no_format

ablate-scale:
	$(PY) scripts/03_train_grpo.py --config configs/grpo.yaml --ablate no_scale

ablate-kl:
	$(PY) scripts/03_train_grpo.py --config configs/grpo.yaml --ablate high_kl

eval:
	$(PY) scripts/04_eval.py --ckpt outputs/grpo/main/final --tasks gsm8k,math500

figs:
	$(PY) -c "from posttrain_lab.viz import plot_all; plot_all('outputs/grpo/main/metrics.jsonl')"

ablation:
	$(PY) scripts/05_data_mix_ablation.py --total 8000 --dry-run

safety:
	$(PY) scripts/06_safety_eval.py --ckpt outputs/grpo/main/final

all: smoke data sft grpo eval ablation safety
