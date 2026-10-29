"""免费算力平台必备的两个回调。

**TimeBudgetCallback 是本项目能在免费平台上活下去的关键。**

为什么必须有它：
  - Kaggle 单次会话最长 12 小时，到点**强制断连**；Colab 也有 12 小时上限
  - 平台不给任何"即将超时"的提前预警，断了就是断了
  - 如果训练到 11.5 小时还没触发过 save_steps，你这 11.5 小时就白烧了

正确做法：开跑时告诉它一个**小于会话上限的预算**（比如 11 小时），
它会在预算用尽的那一刻**主动、优雅地**停下来并保存 checkpoint，然后你可以下次会话 `--resume` 继续。
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from transformers import TrainerCallback


class TimeBudgetCallback(TrainerCallback):
    """按墙上时间预算停止训练，并在停止前保存 checkpoint。"""

    def __init__(self, max_minutes: float, save_dir: str, safety_margin_min: float = 8.0):
        self.max_seconds = max(minutes_to_sec(max_minutes), 60)
        self.save_dir = save_dir
        self.safety = minutes_to_sec(safety_margin_min)
        self.start = None
        self.stopped_by_budget = False

    def on_train_begin(self, args, control, state=None, **kwargs):
        self.start = time.time()
        print(f"[TimeBudget] 预算 {self.max_seconds/3600:.2f} h；"
              f"将在剩余 {self.safety/60:.0f} 分钟时强制停止并保存")

    def _maybe_stop(self, args, control, model=None, tokenizer=None, **kwargs):
        if self.start is None:
            return control
        elapsed = time.time() - self.start
        remaining = self.max_seconds - elapsed
        if remaining <= self.safety:
            print(f"\n[TimeBudget] 预算即将用尽（已跑 {elapsed/3600:.2f}h），正在保存 -> {self.save_dir}")
            self.stopped_by_budget = True
            os.makedirs(self.save_dir, exist_ok=True)
            try:
                trainer = kwargs.get("trainer")
                if trainer is not None:
                    trainer.save_model(self.save_dir)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(self.save_dir)
                    elif getattr(trainer, "processing_class", None) is not None:
                        trainer.processing_class.save_pretrained(self.save_dir)
                elif model is not None:
                    model.save_pretrained(self.save_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[TimeBudget] 保存失败: {e}")
            control.should_training_stop = True
            json.dump(
                {"stopped_by_budget": True, "elapsed_s": round(elapsed, 1)},
                open(os.path.join(self.save_dir, "timebudget_stop.json"), "w"),
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - (self.start or time.time())
        # 每 30 分钟报一次进度，方便你看着配额表决定要不要提前收手
        if int(elapsed // 1800) != int((elapsed - args.logging_steps) // 1800):
            print(f"[TimeBudget] 已用 {elapsed/3600:.2f} h / 预算 {self.max_seconds/3600:.2f} h", flush=True)
        return self._maybe_stop(args, control, **kwargs)

    def on_log(self, args, state, control, **kwargs):
        return self._maybe_stop(args, control, **kwargs)


def minutes_to_sec(m: float) -> float:
    return float(m) * 60.0


class CheckpointSync(TrainerCallback):
    """每隔 N 步把 ckpt 推到持久化位置（HF Hub 私有仓）。

    免费平台的磁盘在会话结束后不保证保留，这一步是唯一的保险。
    """

    def __init__(self, every_steps: int, local_dir: str, repo_id: Optional[str]):
        self.every = every_steps
        self.local_dir = local_dir
        self.repo_id = repo_id

    def on_save(self, args, state, control, **kwargs):
        if not self.repo_id or state.global_step % self.every != 0:
            return
        from posttrain_lab.runtime import save_checkpoint_hub

        save_checkpoint_hub(args.output_dir, self.repo_id)
