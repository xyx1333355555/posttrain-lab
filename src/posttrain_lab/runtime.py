"""运行环境自适应：让同一份代码在 Kaggle / Colab / 魔搭 / 本地都能直接跑。

这是把项目搬到免费算力平台时最容易崩的三个地方：
  1. **精度**：T4 / P100 是 Turing / Volta 架构，**不支持 bf16**，必须切回 fp16
               而 A10 / A100 / 4090 是 Ampere+，可以直接用 bf16
  2. **注意力实现**：flash-attn 在 Turing 上经常编译/运行失败，必须能自动回退到 sdpa
  3. **路径**：每个平台的持久化目录都不一样（/kaggle/working、/content、/mnt/workspace）

这个模块把三件事全部自动化。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Dict, Any


# Ampere 及更新架构（A10 / A100 / H 系列 / RTX 30-40 系）才原生支持 bf16
SDPA_FALLBACK_ARCH = ("T4", "P100", "K80", "V100")


def detect_platform() -> str:
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    if os.path.isdir("/content") and os.environ.get("COLAB_GPU"):
        return "colab"
    if os.path.isdir("/mnt/workspace") or os.environ.get("MODELSCOPE_NOTEBOOK"):
        return "modelscope"
    if os.environ.get("DLWS_ENV"):
        return "ali-dsw"
    return "local"


def gpu_info() -> Dict[str, Any]:
    info = {"name": "unknown", "vram_gb": 0.0, "available": False}
    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def pick_precision(gpu_name: str) -> str:
    gn = (gpu_name or "").upper()
    if any(k in gn for k in ("A10", "A100", "A800", "H100", "H800", "L4", "L40", "4090", "4080", "3090")):
        return "bfloat16"
    return "float16"  # T4 / P100 / 未知 -> 保守用 fp16


def pick_attn(gpu_name: str) -> str:
    gn = (gpu_name or "").upper()
    if any(k in gn for k in SDPA_FALLBACK_ARCH):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:  # noqa: BLE001
        return "sdpa"


def pick_model_size(vram_gb: float, role: str = "main") -> str:
    """按显存推荐模型规模。GRPO 的显存压力远大于 SFT，因为要同时放训练和采样。"""
    if role == "main":
        if vram_gb >= 24:
            return "Qwen/Qwen2.5-Math-1.5B-Instruct"
        if vram_gb >= 16:
            return "Qwen/Qwen2.5-0.5B-Instruct"
        return "Qwen/Qwen2.5-0.5B-Instruct"
    # proxy / 消融
    return "Qwen/Qwen2.5-0.5B-Instruct"


def paths(platform: str) -> Dict[str, str]:
    if platform == "kaggle":
        base = "/kaggle/working"
    elif platform == "colab":
        base = "/content"
    elif platform == "modelscope":
        base = "/mnt/workspace"
    else:
        base = "."
    return {
        "base": base,
        "repo": os.path.join(base, "posttrain-lab"),
        "outputs": os.path.join(base, "posttrain-lab", "outputs"),
        "data": os.path.join(base, "posttrain-lab", "data"),
    }


def setup_mirror(platform: str) -> None:
    """国内平台优先走 hf-mirror / ModelScope，避免下载卡死浪费配额。"""
    if platform in ("kaggle", "colab"):
        # 海外机器直连 HF 更快
        os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def save_checkpoint_hub(local_dir: str, repo_id: str, private: bool = True) -> None:
    """把 checkpoint 推到 HF Hub 私有仓库 —— 免费平台会话结束后文件是会消失的。

    这是免费算力最重要的生存技巧：**不把 checkpoint 推出去，等于白跑。**
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[warn] 未设置 HF_TOKEN，跳过上传。请务必手动下载 /kaggle/working 下的输出！")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, path_in_repo=".")
    print(f"[hub] 已上传 -> {repo_id}")


def summary() -> Dict[str, Any]:
    p = detect_platform()
    g = gpu_info()
    out = {
        "platform": p,
        "gpu": g,
        "precision": pick_precision(g["name"]),
        "attn": pick_attn(g["name"]),
        "recommended_model": pick_model_size(g.get("vram_gb", 0)),
        "paths": paths(p),
    }
    setup_mirror(p)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), ensure_ascii=False, indent=2))
