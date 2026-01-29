import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

# 允许直接 python 运行该文件
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nanoverl.rollout.sglang import SGLangRolloutWorker
from nanoverl.utils.logger import logger


def _summarize(name, x):
    if isinstance(x, torch.Tensor):
        logger.info(
            f"{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
        )
    else:
        logger.info(f"{name}: type={type(x)} value_preview={str(x)[:200]}")


def main():
    model_path = os.environ.get("MODEL_PATH", "/hpc2hdd/home/cyuan866/Hybrid_RL/Qwen/Qwen2.5-3B-Instruct/dir/")
    logger.info(f"Using model_path={model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # 常见：Qwen/LLaMA 系列默认没有 pad token，用 eos 代替
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "你好，简单介绍一下你自己。",
        "用一句话解释 attention_mask 是什么。",
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # 让输入跟 rollout/engine 一致（通常在 GPU 上）
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    config = SimpleNamespace(
        model_path=model_path,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=16,
    )
    worker = SGLangRolloutWorker(config)

    out = worker.generate({"input_ids": input_ids, "attention_mask": attention_mask})

    logger.info("=== rollout.generate() return summary ===")
    for k in ["prompts", "responses", "tokens", "logprobs", "attention_mask", "loss_mask"]:
        _summarize(k, out[k])

    # 一些一致性断言（不通过就说明格式不符合预期）
    assert out["tokens"].shape == out["attention_mask"].shape
    assert out["tokens"].shape == out["loss_mask"].shape
    assert out["responses"].shape == out["logprobs"].shape
    assert out["tokens"].shape[0] == out["prompts"].shape[0] == out["responses"].shape[0]

    logger.info("Sanity checks passed.")


if __name__ == "__main__":
    main()
