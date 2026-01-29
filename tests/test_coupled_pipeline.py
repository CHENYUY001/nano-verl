import os
import torch
import sys
from types import SimpleNamespace
from transformers import AutoTokenizer
from pathlib import Path
# 允许直接 python 运行该文件
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# 导入你现有的模块
from nanoverl.rollout.sglang import SGLangRolloutWorker
from nanoverl.reward.reward_model import NanoRewardModel
from nanoverl.utils.logger import logger

def test_rlhf_data_flow():
    logger.info("=== 开始耦合流程测试 (Rollout + Reward) ===")

    # 1. 配置路径（请根据你的实际路径调整）
    # 使用 3B 模型做推理，gpt2 做打分测试
    actor_path = os.environ.get("MODEL_PATH", "/hpc2hdd/home/cyuan866/Hybrid_RL/Qwen/Qwen2.5-3B-Instruct/dir/")
    reward_path = "/hpc2hdd/home/cyuan866/Hybrid_RL/Qwen/Qwen2.5-3B-Instruct/dir/" # 实际使用时换成你的 RM 路径
    
    # 2. 初始化 Rollout Worker
    rollout_config = SimpleNamespace(
        model_path=actor_path,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=32,
        mem_fraction_static=0.2, # 为 Reward Model 留出显存空间
    )
    logger.info("正在初始化 SGLangRolloutWorker...")
    worker = SGLangRolloutWorker(rollout_config)

    # 3. 初始化 Reward Model
    logger.info("正在初始化 NanoRewardModel (Offload=True)...")
    rm = NanoRewardModel(model_path=reward_path, device="cuda", offload=True)

    # 4. 模拟输入数据
    tokenizer = AutoTokenizer.from_pretrained(actor_path)
    prompts = [
        "请写一首关于人工智能的短诗。",
        "如何评价大语言模型的异步训练技术？"
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    # ---------------------------------------------------------
    # 核心耦合逻辑开始
    # ---------------------------------------------------------

    # 第一步：Rollout 生成 (推理阶段)
    logger.info("执行 Rollout.generate()...")
    rollout_output = worker.generate({
        "input_ids": inputs["input_ids"], 
        "attention_mask": inputs["attention_mask"]
    })

    # 第二步：使用 Rollout 的输出作为 Reward Model 的输入 (打分阶段)
    # 你的接口设计得很好：rollout_output["tokens"] 已经拼接好了 Prompt + Response
    logger.info("执行 RewardModel.compute_reward()...")
    scores = rm.compute_reward(
        input_ids=rollout_output["tokens"],
        attention_mask=rollout_output["attention_mask"]
    )

    # 第三步：整合结果
    # 将打分结果塞回字典，后续可以传给 PPO Trainer
    rollout_output["rewards"] = scores

    # ---------------------------------------------------------
    # 核心耦合逻辑结束
    # ---------------------------------------------------------

    # 5. 验证数据流一致性
    batch_size = len(prompts)
    
    logger.info("=== 数据流一致性检查 ===")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Tokens Shape: {rollout_output['tokens'].shape}")
    logger.info(f"Scores Shape: {rollout_output['rewards'].shape}")
    logger.info(f"Scores Values: {rollout_output['rewards'].tolist()}")

    # 断言检查
    assert rollout_output["rewards"].shape == (batch_size,), "Reward 数量与 Batch 不匹配"
    assert rollout_output["tokens"].device.type == "cuda", "Tokens 应该在 GPU 上"
    assert rollout_output["rewards"].device.type == "cpu", "最终 Score 应该被回传到了 CPU"

    logger.info("✅ 耦合测试成功：Rollout 数据已成功流向 Reward Model 并产出分数。")

if __name__ == "__main__":
    try:
        test_rlhf_data_flow()
        logger.info("Test finished successfully.")
        
        # 核心修改：使用 os._exit(0) 强制退出
        # 这会立即停止当前进程及其追踪器，避免弹出 leaked semaphore 警告
        os._exit(0) 
        
    except Exception as e:
        logger.error(f"❌ 测试中途出错: {e}")
        os._exit(1)