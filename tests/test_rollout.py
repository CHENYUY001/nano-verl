"""
测试 SGLangRolloutWorker 的输出

运行方式:
    python tests/test_rollout.py              # 运行 mock 测试
    python tests/test_rollout.py --real       # 运行真实模型测试（需要修改 model_path）
"""
import sys
import torch
from unittest.mock import Mock, patch
from nanoverl.utils.logger import logger


def test_mock():
    """使用 Mock 测试，不需要真实模型"""
    logger.info("========== Mock 测试 ==========")
    
    # Mock sglang.Engine
    with patch('nanoverl.rollout.sglang.Engine') as MockEngine:
        # 配置 mock
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        MockEngine.return_value.tokenizer = mock_tokenizer
        
        # 模拟生成结果
        class MockOutput:
            def __init__(self, token_logprobs):
                self.meta_info = {"output_token_logprobs": token_logprobs}
        
        MockEngine.return_value.generate.return_value = [
            MockOutput([[-0.1, 100], [-0.2, 101], [-0.3, 102]]),
            MockOutput([[-0.15, 200], [-0.25, 201], [-0.35, 202], [-0.4, 203]])
        ]
        
        # 创建 worker
        from nanoverl.rollout.sglang import SGLangRolloutWorker
        
        class Config:
            model_path = "mock-model"
            temperature = 1.0
            top_p = 0.9
            max_new_tokens = 128
        
        worker = SGLangRolloutWorker(Config())
        
        # 准备输入
        batches = {
            "input_ids": torch.tensor([
                [1, 2, 3, 4],
                [5, 6, 7, 0]
            ])
        }
        
        # 执行生成
        result = worker.generate(batches)
        
        # 输出结果
        logger.info(f"输出类型: {type(result)}")
        logger.info(f"输出 keys: {list(result.keys())}")
        logger.info("输出 shapes:")
        logger.info(f"  - prompts:        {result['prompts'].shape}")
        logger.info(f"  - responses:      {result['responses'].shape}")
        logger.info(f"  - tokens:         {result['tokens'].shape}")
        logger.info(f"  - logprobs:       {result['logprobs'].shape}")
        logger.info(f"  - attention_mask: {result['attention_mask'].shape}")
        logger.info(f"  - loss_mask:      {result['loss_mask'].shape}")
        
        logger.info("示例数据:")
        logger.info(f"  - tokens[0]:      {result['tokens'][0]}")
        logger.info(f"  - loss_mask[0]:   {result['loss_mask'][0]}")
        
        logger.info("Mock 测试通过！")


def test_real():
    """使用真实模型测试（需要修改 model_path）"""
    logger.info("========== 真实模型测试 ==========")
    
    # 修改为你的模型路径
    model_path = "Qwen/Qwen2-0.5B"
    
    logger.info(f"加载模型: {model_path}")
    
    from nanoverl.rollout.sglang import SGLangRolloutWorker
    
    class Config:
        model_path = model_path
        temperature = 0.8
        top_p = 0.9
        max_new_tokens = 32
    
    worker = SGLangRolloutWorker(Config())
    
    # 准备输入（假设这些是有效的 token ID）
    batches = {
        "input_ids": torch.tensor([
            [1, 2, 3],
            [4, 5, 6, 7]
        ])
    }
    
    logger.info("开始生成...")
    result = worker.generate(batches)
    
    # 输出结果
    logger.info(f"输出类型: {type(result)}")
    logger.info(f"输出 keys: {list(result.keys())}")
    logger.info("输出 shapes:")
    logger.info(f"  - responses:      {result['responses'].shape}")
    logger.info(f"  - tokens:         {result['tokens'].shape}")
    logger.info(f"  - logprobs:       {result['logprobs'].shape}")
    logger.info(f"  - attention_mask: {result['attention_mask'].shape}")
    logger.info(f"  - loss_mask:      {result['loss_mask'].shape}")
    
    logger.info("生成的 tokens:")
    logger.info(f"  样本 1: {result['responses'][0]}")
    logger.info(f"  样本 2: {result['responses'][1]}")
    
    logger.info("真实模型测试通过！")


if __name__ == "__main__":
    if "--real" in sys.argv:
        test_real()
    else:
        test_mock()
