import torch
from nanoverl.reward.reward_model import NanoRewardModel  # 假设你把上面的类保存为了 reward_model.py

def test_nano_reward_model():
    print("=== 开始测试 NanoRewardModel ===")
    
    # 1. 初始化
    # 使用 gpt2 作为测试模型，因为它很小且 HF 默认支持加载为 SequenceClassification
    # 在实际使用中，请替换为你的 Reward Model 路径，如 "Qwen/Qwen2.5-Math-RM-72B"
    test_model_name = "gpt2" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"1. 加载模型: {test_model_name} (Device: {device})")
    rm = NanoRewardModel(model_path=test_model_name, device=device, offload=True)
    
    # 验证初始状态：应该在 CPU 上 (因为 offload=True)
    first_param_device = next(rm.model.parameters()).device
    print(f"   -> 初始模型参数位置: {first_param_device}")
    assert first_param_device.type == "cpu", "错误：初始化后模型应该在 CPU 上"

    # 2. 测试文本打分接口
    print("\n2. 测试 compute_reward_with_text (文本输入)")
    prompts = ["Question: 1+1=?\n", "Question: What is Python?\n"]
    responses = ["Answer: 2", "Answer: A snake."]
    
    scores = rm.compute_reward_with_text(prompts, responses)
    
    print(f"   -> 打分结果: {scores}")
    print(f"   -> 结果形状: {scores.shape}")
    
    # 验证输出
    assert isinstance(scores, torch.Tensor), "错误：输出应该是 Tensor"
    assert scores.shape == (2,), f"错误：Batch size 为 2，但输出形状为 {scores.shape}"
    assert scores.device.type == "cpu", "错误：计算后的结果 Tensor 应该被传回 CPU"

    # 验证计算后模型是否回到了 CPU
    current_param_device = next(rm.model.parameters()).device
    print(f"   -> 计算后模型参数位置: {current_param_device}")
    if rm.offload_to_cpu:
        assert current_param_device.type == "cpu", "错误：计算后模型没有回到 CPU"

    # 3. 测试 Tensor 底层接口 (模拟 vLLM 生成后的数据)
    print("\n3. 测试 compute_reward (Tensor 输入)")
    # 手动构造一些 input_ids (Batch=2, SeqLen=5)
    input_ids = torch.randint(0, 1000, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    
    scores_tensor = rm.compute_reward(input_ids, attention_mask)
    print(f"   -> Tensor 打分结果: {scores_tensor}")
    assert scores_tensor.shape == (2,), "错误：Tensor 输入打分形状不对"

    print("\n=== ✅ 所有测试通过！Nano-Verl Reward 模块工作正常 ===")

if __name__ == "__main__":
    try:
        test_nano_reward_model()
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")