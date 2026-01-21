# 1. 初始化 (在训练开始前)
# 这一步模型会被加载到 CPU，不占显存
reward_model = NanoRewardModel(
    model_path="Qwen/Qwen2.5-Math-RM-72B", # 举例
    device="cuda",
    offload=True
)

# ... (Actor 模型初始化, vLLM 初始化) ...

# 2. 在 Training Loop 中
for step in range(steps):
    
    # --- 阶段 1: Actor 生成 (vLLM) ---
    prompts = [...]
    # 这里 vLLM 占用了显存
    outputs = llm.generate(prompts, ...)
    responses = [o.outputs[0].text for o in outputs]
    
    # --- 阶段 2: Reward Model 打分 ---
    # 此时 vLLM 暂停，显存空闲或被 swap 出去
    # NanoRewardModel 会自动把参数搬到 GPU，算完搬回去
    
    # 方式 A: 如果 Actor 和 RM Tokenizer 一样，直接用 tensor
    # rewards = reward_model.compute_reward(input_ids, attention_mask)
    
    # 方式 B: 如果不一样，直接传文本 (最省心)
    scores = reward_model.compute_reward_with_text(prompts, responses)
    
    # --- 阶段 3: Actor 训练 (FSDP) ---
    # 此时 RM 已经在 CPU 上了，Actor 可以独占显存进行反向传播
    # ...