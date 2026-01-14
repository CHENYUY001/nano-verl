#!/bin/bash
set -x
# ================= 环境变量配置 =================
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
unset ROCR_VISIBLE_DEVICES

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=4
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# ================= 路径配置 =================
HOME_DIR=/hpc2hdd/home/cyuan866/Hybrid_RL
TRAIN_FILE=$HOME_DIR/data/gsm8k/train.parquet
TEST_FILE=$HOME_DIR/data/gsm8k/test.parquet
MODEL_PATH=$HOME_DIR/Qwen/Qwen2.5-0.5B-Instruct

# ================= 训练参数 =================
TRAIN_BATCH_SIZE=8
MINI_BATCH_SIZE=4
MICRO_BATCH_SIZE=2
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
VLLM_MAX_MODEL_LEN=4096 

echo "开始 PPO 训练 (Critic 不使用 LoRA)..."

# 使用 uv 环境下的 python 启动
PYTHONPATH=. .venv/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.dataloader_num_workers=1 \
    data.filter_overlong_prompts_workers=2 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
     data.shuffle=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=8 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
    ++actor_rollout_ref.actor.use_torch_compile=False \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=$VLLM_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    \
    +actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.strategy=fsdp \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    ++actor_rollout_ref.ref.use_torch_compile=False \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name="verl_grpo_data_disk" \
    trainer.experiment_name="qwen2_5_data_disk_demo" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$HOME_DIR/checkpoints" \
    trainer.max_actor_ckpt_to_keep=5 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'hf_model', 'optimizer', 'extra']" \
    2>&1 | tee run_ppo.log

: '

    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_model_len=$VLLM_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.strategy=fsdp \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    \
    critic.strategy=fsdp \
    critic.optim.lr=1e-5 \
    critic.model.path="$MODEL_PATH" \
    critic.model.use_remove_padding=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.lora_rank=0 \
    critic.model.target_modules=all-linear \
    +critic.fsdp_config.model_dtype=bfloat16 \
    +critic.fsdp_config.use_orig_params=True \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    +trainer.tensorboard_log_dir="tensorboard_logs" \
    trainer.project_name="verl_ppo_gsm8k" \
    trainer.experiment_name="qwen2.5_0.5b_ppo" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$HOME_DIR/checkpoints" \
    trainer.max_actor_ckpt_to_keep=5 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'hf_model', 'optimizer', 'extra']" \
    2>&1 | tee run_ppo.log
    '