import torch
from torch.nn.utils.rnn import pad_sequence
from sglang import Engine
from nanoverl.rollout.base import RolloutWorker
from nanoverl.utils.logger import logger

class SGLangRolloutWorker(RolloutWorker):
    """
    使用 sglang.Engine 实现的最小 RolloutWorker
    """

    def __init__(self, config):
        super().__init__(config)
        
        # 1. 初始化 SGLang 推理引擎
        # model_path 可以是本地路径或 HuggingFace ID
        self.engine = Engine(
            model_path=config.model_path,
            tp_size=1,                    # 单卡推理
            mem_fraction_static=0.8,      # 显存占比，留一部分给训练
            trust_remote_code=True
        )
        
        # 2. 获取 tokenizer 的 pad_token_id 用于后续 padding
        self.pad_token_id = self.engine.tokenizer.pad_token_id or 0
        
        # 3. 配置采样参数（sglang 接受 dict 格式）
        self.sampling_params = {
            "temperature": getattr(config, "temperature", 1.0),
            "top_p": getattr(config, "top_p", 1.0),
            "max_new_tokens": getattr(config, "max_new_tokens", 1024),
        }

    def generate(self, batches):
        """
        输入 batches 包含 'input_ids' [B, L_p]
        返回要求的 tensor 字典
        """
        # 将 torch tensor 转为 list[list[int]] 给 sglang
        input_ids_list = batches["input_ids"].tolist()
        
        # 调用 sglang 引擎进行推理
        # sglang 内部会自动处理 batching
        outputs = self.engine.generate(
            input_ids=input_ids_list,
            sampling_params=self.sampling_params,
            return_logprob=True
        )

        all_prompts = batches["input_ids"]
        all_responses = []
        all_logprobs = []
        all_tokens = []

        for i, output in enumerate(outputs):
            # 获取生成的 token IDs 和对应的 logprobs
            # output.meta_info['output_token_logprobs'] 格式为: [[logprob, id], ...]
            resp_tokens = [item[1] for item in output.meta_info["output_token_logprobs"]]
            resp_logprobs = [item[0] for item in output.meta_info["output_token_logprobs"]]
            
            # 转换为 tensor
            resp_tensor = torch.tensor(resp_tokens)
            logprob_tensor = torch.tensor(resp_logprobs)
            
            all_responses.append(resp_tensor)
            all_logprobs.append(logprob_tensor)
            
            # 拼接 prompt 和 response
            all_tokens.append(torch.cat([all_prompts[i], resp_tensor]))

        # 使用 pad_sequence 处理变长输出
        responses = pad_sequence(all_responses, batch_first=True, padding_value=self.pad_token_id)
        logprobs = pad_sequence(all_logprobs, batch_first=True, padding_value=0.0)
        tokens = pad_sequence(all_tokens, batch_first=True, padding_value=self.pad_token_id)
        
        # 计算 Mask
        attention_mask = (tokens != self.pad_token_id).long()
        
        # loss_mask: response 部分为 1，prompt 部分为 0
        loss_mask = torch.zeros_like(tokens)
        for i, resp in enumerate(all_responses):
            prompt_len = all_prompts[i].size(0)
            loss_mask[i, prompt_len : prompt_len + len(resp)] = 1

        return {
            'prompts': all_prompts,
            'responses': responses,
            'tokens': tokens,
            'logprobs': logprobs,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask
        }

    def sync_actor_to_rollout(self, actor_model=None):
        """
        核心同步逻辑：将 Actor 的权重推送到推理引擎
        """
        if actor_model is None:
            return
            
        # sglang 提供了直接更新权重的 API
        # 接收一个 dict: {name: tensor}
        # 注意：tensor 必须在显存中且与推理卡对应
        state_dict = actor_model.state_dict()
        
        # 这里的 key 映射需要根据模型实现微调，sglang 默认支持大部分 HF 格式
        self.engine.update_weights(state_dict)
        logger.info("Rollout weights synchronized via sglang.update_weights")