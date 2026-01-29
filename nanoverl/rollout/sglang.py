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
        self.engine = Engine(
            model_path=config.model_path,
            tp_size=1,                    # 单卡推理
            mem_fraction_static=0.2,      # 显存占比
            trust_remote_code=True
        )

        # 3. 配置采样参数（sglang 接受 dict 格式）
        self.sampling_params = {
            "temperature": getattr(config, "temperature", 1.0),
            "top_p": getattr(config, "top_p", 1.0),
            "max_new_tokens": getattr(config, "max_new_tokens", 1024),
        }


    def generate(self, batches):
        input_ids = batches["input_ids"]
        # 上游保证提供 prompt attention_mask（与 input_ids 对齐）
        prompt_attention_mask = batches["attention_mask"]
        # 约定：上游已保证都是 torch.Tensor 且已在正确 device 上
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(prompt_attention_mask, torch.Tensor)

        # 2. 推理: 调用 sglang
        # 注意：sglang Engine.generate 这里需要 python 的 list[list[int]] 作为 batch 输入
        input_ids_list = input_ids.tolist()
        outputs = self.engine.generate(
            input_ids=input_ids_list,
            sampling_params=self.sampling_params,
            return_logprob=True
        )

        # 3. 解析输出: outputs 是 List[dict]，每条 dict 至少包含:
        # - "output_ids": response token ids (list[int])
        # - "meta_info"["output_token_logprobs"]: [(logprob, token_id, token_text_or_None), ...]
        response_list = []
        logprob_list = []
        for output in outputs:
            response_list.append(output["output_ids"])
            logprob_list.append(output["meta_info"]["output_token_logprobs"])
        
        # 4. ragged -> padded tensors
        # - responses: List[List[int]] -> LongTensor [B, Lr]
        # - logprobs:  List[List[tuple]] -> FloatTensor [B, Lr]
        device = input_ids.device

        response_tensors = [
            torch.tensor(r, dtype=torch.long, device=device) for r in response_list
        ]
        resp_lens = torch.tensor([t.numel() for t in response_tensors], dtype=torch.long, device=device)
        max_lr = int(resp_lens.max().item()) if resp_lens.numel() > 0 else 0

        if max_lr == 0:
            # 全部没有生成 token 的极端情况
            responses = torch.empty((len(response_tensors), 0), dtype=torch.long, device=device)
            logprobs = torch.empty((len(response_tensors), 0), dtype=torch.float32, device=device)
            response_attention_mask = torch.empty((len(response_tensors), 0), dtype=torch.long, device=device)
        else:
            # responses padding（pad id 对 response 本身意义不大，反正会配合 mask）
            responses = pad_sequence(response_tensors, batch_first=True, padding_value=0)
            # response attention mask: 1 for valid response tokens, 0 for padding
            arange_lr = torch.arange(max_lr, device=device).unsqueeze(0)  # [1, Lr]
            response_attention_mask = (arange_lr < resp_lens.unsqueeze(1)).to(dtype=torch.long)

            # logprobs: 取 meta_info 的第 0 个字段 (logprob)，并 pad 到 max_lr
            logprob_tensors = []
            for lp in logprob_list:
                # lp: [(logprob, token_id, token_text_or_None), ...]
                vals = [float(x[0]) for x in lp]
                logprob_tensors.append(torch.tensor(vals, dtype=torch.float32, device=device))
            logprobs = pad_sequence(logprob_tensors, batch_first=True, padding_value=0.0)

        # 5. tokens / attention_mask / loss_mask
        # prompts: 期望为 [B, Lp]（已经是 padding 后的 prompt）
        prompts = input_ids
        prompt_attention_mask = prompt_attention_mask.to(dtype=torch.long, device=device)

        # 拼接得到全序列 tokens（prompt padding + response padding）
        tokens = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)

        # loss_mask: 只对 response 的有效 token 计算 loss / reward，prompt 和 padding 都是 0
        loss_mask = torch.cat([torch.zeros_like(prompt_attention_mask), response_attention_mask], dim=1)

        return {
            "prompts": prompts,
            "responses": responses,
            "tokens": tokens,
            "logprobs": logprobs,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    def sync_actor_to_rollout(self, actor_model=None):
        """
        核心同步逻辑：将 Actor 的权重推送到推理引擎
        """
        if actor_model is None:
            return


        state_dict = actor_model.state_dict()
        
        self.engine.update_weights(state_dict)
        logger.info("Rollout weights synchronized via sglang.update_weights")