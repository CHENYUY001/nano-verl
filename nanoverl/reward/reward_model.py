import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NanoRewardModel:
    def __init__(self, model_path, device="cuda", offload=True):
        """
        Args:
            model_path: HuggingFace model path
            device: running devices 
            offload: CPU offload(recommend) 
        """
        self.device = device
        self.offload_to_cpu = offload
        
        print(f"[NanoRewardModel] Loading Reward Model from {model_path}...")
        #load model
        #offload model to CPU
        
        # 1. load Tokenizer to identify pad_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            # use eos_token as pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. transfer pad_token_id
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            num_labels=1,
            trust_remote_code=True,
            device_map="cpu",
            pad_token_id=self.tokenizer.pad_token_id 
        )
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def _load_to_device(self):
        """load to GPU"""
        if self.offload_to_cpu:
            self.model.to(self.device)

    def _offload_to_cpu(self):
        """offload to CPU"""
        if self.offload_to_cpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_reward(self, input_ids, attention_mask):
        """
        Computing Reward Score。
        
        Args:
            input_ids: (Batch, SeqLen) - including Prompt + Response
            attention_mask: (Batch, SeqLen)
            
        Returns:
            rewards: (Batch,) - one score for one sample
        """
        # 1. load the model
        self._load_to_device()
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 2. foward pass
        # for most HuggingFace RM output is logits=[batch, 1] or [batch, num_labels]
        # for Reward Model，normally logit==reward
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 3. processing dim shape of output
            if logits.dim() == 2:
                # (Batch, 1) -> (Batch,)
                rewards = logits.squeeze(-1) 
            elif logits.dim() == 1:
                rewards = logits
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
        except Exception as e:
            print(f"[NanoRewardModel] Error during computation: {e}")
            rewards = torch.zeros(input_ids.size(0), device=self.device)
        # 4. memory optimization
        self._offload_to_cpu()
        
        return rewards.cpu()

    @torch.no_grad()
    def compute_reward_with_text(self, prompts, responses):
        """
        nano verl interface: processing text directly。
        """
        assert self.tokenizer is not None, "Tokenizer not loaded!"
        
        # text joint (change with specific Chat Template)
        # assume easy combine (maybe apply_chat_template)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        inputs = self.tokenizer(
            full_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=2048
        )
        
        return self.compute_reward(inputs["input_ids"], inputs["attention_mask"])