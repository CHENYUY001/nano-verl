class RolloutWorker:
    """
    Worker that manages rollout for model inference during rollouts.

    Args:
        config: Configuration object containing rollout settings.
    """

    def __init__(self, config):
        self.config = config

    def generate(self, batches):
        """
        Generate tokens for the given batches.

        Args:
            batches: The input batches for generation, typically containing 'input_ids' and 'attention_mask'.

        Returns:
            dict: A dictionary containing the following tensors:
                - 'prompts' (torch.Tensor): Token IDs of the input prompts. Shape: [B, L_p]
                - 'responses' (torch.Tensor): Token IDs of the generated responses. Shape: [B, L_r]
                - 'tokens' (torch.Tensor): Concatenated prompt and response token IDs. Shape: [B, L_p + L_r]
                - 'logprobs' (torch.Tensor): Log probabilities of the generated response tokens. Shape: [B, L_r]
                - 'attention_mask' (torch.Tensor): Attention mask for the full 'tokens' sequence. Shape: [B, L_p + L_r]
                - 'loss_mask' (torch.Tensor): Mask indicating the response tokens (1 for response, 0 for prompt/padding). Shape: [B, L_p + L_r]
        """
        raise NotImplementedError

    def sync_actor_to_rollout(self):
        """
        Synchronize parameters from the actor model to the rollout model across all workers.
        """
        raise NotImplementedError