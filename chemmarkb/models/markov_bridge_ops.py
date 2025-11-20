import torch
import numpy as np


class MarkovBridgeOps:
    @staticmethod
    def apply_noise_interpolation(
        X_0: torch.Tensor,
        X_T: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        while mask.dim() < X_T.dim():
            mask = mask.unsqueeze(-1)
        
        X_t = mask * X_0 + (1 - mask) * X_T
        return X_t
    
    @staticmethod
    def create_block_mask(
        bsz: int,
        seq_len: int,
        block_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_blocks = (seq_len + block_size - 1) // block_size
        block_idx = torch.randint(0, num_blocks, (bsz,), device=device)
        
        start_pos = block_idx * block_size + 1
        end_pos = torch.minimum(
            start_pos + block_size,
            torch.tensor(seq_len, device=device)
        )
        
        block_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
        for i in range(bsz):
            block_mask[i, start_pos[i]:end_pos[i]] = True
        
        return block_mask, start_pos, end_pos
    
    @staticmethod
    def skeptical_unmasking(
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        t: torch.Tensor,
        T: int,
        rate_schedule: str = 'linear',
        topk_mode: str = 'deterministic',
    ) -> torch.Tensor:
        if rate_schedule == "linear":
            rate = 1 - (t + 1) / T
        elif rate_schedule == "cosine":
            rate = torch.cos((t + 1) / T * np.pi * 0.5)
        else:
            raise NotImplementedError(f"Unknown rate schedule: {rate_schedule}")
        
        rate = rate.to(scores.device)
        
        cutoff_len = (
            valid_mask.sum(1, keepdim=True).type_as(scores) * rate
        ).long()
        
        _scores_for_topk = scores.masked_fill(~valid_mask, 1000.0)
        
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(_scores_for_topk) + 1e-8
            ) + 1e-8)
            _scores = _scores_for_topk + noise_scale * rate * gumbel_noise
        elif topk_mode == "deterministic":
            _scores = _scores_for_topk
        else:
            raise ValueError(f"Unknown topk_mode: {topk_mode}")
        
        sorted_scores = _scores.sort(-1)[0]
        cutoff = sorted_scores.gather(dim=-1, index=cutoff_len) + 1e-10
        
        update_mask = _scores < cutoff
        
        return update_mask