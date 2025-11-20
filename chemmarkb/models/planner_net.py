import torch
from torch import nn

from .timestep_embed import TimestepEmbedder


class NoiseIdentifier(nn.Module):

    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        self.embed_timesteps = TimestepEmbedder(d_model)
    
    def forward(self, h, t):
        t = self.embed_timesteps(t)
        t_expanded = t.expand(-1, h.size(1), -1)
        h = h + t_expanded
        return self.mlp(h).squeeze(-1) 
    
    def get_loss(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        is_corrupted: torch.Tensor,
    ) -> torch.Tensor:

        probs = self.forward(h, t).squeeze(-1)
        loss = nn.functional.binary_cross_entropy(
            probs, 
            is_corrupted.float(), 
            reduction='mean'
        )
        return loss