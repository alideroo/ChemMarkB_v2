import math
import torch
from torch import nn

from .kan import ChebyKAN


class SimpleTimestepEmbedder(nn.Module):

    def __init__(
        self, 
        hidden_size: int, 
        frequency_embedding_size: int = 256
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, 
        dim: int, 
        max_period: int = 10000
    ) -> torch.Tensor:

        half = dim // 2
        
        freqs = torch.exp(
            -math.log(max_period) 
            * torch.arange(start=0, end=half, dtype=torch.float32) 
            / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], 
                dim=-1
            )
        
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.squeeze(-1)
        
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        
        t_emb = self.mlp(t_freq)
        
        return t_emb

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ChebyKAN(
                layers_hidden=[frequency_embedding_size, hidden_size],
                degree=5,
                scale_base=1.0,
                scale_cheby=1.0,
                base_activation=torch.nn.SiLU,
                use_bias=True,
            ),
            nn.SiLU(),
            ChebyKAN(
                layers_hidden=[hidden_size, hidden_size],
                degree=5,
                scale_base=1.0,
                scale_cheby=1.0,
                base_activation=torch.nn.SiLU,
                use_bias=True,
            )
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb