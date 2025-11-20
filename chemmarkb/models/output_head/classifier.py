import torch
from torch import nn
from torch.nn import functional as F

from .base import SimpleMLP


class ClassifierHead(nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        num_classes: int, 
        dim_hidden: int | None = None
    ):
        super().__init__()
        dim_hidden = dim_hidden or d_model * 2
        
        self.mlp = SimpleMLP(d_model, num_classes, dim_hidden=dim_hidden)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

    def get_loss(
        self, 
        h: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor | None
    ) -> torch.Tensor:
        logits = self.predict(h)
        
        logits_flat = logits.view(-1, logits.size(-1))  
        target_flat = target.view(-1) 
        
        if mask is not None:
            mask_flat = mask.view(-1)
            total = mask_flat.sum().to(logits_flat) + 1e-6
            loss = F.cross_entropy(
                logits_flat, 
                target_flat, 
                reduction="none"
            )
            loss = (loss * mask_flat).sum() / total
        else:
            loss = F.cross_entropy(logits_flat, target_flat)
        
        return loss