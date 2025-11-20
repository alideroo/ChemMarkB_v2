import abc
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...chem.fpindex import FingerprintIndex
from ...chem.mol import Molecule
from .base import SimpleMLP, LossDict, AuxDict, ReactantRetrievalResult


class BaseFingerprintHead(nn.Module, abc.ABC):
    
    def __init__(self, fingerprint_dim: int):
        super().__init__()
        self._fingerprint_dim = fingerprint_dim

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    @abc.abstractmethod
    def predict(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        ...

    def retrieve_reactants(
        self,
        h: torch.Tensor,
        fpindex: FingerprintIndex,
        topk: int = 4,
        **options,
    ) -> ReactantRetrievalResult:
        fp = self.predict(h, **options)
        fp_dim = fp.shape[-1]
        
        out = np.empty(list(fp.shape[:-1]) + [topk], dtype=Molecule)
        out_fp = np.empty(list(fp.shape[:-1]) + [topk, fp_dim], dtype=np.float32)
        out_dist = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.float32)
        out_idx = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.int64)

        fp_flat = fp.view(-1, fp_dim)
        out_flat = out.reshape(-1, topk)
        out_fp_flat = out_fp.reshape(-1, topk, fp_dim)
        out_dist_flat = out_dist.reshape(-1, topk)
        out_idx_flat = out_idx.reshape(-1, topk)

        query_res = fpindex.query_cuda(q=fp_flat, k=topk)
        
        for i, q_res_subl in enumerate(query_res):
            for j, q_res in enumerate(q_res_subl):
                out_flat[i, j] = q_res.molecule
                out_fp_flat[i, j] = q_res.fingerprint
                out_dist_flat[i, j] = q_res.distance
                out_idx_flat[i, j] = q_res.index

        return ReactantRetrievalResult(
            reactants=out,
            fingerprint_predicted=fp.detach().cpu().numpy(),
            fingerprint_retrieved=out_fp,
            distance=out_dist,
            indices=out_idx,
        )

    @abc.abstractmethod
    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:
        ...


class MultiFingerprintHead(BaseFingerprintHead):
    
    def __init__(
        self,
        embedding_dim: int,
        num_out_fingerprints: int,
        fingerprint_dim: int,
        dim_hidden: int,
        num_layers: int = 3,
        warmup_prob: float = 1.0,
    ):
        super().__init__(fingerprint_dim=fingerprint_dim)
        self.embedding_dim = embedding_dim
        self.num_out_fingerprints = num_out_fingerprints
        self.warmup_prob = warmup_prob
        
        d_out = fingerprint_dim * num_out_fingerprints
        
        self.mlp = SimpleMLP(embedding_dim, d_out, dim_hidden, num_layers=num_layers)

    def predict(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        y_fingerprint = torch.sigmoid(self.mlp(h))
        
        out_shape = h.shape[:-1] + (self.num_out_fingerprints, self.fingerprint_dim)
        return y_fingerprint.view(out_shape)

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        warmup: bool = False,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:

        bsz, seqlen, _ = h.shape
        
        y_fingerprint = self.mlp(h)  
        fp_shape = [bsz, seqlen, self.num_out_fingerprints, self.fingerprint_dim]
        y_fingerprint = y_fingerprint.view(fp_shape)

        fp_target = fp_target[:, :, None, :].expand(fp_shape)
        
        loss_fingerprint_all = F.binary_cross_entropy_with_logits(
            y_fingerprint,
            fp_target,
            reduction="none",
        ).sum(dim=-1)
        
        loss_fingerprint_min, fp_select = loss_fingerprint_all.min(dim=-1)
        
        if self.training and warmup:
            loss_fingerprint_avg = loss_fingerprint_all.mean(dim=-1)
            
            loss_fingerprint = torch.where(
                torch.rand_like(loss_fingerprint_min) < self.warmup_prob,
                loss_fingerprint_avg,
                loss_fingerprint_min,
            )
        else:
            loss_fingerprint = loss_fingerprint_min
        
        loss_fingerprint = (loss_fingerprint * fp_mask).sum() / (fp_mask.sum() + 1e-6)

        return (
            {"fingerprint": loss_fingerprint}, 
            {"fp_select": fp_select}
        )