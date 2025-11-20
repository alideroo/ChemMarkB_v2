import torch
from torch import nn

from ...data.common import ProjectionBatch
from ..transformer.positional_encoding import PositionalEncoding
from .base import BaseEncoder


class SMILESEncoder(BaseEncoder):
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        max_sequence_length: int,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        
        self.character_embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0
        )
        
        self.position_encoder = PositionalEncoding(
            d_model=hidden_dim,
            max_len=max_sequence_length,
        )
        
        self.transformer_stack = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    # 向后兼容
    @property
    def dim(self) -> int:
        return self._hidden_dim
    
    def forward(
        self,
        batch: ProjectionBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "smiles" not in batch:
            raise ValueError(
                "Input batch must contain 'smiles' key for SequenceEncoder. "
                "Ensure SMILES strings are tokenized before encoding."
            )
        
        smiles_tokens = batch["smiles"]
        
        embedded = self.character_embedder(smiles_tokens)
        
        positional_encoded = self.position_encoder(embedded)
        
        padding_mask = (smiles_tokens == 0)
        
        sequence_features = self.transformer_stack(
            positional_encoded,
            src_key_padding_mask=padding_mask
        )
        
        return sequence_features, padding_mask