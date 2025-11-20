import torch
from torch import nn

from ...data.common import ProjectionBatch
from ..transformer.graph_transformer import GraphTransformer
from .base import BaseEncoder


class GraphEncoder(BaseEncoder):
    
    def __init__(
        self,
        num_atom_classes: int,
        num_bond_classes: int,
        hidden_dim: int,
        num_layers: int,
        attention_head_dim: int,
        edge_feature_dim: int,
        num_attention_heads: int,
        use_relative_position: bool,
        apply_output_norm: bool,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        
        self.atom_embedder = nn.Embedding(
            num_embeddings=num_atom_classes + 1,
            embedding_dim=hidden_dim,
            padding_idx=0
        )
        
        self.bond_embedder = nn.Embedding(
            num_embeddings=num_bond_classes + 1,
            embedding_dim=edge_feature_dim,
            padding_idx=0
        )
        
        self.graph_transformer = GraphTransformer(
            dim=hidden_dim,
            depth=num_layers,
            dim_head=attention_head_dim,
            edge_dim=edge_feature_dim,
            heads=num_attention_heads,
            rel_pos_emb=use_relative_position,
            output_norm=apply_output_norm,
        )
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    @property
    def dim(self) -> int:
        return self._hidden_dim
    
    def forward(
        self,
        batch: ProjectionBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_fields = ["atoms", "bonds", "atom_padding_mask"]
        missing_fields = [key for key in required_fields if key not in batch]
        if missing_fields:
            raise ValueError(
                f"Input batch missing required fields: {missing_fields}. "
                f"Expected fields: {required_fields}"
            )
        
        atom_types = batch["atoms"]
        bond_types = batch["bonds"] 
        padding_mask = batch["atom_padding_mask"]
        atom_vectors = self.atom_embedder(atom_types)
        
        bond_vectors = self.bond_embedder(bond_types)

        atom_features, _ = self.graph_transformer(
            nodes=atom_vectors,
            edges=bond_vectors,
            mask=padding_mask
        )
        
        return atom_features, padding_mask