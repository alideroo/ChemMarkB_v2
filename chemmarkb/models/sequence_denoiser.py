from typing import Optional, TYPE_CHECKING
import torch
from torch import nn

from ..data.common import TokenType
from .transformer.positional_encoding import PositionalEncoding
from .timestep_embed import TimestepEmbedder


def _create_projection_network(
    input_dim: int, 
    output_dim: int, 
    hidden_dim: int
) -> nn.Sequential:

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SequenceDecoder(nn.Module):
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        ffn_size: int = 2048,
        num_layers: int = 6,
        max_position: int = 32,
        apply_output_norm: bool = False,
        block_feature_dim: int = 256,
        projection_hidden: int = 512,
        num_templates: int = 100,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.operation_embedder = nn.Embedding(max(TokenType) + 1, hidden_size)

        self.template_embedder = nn.Embedding(num_templates, hidden_size)
        
        self.block_projector = _create_projection_network(
            block_feature_dim, 
            hidden_size, 
            hidden_dim=projection_hidden
        )
        
        self.position_encoder = PositionalEncoding(
            d_model=hidden_size, 
            max_len=max_position
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size) if apply_output_norm else None,
        )
        
        self.time_encoder = TimestepEmbedder(hidden_size)

    def create_empty_condition(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = torch.zeros(
            [batch_size, 0, self.hidden_size], 
            dtype=dtype, 
            device=device
        )
        mask = torch.zeros(
            [batch_size, 0], 
            dtype=torch.bool, 
            device=device
        )
        return condition, mask

    def embed_sequence(
        self,
        operation_types: torch.Tensor,
        template_ids: torch.Tensor,
        block_features: torch.Tensor,
    ) -> torch.Tensor:

        op_encoding = self.operation_embedder(operation_types)
        template_encoding = self.template_embedder(template_ids)
        block_encoding = self.block_projector(block_features)
        
        op_type_expanded = operation_types.unsqueeze(-1).expand(
            operation_types.size(0), operation_types.size(1), self.hidden_size
        )
        
        combined_encoding = torch.where(
            op_type_expanded == TokenType.REACTION, 
            template_encoding, 
            op_encoding
        )
        combined_encoding = torch.where(
            op_type_expanded == TokenType.REACTANT,
            block_encoding,
            combined_encoding
        )
        
        combined_encoding = self.position_encoder(combined_encoding)
        
        return combined_encoding

    def forward(
        self,
        condition: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        operation_types: torch.Tensor,
        template_ids: torch.Tensor,
        block_features: torch.Tensor,
        sequence_mask: Optional[torch.Tensor],
        time_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:

        batch_size, seq_length = operation_types.size()
        
        if condition is None:
            condition, condition_mask = self.create_empty_condition(
                batch_size, 
                device=block_features.device,
                dtype=block_features.dtype
            )

        sequence_encoding = self.embed_sequence(
            operation_types, template_ids, block_features
        )

        if time_condition is not None:
            time_embedding = self.time_encoder(time_condition) 
            time_expanded = time_embedding.expand(
                -1, condition.size(1), -1
            )
            condition = condition + time_expanded

        autoregressive_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=sequence_encoding.size(1),
            dtype=sequence_encoding.dtype,
            device=sequence_encoding.device,
        )
        
        sequence_key_mask = None
        if sequence_mask is not None:
            sequence_key_mask = torch.zeros(
                [batch_size, seq_length],
                dtype=autoregressive_mask.dtype,
                device=autoregressive_mask.device,
            ).masked_fill_(
                sequence_mask, 
                -torch.finfo(autoregressive_mask.dtype).max
            )
            
        output: torch.Tensor = self.transformer_decoder(
            tgt=sequence_encoding,
            memory=condition,
            tgt_mask=autoregressive_mask,
            tgt_key_padding_mask=sequence_key_mask,
            memory_key_padding_mask=condition_mask,
        )
        
        return output

    if TYPE_CHECKING:
        def __call__(
            self,
            condition: Optional[torch.Tensor],
            condition_mask: Optional[torch.Tensor],
            operation_types: torch.Tensor,
            template_ids: torch.Tensor,
            block_features: torch.Tensor,
            sequence_mask: Optional[torch.Tensor],
            time_condition: Optional[torch.Tensor],
        ) -> torch.Tensor: ...