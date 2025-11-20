from .base import BaseEncoder
from .smiles_encoder import SMILESEncoder
from .graph_encoder import GraphEncoder


def get_encoder(encoder_type: str, cfg) -> BaseEncoder:
    if encoder_type == "smiles":
        return SMILESEncoder(**cfg)
    elif encoder_type == "graph":
        return GraphEncoder(**cfg)
    else:
        raise ValueError(
            f"Unknown encoder type: '{encoder_type}'. "
            f"Supported types: 'smiles', 'graph'"
        )


__all__ = [
    'BaseEncoder',
    'SMILESEncoder',
    'GraphEncoder',
    'get_encoder',
]