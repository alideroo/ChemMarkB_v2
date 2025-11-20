import dataclasses
from typing import TypeAlias, Callable

import numpy as np
import torch
from torch import nn

from ...chem.mol import Molecule

LossDict: TypeAlias = dict[str, torch.Tensor]

AuxDict: TypeAlias = dict[str, torch.Tensor]

def SimpleMLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: int,
    num_layers: int = 3,
) -> nn.Sequential:

    if num_layers < 2:
        raise ValueError(f"num_layers must be at least 2, got {num_layers}")
    
    num_intermediate = num_layers - 2
    
    layers = [
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
    ]
    
    for _ in range(num_intermediate):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(dim_hidden, dim_out))
    
    return nn.Sequential(*layers)

@dataclasses.dataclass
class ReactantRetrievalResult:
    reactants: np.ndarray
    fingerprint_predicted: np.ndarray
    fingerprint_retrieved: np.ndarray
    distance: np.ndarray
    indices: np.ndarray