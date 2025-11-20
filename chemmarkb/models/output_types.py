import dataclasses
from typing import Optional

import torch
import numpy as np

from ..chem.mol import Molecule
from ..chem.reaction import Reaction
from ..chem.matrix import ReactantReactionMatrix
from ..data.common import TokenType

from .output_head import ReactantRetrievalResult


@dataclasses.dataclass
class ScoredBuildingBlock:
    molecule: Molecule
    database_index: int
    similarity_score: float

    def __iter__(self):
        return iter([self.molecule, self.database_index, self.similarity_score])


@dataclasses.dataclass
class ScoredTemplate:
    template: Reaction
    template_index: int
    confidence_score: float

    def __iter__(self):
        return iter([self.template, self.template_index, self.confidence_score])


@dataclasses.dataclass
class StepPrediction:
    operation_logits: torch.Tensor
    template_logits: torch.Tensor
    retrieved_blocks: ReactantRetrievalResult

    def to(self, device: torch.device):
        return self.__class__(
            self.operation_logits.to(device),
            self.template_logits.to(device),
            self.retrieved_blocks
        )

    def get_best_operations(self) -> list[TokenType]:
        return [
            TokenType(op_id) 
            for op_id in self.operation_logits.argmax(dim=-1).detach().cpu().tolist()
        ]

    def get_top_templates(
        self, 
        k: int, 
        template_db: ReactantReactionMatrix
    ) -> list[list[ScoredTemplate]]:

        k = min(k, self.template_logits.size(-1))
        scores, indices = self.template_logits.topk(k, dim=-1, largest=True)
        batch_size = scores.size(0)
        
        results = []
        for batch_idx in range(batch_size):
            sample_templates = []
            for rank in range(k):
                template_idx = int(indices[batch_idx, rank].item())
                sample_templates.append(
                    ScoredTemplate(
                        template=template_db.reactions[template_idx],
                        template_index=template_idx,
                        confidence_score=float(scores[batch_idx, rank].item()),
                    )
                )
            results.append(sample_templates)
        return results

    def get_top_building_blocks(self, k: int) -> list[list[ScoredBuildingBlock]]:

        batch_size = self.retrieved_blocks.reactants.shape[0]
        
        distances = self.retrieved_blocks.distance.reshape(batch_size, -1)
        similarity_scores = 1.0 / (distances + 0.1)
        
        indices = self.retrieved_blocks.indices.reshape(batch_size, -1)
        molecules = self.retrieved_blocks.reactants.reshape(batch_size, -1)

        k = min(k, molecules.shape[-1])
        
        rank_order = (-similarity_scores).argsort(axis=-1)

        results = []
        for batch_idx in range(batch_size):
            sample_blocks = []
            for rank in range(k):
                pos = int(rank_order[batch_idx, rank])
                sample_blocks.append(
                    ScoredBuildingBlock(
                        molecule=molecules[batch_idx, pos],
                        database_index=indices[batch_idx, pos],
                        similarity_score=similarity_scores[batch_idx, pos],
                    )
                )
            results.append(sample_blocks)
        return results


@dataclasses.dataclass
class PathwayResult:
    operation_sequence: torch.Tensor 
    template_sequence: torch.Tensor 
    block_features: torch.Tensor
    
    building_blocks: list[list[Optional[Molecule]]] 
    templates: list[list[Optional[Reaction]]]  