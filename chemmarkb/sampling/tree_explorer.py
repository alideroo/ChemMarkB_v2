import copy
import dataclasses
import itertools
import time
from collections.abc import Iterable
from functools import cached_property
from multiprocessing.synchronize import Lock
from typing import Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from ..chem.fpindex import FingerprintIndex
from ..chem.matrix import ReactantReactionMatrix
from ..chem.mol import FingerprintOption, Molecule
from ..chem.stack import Stack
from ..data.collate import (
    apply_collate,
    collate_1d_features,
    collate_padding_masks,
    collate_tokens,
)
from ..data.common import TokenType, featurize_stack
from ..models.molebridge import MoleBridge


@dataclasses.dataclass
class PathwayNode:
    operations: Stack = dataclasses.field(default_factory=Stack)
    step_scores: list[float] = dataclasses.field(default_factory=list)

    @property
    def total_score(self) -> float:
        return sum(self.step_scores)

    def to_features(self, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
        return featurize_stack(self.operations, end_token=False, fpindex=fpindex)


@dataclasses.dataclass
class SynthesisResult:
    product: Molecule
    operations: Stack


class ExecutionTimer:
    
    def __init__(self, max_seconds: float) -> None:

        self._max_seconds = max_seconds
        self._start_time = time.time()

    def is_expired(self) -> bool:
        if self._max_seconds <= 0:
            return False
        return time.time() - self._start_time > self._max_seconds

    def validate(self):
        if self.is_expired():
            raise TimeoutError("Execution time limit exceeded")


class BeamSearchExplorer:

    def __init__(
        self,
        block_index: FingerprintIndex,
        template_db: ReactantReactionMatrix,
        target: Molecule,
        network: MoleBridge,
        beam_width: int = 16,
        max_candidates: int = 256,
    ) -> None:
        self._block_index = block_index
        self._template_db = template_db
        self._network = network
        self._target = target
        
        compute_device = next(iter(network.parameters())).device
        atom_feats, bond_feats = target.featurize_simple()
        smiles_tokens = target.tokenize_csmiles()
        self._atom_feats = atom_feats[None].to(compute_device)
        self._bond_feats = bond_feats[None].to(compute_device)
        self._smiles_tokens = smiles_tokens[None].to(compute_device)
        num_atoms = atom_feats.size(0)
        self._atom_mask = torch.zeros(
            [1, num_atoms], dtype=torch.bool, device=compute_device
        )
        
        self._beam_width = beam_width
        self._max_candidates = max_candidates
        
        self._candidates: list[PathwayNode] = [PathwayNode()]
        self._completed: list[PathwayNode] = []
        self._failed: list[PathwayNode] = []

    @cached_property
    def compute_device(self) -> torch.device:
        return self._atom_feats.device

    @cached_property
    def target_encoding(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return self._network.encode_molecule(
                {
                    "atoms": self._atom_feats,
                    "bonds": self._bond_feats,
                    "atom_padding_mask": self._atom_mask,
                    "smiles": self._smiles_tokens,
                }
            )

    def _rank_and_prune(self) -> None:
        self._candidates.sort(key=lambda n: n.total_score, reverse=True)
        self._candidates = self._candidates[:self._max_candidates]

    def _batch_features(
        self, 
        feature_list: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        collate_specs = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }
        return apply_collate(
            collate_specs, 
            feature_list, 
            feature_list[0]["token_types"].size(0)
        )

    def expand_once(
        self,
        device_lock: Optional[Lock] = None,
        show_progress: bool = False,
        timer: Optional[ExecutionTimer] = None,
    ) -> None:
        if len(self._candidates) == 0:
            return
        
        feature_batch = [
            featurize_stack(
                node.operations,
                end_token=False,
                fpindex=self._block_index,
            )
            for node in self._candidates
        ]
        
        if device_lock is not None:
            device_lock.acquire()
        
        features = {
            k: v.to(self.compute_device) 
            for k, v in self._batch_features(feature_batch).items()
        }
        
        encoding, enc_mask = self.target_encoding
        batch_size = len(feature_batch)
        encoding = encoding.expand([batch_size] + list(encoding.shape[1:]))
        enc_mask = enc_mask.expand([batch_size] + list(enc_mask.shape[1:]))
        
        predictions = self._network.predict_next_segment(
            mol_encoding=encoding,
            encoding_mask=enc_mask,
            step_types=features["token_types"],
            template_ids=features["rxn_indices"],
            block_fingerprints=features["reactant_fps"],
            rxn_matrix=self._template_db,
            fpindex=self._block_index,
            num_candidates=self._beam_width,
        )
        
        if device_lock is not None:
            device_lock.release()
        
        num_nodes = encoding.size(0)
        num_branches = self._beam_width
        expansion_iter: Iterable[tuple[int, int]] = itertools.product(
            range(num_nodes), range(num_branches)
        )
        
        if show_progress:
            expansion_iter = tqdm(
                expansion_iter, 
                total=num_nodes * num_branches, 
                desc="Expanding", 
                dynamic_ncols=True
            )
        
        best_ops = predictions.get_best_operations()
        top_blocks = predictions.get_top_building_blocks(k=num_branches)
        top_templates = predictions.get_top_templates(
            k=num_branches, template_db=self._template_db
        )
        
        next_candidates: list[PathwayNode] = []
        
        for node_idx, branch_idx in expansion_iter:
            if timer is not None and timer.is_expired():
                break
            
            next_operation = best_ops[node_idx]
            base_node = self._candidates[node_idx]
            
            if next_operation == TokenType.END:
                self._completed.append(base_node)
            
            elif next_operation == TokenType.REACTANT:
                block, block_id, block_score = top_blocks[node_idx][branch_idx]
                new_node = copy.deepcopy(base_node)
                new_node.operations.push_mol(block, block_id)
                new_node.step_scores.append(block_score)
                next_candidates.append(new_node)
            
            elif next_operation == TokenType.REACTION:
                template, tmpl_id, tmpl_score = top_templates[node_idx][branch_idx]
                new_node = copy.deepcopy(base_node)
                success = new_node.operations.push_rxn(
                    template, tmpl_id, product_limit=None
                )
                if success:
                    reaction_score = max([
                        self._target.sim(m, fp_option=FingerprintOption.rdkit())
                        for m in new_node.operations.get_top()
                    ])
                    new_node.step_scores.append(reaction_score)
                    next_candidates.append(new_node)
                else:
                    self._failed.append(new_node)
            
            else:
                self._failed.append(base_node)
        
        del self._candidates
        self._candidates = next_candidates
        self._rank_and_prune()

    def get_solutions(self) -> Iterable[SynthesisResult]:
        seen_products: set[Molecule] = set()
        for node in self._completed:
            for product in node.operations.get_top():
                if product in seen_products:
                    continue
                yield SynthesisResult(product, node.operations)
                seen_products.add(product)

    def create_result_table(
        self, 
        num_detailed_metrics: int = 10
    ) -> pd.DataFrame:
        records: list[dict[str, str | float]] = []
        smiles_lookup: dict[str, Molecule] = {}
        
        for solution in self.get_solutions():
            records.append({
                "target": self._target.smiles,
                "smiles": solution.product.smiles,
                "score": self._target.sim(
                    solution.product,
                    FingerprintOption.morgan_for_tanimoto_similarity()
                ),
                "synthesis": solution.operations.get_action_string(),
                "num_steps": solution.operations.count_reactions(),
            })
            smiles_lookup[solution.product.smiles] = solution.product
        
        records.sort(key=lambda r: r["score"], reverse=True)
        
        for record in records[:num_detailed_metrics]:
            product = smiles_lookup[str(record["smiles"])]
            record["scf_sim"] = self._target.scaffold.tanimoto_similarity(
                product.scaffold,
                fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
            )
            record["pharm2d_sim"] = self._target.dice_similarity(
                product, fp_option=FingerprintOption.gobbi_pharm2d()
            )
            record["rdkit_sim"] = self._target.tanimoto_similarity(
                product, fp_option=FingerprintOption.rdkit()
            )
        
        return pd.DataFrame(records)

    def print_statistics(self) -> None:
        print(f"Candidates: {len(self._candidates)}")
        print(f"Completed: {len(self._completed)}")
        print(f"Failed: {len(self._failed)}")