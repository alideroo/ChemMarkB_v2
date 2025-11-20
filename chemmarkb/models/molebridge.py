import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional

from ..chem.fpindex import FingerprintIndex
from ..chem.matrix import ReactantReactionMatrix
from ..chem.mol import Molecule
from ..chem.reaction import Reaction
from ..data.common import ProjectionBatch, TokenType

from .sequence_denoiser import SequenceDecoder
from .encoder import get_encoder
from .output_head import (
    BaseFingerprintHead,
    ClassifierHead,
    MultiFingerprintHead,
)
from .schedule import PredefinedNoiseScheduleDiscrete
from .planner_net import NoiseIdentifier
from .markov_bridge_ops import MarkovBridgeOps
from .output_types import StepPrediction, PathwayResult

class MoleBridge(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.mol_encoder = get_encoder(cfg.encoder_type, cfg.encoder)
        self.seq_decoder = SequenceDecoder(**cfg.decoder)
        self.hidden_dim: int = self.mol_encoder.dim
        
        self.step_classifier = ClassifierHead(
            self.hidden_dim,
            max(TokenType) + 1,
            dim_hidden=cfg.step_classifier.hidden_dim,
        )
        
        self.template_classifier = ClassifierHead(
            self.hidden_dim,
            cfg.template_classifier.num_templates,
            dim_hidden=cfg.template_classifier.hidden_dim,
        )
        
        self.building_block_retriever: BaseFingerprintHead = MultiFingerprintHead(
            **cfg.building_block_retriever
        )

        self.num_denoise_steps = cfg.T
        
        self.noise_scheduler = PredefinedNoiseScheduleDiscrete(
            noise_schedule='interpolation',
            timesteps=self.num_denoise_steps
        )
        
        self.noise_identifier = NoiseIdentifier(
            self.hidden_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.segment_length = cfg.segment_length
        
        self.bridge_operator = MarkovBridgeOps()

    def encode_molecule(
        self, 
        batch: ProjectionBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        return self.mol_encoder(batch)
    
    def compute_training_loss(
        self,
        mol_encoding: Optional[torch.Tensor],
        encoding_mask: Optional[torch.Tensor],
        step_types: torch.Tensor,
        template_ids: torch.Tensor,
        block_fingerprints: torch.Tensor,
        sequence_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        bsz, seq_len = step_types.shape

        target_steps = step_types.clone()
        target_templates = template_ids.clone()
        target_blocks = block_fingerprints.clone()

        noisy_steps = torch.full_like(
            target_steps, TokenType.START
        ).to(step_types.device)

        noisy_templates = torch.zeros_like(target_templates).to(template_ids.device)
        noisy_blocks = torch.zeros_like(target_blocks).to(block_fingerprints.device)

        segment_mask, seg_start, seg_end = self.bridge_operator.create_block_mask(
            bsz, seq_len, self.segment_length, step_types.device
        )

        min_time = 0 if self.training else 1
        time_discrete = torch.randint(
            min_time, self.num_denoise_steps, 
            size=(bsz, 1), 
            device=step_types.device
        ).float()
        time_normalized = time_discrete / self.num_denoise_steps

        noise_level = self.noise_scheduler.get_alpha_bar(t_normalized=time_normalized)

        corruption_mask = torch.bernoulli(
            noise_level.repeat((1, seq_len))
        ).int().to(step_types.device)

        active_corruption = torch.zeros_like(segment_mask, dtype=torch.int)
        active_corruption[segment_mask] = corruption_mask[segment_mask]

        noisy_steps = self.bridge_operator.apply_noise_interpolation(
            noisy_steps, target_steps, active_corruption
        )
        noisy_templates = self.bridge_operator.apply_noise_interpolation(
            noisy_templates, target_templates, active_corruption
        )
        noisy_blocks = self.bridge_operator.apply_noise_interpolation(
            noisy_blocks, target_blocks, active_corruption.unsqueeze(-1)
        )

        hidden_states = self.seq_decoder(
            condition=mol_encoding,
            condition_mask=encoding_mask,
            operation_types=noisy_steps,
            template_ids=noisy_templates,
            block_features=noisy_blocks,
            sequence_mask=sequence_mask,
            time_condition=time_discrete,
        )[:, :-1] 

        labels_steps = step_types[:, 1:].contiguous()
        labels_templates = template_ids[:, 1:].contiguous()
        labels_blocks = block_fingerprints[:, 1:].contiguous()

        losses: dict[str, torch.Tensor] = {}
        auxiliary: dict[str, torch.Tensor] = {}

        losses["step_type"] = self.step_classifier.get_loss(
            hidden_states, labels_steps, None
        )

        losses["template"] = self.template_classifier.get_loss(
            hidden_states, labels_templates, 
            labels_steps == TokenType.REACTION
        )

        block_loss, block_aux = self.building_block_retriever.get_loss(
            hidden_states,
            labels_blocks,
            labels_steps == TokenType.REACTANT,
            **kwargs,
        )
        losses.update(block_loss)
        auxiliary.update(block_aux)

        corrupted_steps = (noisy_steps[:, 1:] != step_types[:, 1:])
        corrupted_templates = (noisy_templates[:, 1:] != template_ids[:, 1:])
        corrupted_blocks = (noisy_blocks[:, 1:] != block_fingerprints[:, 1:]).any(dim=-1)
        is_corrupted = corrupted_steps | corrupted_templates | corrupted_blocks

        losses["planner"] = self.noise_identifier.get_loss(
            hidden_states, time_discrete, is_corrupted
        )

        return losses, auxiliary

    def compute_loss_from_batch(
        self, 
        batch: ProjectionBatch, 
        **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        
        encoding, enc_mask = self.encode_molecule(batch)
        return self.compute_training_loss(
            mol_encoding=encoding,
            encoding_mask=enc_mask,
            step_types=batch["token_types"],
            template_ids=batch["rxn_indices"],
            block_fingerprints=batch["reactant_fps"],
            sequence_mask=batch["token_padding_mask"],
            **kwargs,
        )
    @torch.inference_mode()
    def predict_next_segment(
        self,
        mol_encoding: torch.Tensor,
        encoding_mask: Optional[torch.Tensor],
        step_types: torch.Tensor,
        template_ids: torch.Tensor,
        block_fingerprints: torch.Tensor,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        num_candidates: int = 4,
        **kwargs,
    ) -> StepPrediction:

        batch_size, current_position = step_types.shape

        segment_idx = (current_position - 1) // self.segment_length
        seg_start = segment_idx * self.segment_length + 1
        seg_end = min(
            seg_start + self.segment_length - 1, 
            current_position + self.segment_length
        )

        new_positions = seg_end - current_position + 1

        extended_steps = torch.cat([
            step_types,
            torch.zeros(
                (batch_size, new_positions), 
                dtype=step_types.dtype, 
                device=step_types.device
            )
        ], dim=1)

        extended_templates = torch.cat([
            template_ids,
            torch.zeros(
                (batch_size, new_positions), 
                dtype=template_ids.dtype, 
                device=template_ids.device
            )
        ], dim=1)

        extended_blocks = torch.cat([
            block_fingerprints,
            torch.zeros(
                (batch_size, new_positions, block_fingerprints.shape[-1]),
                dtype=block_fingerprints.dtype,
                device=block_fingerprints.device
            )
        ], dim=1)

        work_steps = extended_steps.clone()
        work_templates = extended_templates.clone()
        work_blocks = extended_blocks.clone()

        for denoise_iter in range(self.num_denoise_steps):
            iter_time = denoise_iter * torch.ones(
                (batch_size, 1)
            ).float().to(mol_encoding.device)

            hidden_states = self.seq_decoder(
                condition=mol_encoding,
                condition_mask=encoding_mask,
                operation_types=work_steps,
                template_ids=work_templates,
                block_features=work_blocks,
                sequence_mask=None,
                time_condition=iter_time,
            )

            hidden_new = hidden_states[:, current_position - 1:seg_end]

            step_logits = self.step_classifier.predict(hidden_new)
            step_probs = torch.softmax(step_logits, dim=-1)
            _, predicted_steps = step_probs.max(-1)

            template_logits = self.template_classifier.predict(hidden_new)[
                ..., :len(rxn_matrix.reactions)
            ]
            template_probs = torch.softmax(template_logits, dim=-1)
            _, predicted_templates = template_probs.max(-1)

            corruption_probs = self.noise_identifier(
                hidden_states, iter_time
            )[:, current_position - 1:seg_end]

            refresh_rate = 1.0 - (denoise_iter + 1) / self.num_denoise_steps
            refresh_mask = torch.bernoulli(
                torch.ones_like(corruption_probs) * refresh_rate * corruption_probs
            ).bool()

            work_steps[:, current_position:seg_end + 1][refresh_mask] = \
                predicted_steps[refresh_mask]
            work_templates[:, current_position:seg_end + 1][refresh_mask] = \
                predicted_templates[refresh_mask]

            block_predictions = self.building_block_retriever.predict(
                hidden_new, **kwargs
            )
            fp_dim = block_predictions.shape[-1]

            retrieved_blocks = self.building_block_retriever.retrieve_reactants(
                hidden_new, fpindex, 1, **kwargs
            )

            output_blocks = torch.empty(
                list(block_predictions.shape[:-1]) + [num_candidates, fp_dim],
                dtype=torch.float32,
                device=block_predictions.device
            )

            work_blocks[:, current_position:seg_end + 1][refresh_mask] = \
                output_blocks[:, :, 0, 0][refresh_mask]

            if (work_steps[:, current_position:seg_end + 1] == TokenType.END).any():
                break

        final_hidden = self.seq_decoder(
            condition=mol_encoding,
            condition_mask=encoding_mask,
            operation_types=work_steps[:, :current_position],
            template_ids=work_templates[:, :current_position],
            block_features=work_blocks[:, :current_position],
            sequence_mask=None,
            time_condition=torch.zeros((batch_size, 1), device=mol_encoding.device),
        )
        next_hidden = final_hidden[:, -1]

        output_step_logits = self.step_classifier.predict(next_hidden)
        output_template_logits = self.template_classifier.predict(next_hidden)[
            ..., :len(rxn_matrix.reactions)
        ]
        output_blocks = self.building_block_retriever.retrieve_reactants(
            next_hidden, fpindex, num_candidates, **kwargs
        )

        return StepPrediction(
            output_step_logits, output_template_logits, output_blocks
        )

    @torch.inference_mode()
    def generate_pathway(
        self,
        batch: ProjectionBatch,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_length: int = 24,
        **kwargs,
    ) -> PathwayResult:

        encoding, enc_mask = self.encode_molecule(batch)
        batch_size = encoding.size(0)
        fp_dim = self.building_block_retriever.fingerprint_dim

        step_types = torch.full(
            [batch_size, 1], 
            fill_value=TokenType.START,
            dtype=torch.long,
            device=encoding.device
        )
        template_ids = torch.full(
            [batch_size, 1],
            fill_value=0,
            dtype=torch.long,
            device=encoding.device
        )
        block_fingerprints = torch.zeros(
            [batch_size, 1, fp_dim],
            dtype=torch.float,
            device=encoding.device
        )

        building_blocks: list[list[Optional[Molecule]]] = [
            [None] for _ in range(batch_size)
        ]
        templates: list[list[Optional[Reaction]]] = [
            [None] for _ in range(batch_size)
        ]

        for _ in tqdm(range(max_length - 1), desc="Generating pathway"):
            prediction = self.predict_next_segment(
                mol_encoding=encoding,
                encoding_mask=enc_mask,
                step_types=step_types,
                template_ids=template_ids,
                block_fingerprints=block_fingerprints,
                rxn_matrix=rxn_matrix,
                fpindex=fpindex,
                **kwargs,
            )

            step_types = torch.cat([
                step_types,
                prediction.token_logits.argmax(dim=-1, keepdim=True)
            ], dim=-1)

            next_template = prediction.reaction_logits.argmax(dim=-1)
            template_ids = torch.cat([
                template_ids,
                next_template[..., None]
            ], dim=-1)

            for b, idx in enumerate(next_template):
                templates[b].append(rxn_matrix.reactions[int(idx.item())])

            next_fp = (
                torch.from_numpy(prediction.retrieved_reactants.fingerprint_retrieved)
                .to(block_fingerprints)
                .reshape(batch_size, -1, fp_dim)
            )[:, 0]

            block_fingerprints = torch.cat([
                block_fingerprints,
                next_fp[..., None, :]
            ], dim=-2)

            next_block = prediction.retrieved_reactants.reactants.reshape(batch_size, -1)[:, 0]
            for b, mol in enumerate(next_block):
                building_blocks[b].append(mol)

            if (step_types[:, -1] == TokenType.END).all():
                break

        return PathwayResult(
            token_types=step_types,
            rxn_indices=template_ids,
            reactant_fps=block_fingerprints,
            reactants=building_blocks,
            reactions=templates,
        )