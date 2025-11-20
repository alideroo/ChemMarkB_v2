import pickle
from typing import Any, Optional
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from ..chem.fpindex import FingerprintIndex
from ..chem.matrix import ReactantReactionMatrix
from ..data.common import ProjectionBatch
from ..models.molebridge import MoleBridge
from .utils import get_optimizer, get_scheduler, sum_weighted_losses

class MoleBridgeLightningModule(pl.LightningModule):
    def __init__(self, config, args: Optional[dict] = None):
        super().__init__()
        
        if config.version != 2:
            raise ValueError(
                f"Only config version 2 is supported, got version {config.version}"
            )
        
        self.save_hyperparameters(
            {
                "config": OmegaConf.to_container(config),
                "args": args or {},
            }
        )
        
        self.model = MoleBridge(config.model)
        
        self.rxn_matrix: Optional[ReactantReactionMatrix] = None
        self.fpindex: Optional[FingerprintIndex] = None

    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    @property
    def args(self):
        return OmegaConf.create(self.hparams.get("args", {}))

    def setup(self, stage: str) -> None:
        super().setup(stage)

        with open(self.config.chem.rxn_matrix, "rb") as f:
            self.rxn_matrix: ReactantReactionMatrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            self.fpindex: FingerprintIndex = pickle.load(f)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        
        if "scheduler" in self.config.train:
            scheduler = get_scheduler(self.config.train.scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config.train.get("monitor", "val/loss"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return optimizer

    def training_step(self, batch: ProjectionBatch, batch_idx: int) -> torch.Tensor:
        warmup = (self.current_epoch == 0)
        
        loss_dict, aux_dict = self.model.compute_loss_from_batch(batch, warmup=warmup)
        
        loss_weights = self.config.train.loss_weights
        loss_sum = sum_weighted_losses(loss_dict, loss_weights)
        
        self.log(
            "train/loss", 
            loss_sum, 
            on_step=True, 
            prog_bar=True, 
            logger=True
        )
        
        self.log(
            "train/type_loss", 
            loss_dict['step_type'], 
            on_step=True, 
            prog_bar=True
        )
        self.log(
            "train/template_loss", 
            loss_dict['template'], 
            on_step=True, 
            prog_bar=True
        )
        self.log(
            "train/fingerprint_loss", 
            loss_dict['fingerprint'], 
            on_step=True, 
            prog_bar=True
        )
        self.log(
            "train/planner_loss", 
            loss_dict['planner'],
            on_step=True, 
            prog_bar=True
        )

        self.log_dict(
            {f"train/loss_{k}": v for k, v in loss_dict.items()},
            on_step=True,
            logger=True
        )

        if "fp_select" in aux_dict:
            fp_select: torch.Tensor = aux_dict["fp_select"]
            fp_ratios: dict[str, float] = {}
            
            num_candidates = int(fp_select.max().item()) + 1
            for i in range(num_candidates):
                ratio = (fp_select == i).float().mean().nan_to_num(0.0)
                fp_ratios[f"train/fp_select_{i}"] = ratio.item()
            
            self.log_dict(fp_ratios, on_step=True, logger=True)
        
        return loss_sum

    def validation_step(
        self, 
        batch: ProjectionBatch, 
        batch_idx: int
    ) -> torch.Tensor:
        loss_dict, _ = self.model.compute_loss_from_batch(batch)
        
        loss_weights = self.config.train.get(
            "val_loss_weights", 
            self.config.train.loss_weights
        )
        loss_sum = sum_weighted_losses(loss_dict, loss_weights)

        self.log(
            "val/loss",
            loss_sum,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        self.log_dict(
            {f"val/loss_{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        
        return loss_sum

    def forward(self, batch: ProjectionBatch) -> Any:
        return self.model.generate_without_stack(
            batch,
            self.rxn_matrix,
            self.fpindex,
            max_len=self.config.get("max_generation_length", 24),
        )

    def on_train_epoch_end(self) -> None:
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", current_lr, on_epoch=True)


ChemProjectorWrapper = MoleBridgeLightningModule