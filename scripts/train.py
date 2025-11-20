import os
import sys
from pathlib import Path

ORIG_CWD = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ORIG_CWD))

import click
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers, strategies
from omegaconf import OmegaConf

from molebridge.data.projection_dataset import ProjectionDataModule
from molebridge.training import MoleBridgeLightningModule
from molebridge.utils.experiment import (
    get_config_name,
    get_experiment_name,
    get_experiment_version,
)
from molebridge.utils.vc import get_vc_info

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


def parse_device_list(device_string: str) -> list[int]:
    try:
        return [int(d.strip()) for d in device_string.split(",")]
    except ValueError:
        raise ValueError(
            f"Invalid device string: '{device_string}'. "
            "Expected format: '0,1,2' or '0'"
        )


def validate_batch_config(batch_size: int, num_devices: int) -> int:
    if batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by "
            f"number of devices ({num_devices})"
        )
    return batch_size // num_devices


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True),
    metavar="CONFIG_FILE"
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Debug mode (allows uncommitted code changes)"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=100,
    help="Total batch size across all devices"
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of data loading workers per device"
)
@click.option(
    "--devices",
    type=str,
    default="0,1",
    help="GPU device IDs (comma-separated, e.g., '0,1,2,3')"
)
@click.option(
    "--num-nodes",
    type=int,
    default=int(os.environ.get("NUM_NODES", 1)),
    help="Number of nodes for distributed training"
)
@click.option(
    "--sanity-checks",
    type=int,
    default=2,
    help="Number of validation sanity check steps"
)
@click.option(
    "--log-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default="./logs",
    help="Directory for logs and checkpoints"
)
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Checkpoint path to resume training from"
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="Custom experiment name (overrides auto-generated)"
)
def main(
    config_path: str,
    seed: int,
    debug: bool,
    batch_size: int,
    num_workers: int,
    devices: str,
    num_nodes: int,
    sanity_checks: int,
    log_dir: str,
    resume: str | None,
    experiment_name: str | None,
):

    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed, workers=True)
    
    click.echo(f"Working directory: {Path.cwd()}")
    
    device_ids = parse_device_list(devices)
    num_devices = len(device_ids)
    
    click.echo(f"Using {num_devices} GPU(s): {device_ids}")
    
    batch_size_per_device = validate_batch_config(batch_size, num_devices)
    click.echo(
        f"Batch size: {batch_size} total "
        f"({batch_size_per_device} per device)"
    )
    
    config = OmegaConf.load(config_path)
    config_name = get_config_name(config_path)
    click.echo(f"Loaded config: {config_name}")
    
    vc_info = get_vc_info()
    vc_info.disallow_changes()
    
    click.echo(f"Code version: {vc_info.display_version}")
    if vc_info.branch:
        click.echo(f"Git branch: {vc_info.branch}")
    
    if experiment_name is None:
        exp_name = get_experiment_name(
            config_name,
            vc_info.display_version,
            vc_info.committed_at
        )
    else:
        exp_name = experiment_name
    
    exp_version = get_experiment_version()
    click.echo(f"Experiment: {exp_name}/{exp_version}")
    
    click.echo("Initializing data module...")
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size_per_device,
        num_workers=num_workers,
        **config.data,
    )
    
    click.echo("Initializing model...")
    model = MoleBridgeLightningModule(
        config,
        args={
            "seed": seed,
            "batch_size": batch_size,
            "num_devices": num_devices,
            "debug": debug,
            "config_path": config_path,
        }
    )
    
    callback_list = [
        callbacks.ModelCheckpoint(
            dirpath=Path(log_dir) / exp_name / exp_version / "checkpoints",
            filename="epoch{epoch:03d}-loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=5,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        callbacks.LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
    ]
    
    if hasattr(config.train, "early_stopping"):
        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val/loss",
                patience=config.train.early_stopping.patience,
                mode="min",
                verbose=True,
            )
        )
        click.echo(
            f"Early stopping enabled "
            f"(patience={config.train.early_stopping.patience})"
        )
    
    logger_list = [
        loggers.TensorBoardLogger(
            save_dir=log_dir,
            name=exp_name,
            version=exp_version,
            default_hp_metric=False,
        ),
    ]
    
    if hasattr(config, "wandb") and config.wandb.get("enabled", False):
        logger_list.append(
            loggers.WandbLogger(
                project=config.wandb.project,
                name=f"{exp_name}/{exp_version}",
                save_dir=log_dir,
                log_model=False,
            )
        )
        click.echo(f"WandB logging enabled (project={config.wandb.project})")
    
    if num_devices > 1:
        training_strategy = strategies.DDPStrategy(
            static_graph=True,
            find_unused_parameters=False,
        )
        click.echo("Using DDP strategy for multi-GPU training")
    else:
        training_strategy = "auto"
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=device_ids,
        num_nodes=num_nodes,
        strategy=training_strategy,
        
        max_steps=config.train.max_iterations,
        gradient_clip_val=config.train.get("max_grad_norm", 1.0),
        accumulate_grad_batches=config.train.get("accumulate_grad_batches", 1),
        precision=config.train.get("precision", 32),
        
        val_check_interval=config.train.validation_frequency,
        num_sanity_val_steps=sanity_checks,
        limit_val_batches=config.train.get("limit_val_batches", 1.0),
        
        logger=logger_list,
        callbacks=callback_list,
        log_every_n_steps=config.train.get("log_every_n_steps", 50),
        
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )

    click.echo("\n" + "=" * 60)
    click.echo("Training Configuration Summary")
    click.echo("=" * 60)
    click.echo(f"Config: {config_name}")
    click.echo(f"Experiment: {exp_name}/{exp_version}")
    click.echo(f"Devices: {num_devices} GPU(s) - {device_ids}")
    click.echo(f"Batch size: {batch_size} ({batch_size_per_device}/device)")
    click.echo(f"Max steps: {config.train.max_iterations}")
    click.echo(f"Validation frequency: every {config.train.validation_frequency} steps")
    click.echo(f"Checkpoints: {Path(log_dir) / exp_name / exp_version / 'checkpoints'}")
    if resume:
        click.echo(f"Resuming from: {resume}")
    click.echo("=" * 60 + "\n")
    
    try:
        click.echo("Starting training...")
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=resume
        )
    except KeyboardInterrupt:
        click.secho("Training interrupted by user!", fg='yellow', bold=True)
        click.echo(f"Last checkpoint: {trainer.checkpoint_callback.last_model_path}")
    except Exception as e:
        click.secho(f"Training failed with error: {e}", fg='red', bold=True)
        raise
    
    click.secho("\n" + "=" * 60, fg='green')
    click.secho("Training completed successfully!", fg='green', bold=True)
    click.secho("=" * 60, fg='green')
    
    if trainer.checkpoint_callback:
        click.echo(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        click.echo(f"Last checkpoint: {trainer.checkpoint_callback.last_model_path}")
        click.echo(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    click.echo(f"\nView logs: tensorboard --logdir {log_dir}")
    click.echo("=" * 60 + "\n")


if __name__ == "__main__":
    main()