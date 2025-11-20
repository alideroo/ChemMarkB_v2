import pathlib

import click

from ..chem.mol import Molecule, read_mol_file
from .distributed_executor import execute_parallel_sampling


def _parse_molecule_file(filepath: str) -> list[Molecule]:
    return list(read_mol_file(filepath))


@click.command()
@click.option(
    "--input", "-i",
    type=_parse_molecule_file,
    required=True,
    help="Input molecule file (SMILES, SDF, etc.)"
)
@click.option(
    "--output", "-o",
    type=click.Path(exists=False, path_type=pathlib.Path),
    required=True,
    help="Output CSV file"
)
@click.option(
    "--checkpoint", "-c",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Model checkpoint file"
)
@click.option(
    "--template-db",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/processed/all/matrix.pkl",
    help="Reaction template database"
)
@click.option(
    "--block-index",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/processed/all/fpindex.pkl",
    help="Building block index"
)
@click.option(
    "--beam-width",
    type=int,
    default=24,
    help="Beam search width"
)
@click.option(
    "--max-candidates",
    type=int,
    default=64,
    help="Maximum candidates to maintain"
)
@click.option(
    "--num-gpus",
    type=int,
    default=-1,
    help="Number of GPUs (-1 for all)"
)
@click.option(
    "--workers-per-gpu",
    type=int,
    default=1,
    help="Worker processes per GPU"
)
@click.option(
    "--task-queue-size",
    type=int,
    default=0,
    help="Task queue size (0 for unlimited)"
)
@click.option(
    "--result-queue-size",
    type=int,
    default=0,
    help="Result queue size (0 for unlimited)"
)
@click.option(
    "--timeout",
    type=int,
    default=180,
    help="Timeout per molecule (seconds)"
)
def main(
    input: list[Molecule],
    output: pathlib.Path,
    checkpoint: pathlib.Path,
    template_db: pathlib.Path,
    block_index: pathlib.Path,
    beam_width: int,
    max_candidates: int,
    num_gpus: int,
    workers_per_gpu: int,
    task_queue_size: int,
    result_queue_size: int,
    timeout: int,
):
    execute_parallel_sampling(
        targets=input,
        output_path=output,
        checkpoint_path=checkpoint,
        template_db_path=template_db,
        block_index_path=block_index,
        beam_width=beam_width,
        max_candidates=max_candidates,
        num_gpus=num_gpus,
        workers_per_gpu=workers_per_gpu,
        task_queue_size=task_queue_size,
        result_queue_size=result_queue_size,
        timeout_seconds=timeout,
    )


if __name__ == "__main__":
    main()