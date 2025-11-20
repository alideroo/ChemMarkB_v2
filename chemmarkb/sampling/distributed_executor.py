import multiprocessing as mp
import os
import pathlib
import pickle
import subprocess
import time
from multiprocessing import synchronize as sync
from typing import TypeAlias

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from ..chem.fpindex import FingerprintIndex
from ..chem.matrix import ReactantReactionMatrix
from ..chem.mol import FingerprintOption, Molecule
from ..models.molebridge import MoleBridge
from .tree_explorer import BeamSearchExplorer, ExecutionTimer

TaskQueue: TypeAlias = "mp.JoinableQueue[Molecule | None]"
ResultQueue: TypeAlias = "mp.Queue[tuple[Molecule, pd.DataFrame, float]]"


class SamplingWorker(mp.Process):
    def __init__(
        self,
        block_index_path: pathlib.Path,
        template_db_path: pathlib.Path,
        checkpoint_path: pathlib.Path,
        task_queue: TaskQueue,
        result_queue: ResultQueue,
        device_id: str,
        device_lock: sync.Lock,
        explorer_config: dict | None = None,
        max_iterations: int = 12,
        max_solutions: int = 100,
        timeout_seconds: int = 120,
    ):
        super().__init__()
        self._block_index_path = block_index_path
        self._template_db_path = template_db_path
        self._checkpoint_path = checkpoint_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._device_id = device_id
        self._device_lock = device_lock
        self._explorer_config = explorer_config or {}
        self._max_iterations = max_iterations
        self._max_solutions = max_solutions
        self._timeout_seconds = timeout_seconds

    def run(self) -> None:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = self._device_id
        
        self._block_index = pickle.load(open(self._block_index_path, "rb"))
        self._template_db = pickle.load(open(self._template_db_path, "rb"))
        
        checkpoint = torch.load(
            self._checkpoint_path, map_location="cpu", weights_only=False
        )
        config = OmegaConf.create(checkpoint["hyper_parameters"]["config"])
        network = MoleBridge(config.model).to("cuda")
        network.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items()})
        network.eval()
        self._network = network
        
        try:
            while True:
                task = self._task_queue.get()
                if task is None:
                    self._task_queue.task_done()
                    break
                
                start_time = time.perf_counter()
                result_table = self._execute_sampling(task)
                elapsed_time = time.perf_counter() - start_time
                
                self._task_queue.task_done()
                self._result_queue.put((task, result_table, elapsed_time))
                
                if len(result_table) == 0:
                    print(f"{self.name}: No results for {task.smiles}")
                else:
                    best_score = result_table["score"].max()
                    print(f"{self.name}: {best_score:.3f} {task.smiles}")
        
        except KeyboardInterrupt:
            print(f"{self.name}: Interrupted by user")

    def _execute_sampling(self, target: Molecule) -> pd.DataFrame:
        explorer = BeamSearchExplorer(
            block_index=self._block_index,
            template_db=self._template_db,
            target=target,
            network=self._network,
            **self._explorer_config,
        )
        
        timer = ExecutionTimer(self._timeout_seconds)
        
        for _ in range(self._max_iterations):
            explorer.expand_once(
                device_lock=self._device_lock, 
                show_progress=False, 
                timer=timer
            )
            
            solutions = list(explorer.get_solutions())
            if solutions:
                best_similarity = max(
                    [
                        sol.product.sim(
                            target, FingerprintOption.morgan_for_tanimoto_similarity()
                        )
                        for sol in solutions
                    ]
                )
                if best_similarity == 1.0:
                    break
        
        result_table = explorer.create_result_table()[:self._max_solutions]
        return result_table


class WorkerCluster:
    
    def __init__(
        self,
        device_ids: list[int | str],
        workers_per_device: int,
        task_queue_size: int,
        result_queue_size: int,
        **worker_kwargs,
    ) -> None:
        self._task_queue = mp.JoinableQueue(task_queue_size)
        self._result_queue = mp.Queue(result_queue_size)
        self._device_ids = [str(d) for d in device_ids]
        self._device_locks = [mp.Lock() for _ in device_ids]
        
        num_devices = len(device_ids)
        total_workers = workers_per_device * num_devices
        
        self._workers = [
            SamplingWorker(
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                device_id=self._device_ids[i % num_devices],
                device_lock=self._device_locks[i % num_devices],
                **worker_kwargs,
            )
            for i in range(total_workers)
        ]
        
        for worker in self._workers:
            worker.start()

    def add_task(
        self, 
        molecule: Molecule, 
        blocking: bool = True, 
        timeout: float | None = None
    ):
        self._task_queue.put(molecule, block=blocking, timeout=timeout)

    def get_result(
        self, 
        blocking: bool = True, 
        timeout: float | None = None
    ):
        return self._result_queue.get(block=blocking, timeout=timeout)

    def force_shutdown(self):
        for worker in self._workers:
            worker.kill()
        self._result_queue.close()
        self._task_queue.close()

    def graceful_shutdown(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for worker in tqdm(self._workers, desc="Shutting down workers"):
            worker.terminate()
        self._result_queue.close()
        self._task_queue.close()


def _detect_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return len([x.strip() for x in cuda_visible.split(",") if x.strip()])
    else:
        return int(
            subprocess.check_output(
                "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
                shell=True,
                text=True
            ).strip()
        )


def _get_available_gpus() -> list[str]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        return [str(i) for i in range(_detect_gpu_count())]


def execute_parallel_sampling(
    targets: list[Molecule],
    output_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    template_db_path: pathlib.Path,
    block_index_path: pathlib.Path,
    beam_width: int = 24,
    max_candidates: int = 64,
    num_gpus: int = -1,
    workers_per_gpu: int = 2,
    task_queue_size: int = 0,
    result_queue_size: int = 0,
    timeout_seconds: int = 180,
) -> None:
    if num_gpus <= 0:
        available_gpus = _get_available_gpus()
        num_gpus = len(available_gpus)
    else:
        available_gpus = _get_available_gpus()[:num_gpus]
    
    cluster = WorkerCluster(
        device_ids=available_gpus,
        workers_per_device=workers_per_gpu,
        task_queue_size=task_queue_size,
        result_queue_size=result_queue_size,
        block_index_path=block_index_path,
        template_db_path=template_db_path,
        checkpoint_path=checkpoint_path,
        explorer_config={
            "beam_width": beam_width,
            "max_candidates": max_candidates,
        },
        timeout_seconds=timeout_seconds,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_tasks = len(targets)
    for target in targets:
        cluster.add_task(target)
    
    execution_times = []
    all_results: list[pd.DataFrame] = []
    
    with open(output_path, "w") as output_file:
        for _ in tqdm(range(total_tasks), desc="Processing"):
            _, result_df, elapsed = cluster.get_result()
            execution_times.append(elapsed)
            
            if len(result_df) == 0:
                continue
            
            result_df.to_csv(
                output_file, 
                float_format="%.3f", 
                index=False, 
                header=output_file.tell() == 0
            )
            all_results.append(result_df)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    aggregated_stats = combined_results.loc[
        combined_results.groupby("target").idxmax()["score"]
    ].select_dtypes(include="number").sum() / total_tasks
    
    metric_descriptions = pd.Series({
        "score": "Tanimoto similarity",
        "scf_sim": "Scaffold similarity",
        "pharm2d_sim": "Gobbi2D Pharmacophore similarity",
        "rdkit_sim": "RDKit fingerprint similarity",
        "num_steps": "Average synthesis steps",
    })
    
    print(pd.DataFrame({
        "result": aggregated_stats, 
        "description": metric_descriptions
    }))
    
    success_count = len(combined_results["target"].unique())
    print(f"Success rate: {success_count}/{total_tasks} = {success_count / total_tasks:.3f}")
    
    exact_matches: set[str] = set()
    for _, row in combined_results.iterrows():
        if row["score"] == 1.0:
            target_mol = Molecule(row["target"])
            result_mol = Molecule(row["smiles"])
            if result_mol.csmiles == target_mol.csmiles:
                exact_matches.add(row["target"])
    
    match_count = len(exact_matches)
    print(f"Exact match rate: {match_count}/{total_tasks} = {match_count / total_tasks:.3f}")
    print(f"Average time: {sum(execution_times)/len(execution_times):.2f}s")
    print(f"Max: {max(execution_times):.2f}s | Min: {min(execution_times):.2f}s")
    
    cluster.graceful_shutdown()