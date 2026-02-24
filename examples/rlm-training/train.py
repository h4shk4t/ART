"""Main training loop for distributed RLM training with ART LocalBackend.

Uses GRPO to train a model on R2E-Gym bug-fixing tasks. All file operations
during rollouts are local; only test execution hits the remote Docker service.

Usage:
    cd ART/examples/rlm-training
    uv run python train.py
    uv run python train.py --experiment-name "longer_context" --max-steps 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path

from datasets import load_dataset

import art
import art.dev
from art.local.backend import LocalBackend
from art.utils.iterate_dataset import iterate_dataset

from config import ExperimentConfig
from docker_client import DockerClient
from health import HealthMonitor
from prompts import r2e_rlm_system_prompt, sub_agent_system_prompt
from rewards import binary_test_reward, partial_test_reward
from rollout import Scenario, rollout
from trajectory_logger import TrajectoryLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def _health_check_loop(health: HealthMonitor, interval: int, stop_event: threading.Event) -> None:
    """Background thread that periodically runs health checks."""
    while not stop_event.is_set():
        try:
            status = health.check()
            for w in status.warnings:
                logger.warning("Health: %s", w)
            for c in status.critical:
                logger.error("Health CRITICAL: %s", c)
            if status.critical:
                logger.error(
                    "Critical health issues detected! Training continues but "
                    "consider investigating."
                )
        except Exception as e:
            logger.warning("Health check failed: %s", e)
        stop_event.wait(interval)


async def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        project=args.project,
        base_model=args.base_model,
        vllm_url=args.vllm_url,
        docker_service_url=args.docker_url,
        repo_cache_dir=args.cache_dir,
        num_epochs=args.num_epochs,
        groups_per_step=args.groups_per_step,
        rollouts_per_group=args.rollouts_per_group,
        max_concurrent=args.max_concurrent,
        max_steps=args.max_steps,
        max_context_tokens=args.max_context_tokens,
        learning_rate=args.learning_rate,
        tensor_parallel_size=args.tensor_parallel_size,
        log_dir=args.log_dir,
        log_trajectory_sample_rate=args.log_sample_rate,
        system_prompt_fn=r2e_rlm_system_prompt,
        reward_fn=binary_test_reward,
        sub_agent_system_prompt_fn=sub_agent_system_prompt,
    )

    logger.info("Experiment: %s (hash: %s)", config.experiment_name, config.config_hash())
    logger.info("Config: %s", config.to_metadata())

    # --- Infrastructure ---
    docker_client = DockerClient(config.docker_service_url)
    health = HealthMonitor.from_config(config)
    traj_logger = TrajectoryLogger.from_config(config)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    # Start background health checker
    health_stop = threading.Event()
    health_thread = threading.Thread(
        target=_health_check_loop,
        args=(health, config.health_check_interval, health_stop),
        daemon=True,
    )
    health_thread.start()

    # --- Backend & model ---
    with LocalBackend() as backend:
        model = art.TrainableModel(
            name=config.model_name,
            project=config.project,
            base_model=config.base_model,
            _internal_config=art.dev.InternalModelConfig(
                engine_args=art.dev.EngineArgs(
                    tensor_parallel_size=config.tensor_parallel_size,
                    gpu_memory_utilization=0.35,
                ),
                init_args=art.dev.InitArgs(
                    max_seq_length=16384,
                ),
            ),
        )
        await model.register(backend)
        logger.info("Model registered: %s (base: %s)", config.model_name, config.base_model)

        # --- Load dataset (filtered to cached repos only) ---
        cache_index_path = Path(config.repo_cache_dir) / "index.json"
        if cache_index_path.exists():
            with open(cache_index_path) as f:
                cache_index = json.load(f)
        else:
            cache_index = {}
            logger.warning("No cache index at %s — rollouts will fail for uncached repos", cache_index_path)

        ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
        all_scenarios = [Scenario.from_dataset_entry(dict(entry)) for entry in ds]
        scenarios = [s for s in all_scenarios if s.instance_id in cache_index]

        if not scenarios:
            logger.error("No cached scenarios found! Run cache_repos.py first.")
            return

        if args.dataset_size is not None:
            scenarios = scenarios[: args.dataset_size]
        logger.info("Loaded %d scenarios (%d cached out of %d total)", len(scenarios), len(scenarios), len(all_scenarios))

        # --- Docker health check ---
        try:
            health_resp = await docker_client.health_check()
            logger.info("Docker service OK: %s", health_resp)
        except Exception as e:
            logger.error("Docker service unreachable: %s", e)
            logger.error("Tests won't run — all rewards will be 0.0")

        # --- Resume support ---
        state = model.read_state()
        initial_step = state["step"] + 1 if state and "step" in state else 0
        if initial_step > 0:
            logger.info("Resuming from step %d", initial_step)

        # --- Training loop ---
        for batch in iterate_dataset(
            scenarios,
            groups_per_step=config.groups_per_step,
            num_epochs=config.num_epochs,
            initial_step=initial_step,
        ):
            step_t0 = time.time()

            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(
                            model,
                            scenario,
                            config,
                            docker_client,
                            health,
                            traj_logger,
                            semaphore,
                            training_step=batch.step,
                        )
                        for _ in range(config.rollouts_per_group)
                    )
                    for scenario in batch.items
                ),
                max_exceptions=0.5,
            )

            # Log pre-train rewards for debugging
            _all_rewards = [t.reward for g in groups for t in g]
            _group_rewards = [[t.reward for t in g] for g in groups]
            _dbg_path = "/home/colligo/experiments/RLM/.cursor/debug.log"
            with open(_dbg_path, "a") as _f:
                _f.write(json.dumps({"timestamp": int(time.time()*1000), "location": f"train.py:step={batch.step}", "message": "pre_train_rewards", "data": {"all_rewards": _all_rewards, "group_rewards": _group_rewards, "num_groups": len(groups), "group_sizes": [len(g) for g in groups]}}) + "\n")

            zero_var_groups = 0
            for g in groups:
                rewards = [t.reward for t in g]
                if not rewards: continue
                std = (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / max(1, len(rewards)))**0.5
                if std < 1e-8:
                    zero_var_groups += 1
            
            if zero_var_groups > 0:
                logger.warning("Found %d/%d groups with zero reward variance.", zero_var_groups, len(groups))
                if zero_var_groups == len(groups):
                    logger.warning("All groups have zero variance! Injecting noise to prevent NaN loss.")
                    for g in groups:
                        for t in g:
                            t.reward += random.uniform(-1e-4, 1e-4)
            # Free leftover inference memory before training
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log GPU memory stats for debugging
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                    logger.info(
                        "GPU %d pre-train memory: allocated=%.1fGB reserved=%.1fGB total=%.1fGB free=%.1fGB",
                        gpu_id, allocated, reserved, total, total - reserved,
                    )

            try:
                result = await backend.train(
                    model, groups, learning_rate=config.learning_rate
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_str = str(e)
                if "out of memory" in err_str.lower() or "CUDA" in err_str:
                    logger.error(
                        "CUDA OOM during training step %d: %s", batch.step, err_str
                    )
                    # Log GPU memory at crash time
                    for gpu_id in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                        reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                        logger.error(
                            "GPU %d at OOM: allocated=%.1fGB reserved=%.1fGB",
                            gpu_id, allocated, reserved,
                        )
                    torch.cuda.empty_cache()
                    logger.error(
                        "vLLM workers are likely stuck in sleep state after OOM. "
                        "Training must restart. Consider reducing max_seq_length "
                        "or setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                    )
                    sys.exit(1)
                raise  # Re-raise non-OOM RuntimeErrors

            # Aggregate metrics
            total_traj = max(1, sum(len(g) for g in groups))
            avg_reward = sum(t.reward for g in groups for t in g) / total_traj
            avg_sub_queries = (
                sum(float(t.metrics.get("num_sub_queries", 0)) for g in groups for t in g)
                / total_traj
            )
            avg_finished = (
                sum(1.0 if t.metrics.get("finished") else 0.0 for g in groups for t in g)
                / total_traj
            )
            avg_steps = (
                sum(float(t.metrics.get("num_steps", 0)) for g in groups for t in g)
                / total_traj
            )

            step_elapsed = time.time() - step_t0
            metrics = {
                **result.metrics,
                "avg_reward": avg_reward,
                "avg_sub_queries": avg_sub_queries,
                "avg_finished": avg_finished,
                "avg_steps": avg_steps,
                "step_time": step_elapsed,
            }

            await model.log(groups, split="train", metrics=metrics, step=result.step)
            model.merge_state({"step": batch.step})

            logger.info(
                "Step %d (%.0fs): reward=%.3f, finished=%.2f, steps=%.1f, "
                "sub_queries=%.2f, loss=%.4f",
                batch.step,
                step_elapsed,
                avg_reward,
                avg_finished,
                avg_steps,
                avg_sub_queries,
                result.metrics.get("loss", float("nan")),
            )

    health_stop.set()
    health_thread.join(timeout=5)
    logger.info("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed RLM GRPO Training")

    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--model-name", default="r2e-rlm-qwen3-14b")
    parser.add_argument("--project", default="rlm-training")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")

    parser.add_argument("--vllm-url", default="http://localhost:8001/v1")
    parser.add_argument("--docker-url", default="http://localhost:8000")
    parser.add_argument("--cache-dir", default="./repo-cache")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)

    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--groups-per-step", type=int, default=3)
    parser.add_argument("--rollouts-per-group", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-context-tokens", type=int, default=28000)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--dataset-size", type=int, default=None)

    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-sample-rate", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
