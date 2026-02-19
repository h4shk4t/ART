"""Main training loop for R2E-Gym RLM-style RL training with ART.

Uses GRPO via ServerlessBackend to train a model on R2E-Gym repo bug-fixing
tasks with binary test-pass reward.

Usage:
    cd examples/r2e-gym-rlm
    uv run python train.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from datasets import load_dataset

import art
from art.utils.iterate_dataset import iterate_dataset

from rollout import R2EScenario as LegacyScenario, rollout as legacy_rollout  # type: ignore[import-not-found]
from rollout_rlm_env import R2ERLMScenario, rollout as rlm_rollout  # type: ignore[import-not-found]


@dataclass
class Config:
    model_name: str = "r2e-rlm-qwen3-14b"
    project: str = "r2e-gym-rlm"
    base_model: str = "OpenPipe/Qwen3-14B-Instruct"

    num_epochs: int = 3
    groups_per_step: int = 8       # scenarios per training step
    rollouts_per_group: int = 4    # rollouts per scenario (GRPO)
    learning_rate: float = 5e-6
    max_concurrent: int = 8        # semaphore limit for Docker containers

    max_steps: int = 15            # agent steps per rollout
    dataset_size: int | None = None  # full dataset
    use_rlm_env: bool = True


async def main() -> None:
    config = Config()

    # --- Backend & model setup ---
    backend = art.ServerlessBackend()
    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
    )
    await model.register(backend)

    # --- Load dataset ---
    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
    if config.use_rlm_env:
        scenarios = [
            R2ERLMScenario(ds=entry, max_steps=config.max_steps)
            for entry in ds
        ]
        rollout_fn = rlm_rollout
    else:
        scenarios = [
            LegacyScenario(ds=entry, max_steps=config.max_steps)
            for entry in ds
        ]
        rollout_fn = legacy_rollout
    if config.dataset_size is not None:
        scenarios = scenarios[: config.dataset_size]

    print(f"Loaded {len(scenarios)} scenarios")

    # --- Concurrency limiter ---
    semaphore = asyncio.Semaphore(config.max_concurrent)

    # --- Resume support ---
    state = model.read_state()
    initial_step = state["step"] + 1 if state and "step" in state else 0

    # --- Training loop ---
    for batch in iterate_dataset(
        scenarios,
        groups_per_step=config.groups_per_step,
        num_epochs=config.num_epochs,
        initial_step=initial_step,
    ):
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(  # type: ignore[arg-type]
                    rollout_fn(model, scenario, semaphore)
                    for _ in range(config.rollouts_per_group)
                )
                for scenario in batch.items
            ),
            max_exceptions=0.5,
        )

        result = await backend.train(
            model, groups, learning_rate=config.learning_rate
        )

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

        metrics = {
            **result.metrics,
            "avg_reward": avg_reward,
            "avg_sub_queries": avg_sub_queries,
            "avg_finished": avg_finished,
            "avg_steps": avg_steps,
        }

        await model.log(groups, split="train", metrics=metrics, step=result.step)
        model.merge_state({"step": batch.step})

        print(
            f"Step {batch.step}: "
            f"reward={avg_reward:.3f}, "
            f"sub_queries={avg_sub_queries:.2f}, "
            f"finished={avg_finished:.2f}, "
            f"steps={avg_steps:.1f}, "
            f"train_loss={result.metrics.get('loss', float('nan')):.4f}"
        )

    print("Training complete.")


if __name__ == "__main__":
    asyncio.run(main())
