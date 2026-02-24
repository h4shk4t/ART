"""Dry-run training benchmark: real rollouts, random rewards, no weight updates.

Runs N concurrent rollouts against cached repos, assigns random rewards,
and prints timing / throughput / resource stats.

Two modes:
  --external (default): Uses an external vLLM server you started manually.
  --use-backend:        Uses ART LocalBackend which manages its own vLLM
                        (with tensor parallelism). This benchmarks the real
                        training infrastructure path (sleep/wake cycle).

Usage:
    # External vLLM (you start it yourself on port 8001)
    uv run python debug_train.py --num-rollouts 16 --max-concurrent 8

    # LocalBackend-managed vLLM with TP=8 (kills any external vLLM)
    uv run python debug_train.py --use-backend --tp 8 --num-rollouts 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from openai.types.chat.chat_completion import Choice

import art
import art.dev

from config import ExperimentConfig
from docker_client import DockerClient, get_modified_files
from health import HealthMonitor
from prompts import r2e_rlm_system_prompt, sub_agent_system_prompt
from repl import LocalREPL, extract_python_code
from rewards import binary_test_reward
from rollout import Scenario, _trim_context, _to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug_train")

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class RolloutStats:
    instance_id: str
    rollout_idx: int
    wall_time: float = 0.0
    num_steps: int = 0
    finished: bool = False
    num_sub_queries: int = 0
    reward: float = 0.0
    llm_time: float = 0.0
    repl_time: float = 0.0
    copy_time: float = 0.0
    error: str | None = None


def _copy_from_cache(instance_id: str, cache_dir: str) -> tuple[Path, Path]:
    index_path = Path(cache_dir) / "index.json"
    with open(index_path) as f:
        index = json.load(f)
    entry = index.get(instance_id)
    if not entry:
        raise FileNotFoundError(f"Instance {instance_id!r} not in cache")
    cache_src = Path(cache_dir) / instance_id
    if not cache_src.exists():
        fallback = entry["cache_path"] if isinstance(entry, dict) else entry
        cache_src = Path(fallback)
    _ignore = shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc")
    work_dir = Path(tempfile.mkdtemp(prefix="rlm_bench_"))
    shutil.copytree(str(cache_src), str(work_dir), dirs_exist_ok=True, symlinks=True, ignore=_ignore)
    return work_dir, cache_src


async def bench_rollout(
    rollout_idx: int,
    model: art.Model,
    scenario: Scenario,
    config: ExperimentConfig,
    health: HealthMonitor,
    semaphore: asyncio.Semaphore,
    docker_client: DockerClient | None = None,
) -> RolloutStats:
    stats = RolloutStats(instance_id=scenario.instance_id, rollout_idx=rollout_idx)
    t0 = time.time()
    work_dir: Path | None = None

    try:
        async with semaphore:
            copy_t0 = time.time()
            work_dir, _cache_src = await asyncio.to_thread(
                _copy_from_cache, scenario.instance_id, config.repo_cache_dir,
            )
            stats.copy_time = time.time() - copy_t0
            health.track_rollout(work_dir)

            traj = art.Trajectory(
                messages_and_choices=[],
                reward=0.0,
                metadata={"instance_id": scenario.instance_id},
                metrics={"num_steps": 0, "finished": False, "num_sub_queries": 0},
            )

            system_prompt = config.system_prompt_fn() if config.system_prompt_fn else r2e_rlm_system_prompt()
            sub_prompt = config.sub_agent_system_prompt_fn() if config.sub_agent_system_prompt_fn else sub_agent_system_prompt()

            repl = LocalREPL(
                work_dir=work_dir,
                model=model,
                trajectory=traj,
                health=health,
                sub_agent_system_prompt=sub_prompt,
                max_output_chars=config.max_output_chars,
                max_sub_agent_steps=config.max_sub_agent_steps,
                max_completion_tokens=config.max_completion_tokens,
                max_depth=config.max_sub_agent_depth,
            )

            traj.messages_and_choices = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Bug report:\n\n{scenario.problem_statement}\n\nFix this bug."},
            ]

            consecutive_no_code = 0

            for step in range(config.max_steps):
                messages_for_api = _trim_context(
                    [_to_dict(m) for m in traj.messages_and_choices],
                    budget=config.max_context_tokens,
                    chars_per_token=config.chars_per_token,
                )

                llm_t0 = time.time()
                try:
                    client = model.openai_client()
                    response = await client.chat.completions.create(
                        model=model.get_inference_name(),
                        messages=messages_for_api,
                        temperature=1.0,
                        max_completion_tokens=config.max_completion_tokens,
                    )
                except openai.BadRequestError as e:
                    logger.warning("Rollout %d step %d API error: %s", rollout_idx, step, e)
                    traj.messages_and_choices.append({
                        "role": "user",
                        "content": "[Context too long — earlier turns were dropped.]",
                    })
                    stats.llm_time += time.time() - llm_t0
                    continue
                except (openai.InternalServerError, openai.APIConnectionError) as e:
                    logger.warning("Rollout %d step %d server error (retrying): %s", rollout_idx, step, e)
                    stats.llm_time += time.time() - llm_t0
                    await asyncio.sleep(2)
                    continue
                stats.llm_time += time.time() - llm_t0

                choice = response.choices[0]
                traj.messages_and_choices.append(choice)
                assistant_text = choice.message.content or ""
                code = extract_python_code(assistant_text)

                if code is None:
                    consecutive_no_code += 1
                    nudge = (
                        "You must write Python code in a ```python or ```repl block. "
                        "If you're done, write: ```python\nfinish()\n```"
                        if consecutive_no_code >= 2 else
                        "Write Python code in a ```python or ```repl block to proceed."
                    )
                    traj.messages_and_choices.append({"role": "user", "content": nudge})
                    continue

                consecutive_no_code = 0
                repl_t0 = time.time()
                exec_result = await asyncio.to_thread(repl.execute, code)
                stats.repl_time += time.time() - repl_t0

                stats.num_steps = step + 1
                stats.num_sub_queries = len(traj.additional_histories)

                if repl.is_finished():
                    stats.finished = True
                    traj.messages_and_choices.append({
                        "role": "user",
                        "content": f"Output:\n```\n{exec_result.stdout}\n```\nAgent called finish().",
                    })
                    break

                traj.messages_and_choices.append({
                    "role": "user",
                    "content": f"Output:\n```\n{exec_result.stdout}\n```",
                })

            if docker_client is not None and work_dir is not None:
                modified = await asyncio.to_thread(
                    get_modified_files, work_dir, _cache_src,
                )
                if modified:
                    try:
                        test_result = await docker_client.run_tests(
                            docker_image=scenario.docker_image,
                            modified_files=modified,
                        )
                        test_output = test_result.get("test_output", "")
                        entry = {"expected_output_json": scenario.expected_output_json}
                        stats.reward = binary_test_reward(test_output, entry)
                        logger.info(
                            "Rollout %d: Docker reward=%.2f (exit=%s, %.1fs)",
                            rollout_idx, stats.reward,
                            test_result.get("exit_code"),
                            test_result.get("elapsed_seconds", 0),
                        )
                    except Exception as e:
                        logger.warning("Rollout %d: Docker test failed: %s", rollout_idx, e)
                        stats.reward = 0.0
                else:
                    logger.info("Rollout %d: no modified files, reward=0.0", rollout_idx)
                    stats.reward = 0.0
            else:
                stats.reward = random.uniform(0.0, 1.0)

    except Exception as e:
        stats.error = str(e)
        logger.exception("Rollout %d failed: %s", rollout_idx, e)
    finally:
        stats.wall_time = time.time() - t0
        if work_dir is not None:
            health.unregister_rollout(work_dir)
            try:
                await asyncio.to_thread(shutil.rmtree, str(work_dir), True)
            except Exception:
                pass

    return stats


def _print_stats(all_stats: list[RolloutStats], total_wall: float) -> None:
    lines: list[str] = []
    def p(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    p(f"\n{'='*70}")
    p(f"  Benchmark Results  ({len(all_stats)} rollouts, {total_wall:.1f}s wall clock)")
    p(f"{'='*70}\n")

    errors = [s for s in all_stats if s.error]
    ok = [s for s in all_stats if not s.error]

    if not ok:
        p("All rollouts failed!")
        for s in errors:
            p(f"  Rollout {s.rollout_idx}: {s.error}")
        _save_results(lines)
        return

    p("Per-rollout breakdown:")
    p(f"  {'#':>3}  {'Instance':>30}  {'Wall':>6}  {'LLM':>6}  {'REPL':>6}  {'Copy':>6}  {'Steps':>5}  {'Fin':>3}  {'Reward':>6}")
    p(f"  {'---':>3}  {'---':>30}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>6}  {'---':>5}  {'---':>3}  {'---':>6}")
    for s in sorted(ok, key=lambda x: x.rollout_idx):
        fin = "Y" if s.finished else "N"
        p(
            f"  {s.rollout_idx:>3}  {s.instance_id:>30}  "
            f"{s.wall_time:>5.1f}s  {s.llm_time:>5.1f}s  {s.repl_time:>5.1f}s  "
            f"{s.copy_time:>5.1f}s  {s.num_steps:>5}  {fin:>3}  {s.reward:>5.2f}"
        )

    wall_times = [s.wall_time for s in ok]
    llm_times = [s.llm_time for s in ok]
    repl_times = [s.repl_time for s in ok]
    copy_times = [s.copy_time for s in ok]
    steps = [s.num_steps for s in ok]

    p(f"\nAggregates:")
    p(f"  Rollouts OK / Failed:      {len(ok)} / {len(errors)}")
    p(f"  Finished:                  {sum(1 for s in ok if s.finished)} / {len(ok)}")
    p(f"  Total wall clock:          {total_wall:.1f}s")
    p(f"  Avg wall / rollout:        {sum(wall_times)/len(ok):.1f}s")
    p(f"  Avg LLM time / rollout:    {sum(llm_times)/len(ok):.1f}s")
    p(f"  Avg REPL time / rollout:   {sum(repl_times)/len(ok):.1f}s")
    p(f"  Avg copy time / rollout:   {sum(copy_times)/len(ok):.1f}s")
    p(f"  Avg steps / rollout:       {sum(steps)/len(ok):.1f}")
    p(f"  Min / Max wall:            {min(wall_times):.1f}s / {max(wall_times):.1f}s")
    p(f"  Throughput:                {len(ok)/total_wall:.2f} rollouts/s")
    p(f"  Effective parallelism:     {sum(wall_times)/total_wall:.1f}x")

    if errors:
        p(f"\nErrors ({len(errors)}):")
        for s in errors:
            p(f"  Rollout {s.rollout_idx}: {s.error}")

    _save_results(lines)


def _save_results(lines: list[str]) -> None:
    results_path = Path("benchmark_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {results_path.resolve()}", flush=True)

    if errors:
        print(f"\n{RED}Errors ({len(errors)}):{RESET}")
        for s in errors:
            print(f"  Rollout {s.rollout_idx}: {s.error}")


def _load_scenarios(cache_dir: str) -> tuple[dict[str, Any], dict[str, Scenario]]:
    """Load cache index and build Scenario objects for cached instances."""
    index_path = Path(cache_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Cache index not found at {index_path}")
    with open(index_path) as f:
        cache_index = json.load(f)
    if not cache_index:
        raise ValueError("No cached instances found")

    from datasets import load_dataset
    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")

    scenario_map: dict[str, Scenario] = {}
    for row in ds:
        repo = row.get("repo_name", "unknown")
        commit = row.get("commit_hash", "unknown")
        iid = f"{repo}__{commit[:12]}"
        if iid in cache_index:
            scenario_map[iid] = Scenario.from_dataset_entry(dict(row))

    if not scenario_map:
        raise ValueError("No cached instances match dataset entries")
    return cache_index, scenario_map


async def _run_benchmark(
    model: art.Model,
    scenario_map: dict[str, Scenario],
    config: ExperimentConfig,
    args: argparse.Namespace,
    docker_client: DockerClient | None = None,
) -> None:
    """Run the benchmark rollouts and print results."""
    health = HealthMonitor.from_config(config)
    semaphore = asyncio.Semaphore(config.max_concurrent)
    available = list(scenario_map.keys())

    tasks = []
    total_t0 = time.time()
    for i in range(args.num_rollouts):
        iid = available[i % len(available)]
        tasks.append(
            bench_rollout(
                i, model, scenario_map[iid], config, health, semaphore,
                docker_client=docker_client,
            )
        )

    print(f"Launching {len(tasks)} rollouts (max {args.max_concurrent} concurrent)...\n", flush=True)
    all_stats = await asyncio.gather(*tasks)
    total_wall = time.time() - total_t0
    _print_stats(list(all_stats), total_wall)
    sys.stdout.flush()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark rollout throughput")
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--cache-dir", default="./repo-cache")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--external", action="store_true", default=True,
        help="Use an external vLLM server (default)",
    )
    mode.add_argument(
        "--use-backend", action="store_true",
        help="Use ART LocalBackend (manages its own vLLM with TP)",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1",
                        help="vLLM URL (only for --external mode)")
    parser.add_argument("--tp", type=int, default=8,
                        help="Tensor parallel size (only for --use-backend mode)")
    parser.add_argument("--docker-url", default=None,
                        help="Docker service URL for real test rewards (e.g. https://xyz.ngrok-free.app)")
    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="bench",
        model_name="rlm-bench",
        base_model=args.model,
        vllm_url=args.vllm_url,
        docker_service_url=args.docker_url or "",
        repo_cache_dir=args.cache_dir,
        max_steps=args.max_steps,
        max_completion_tokens=args.max_completion_tokens,
        max_concurrent=args.max_concurrent,
        tensor_parallel_size=args.tp,
        system_prompt_fn=r2e_rlm_system_prompt,
        sub_agent_system_prompt_fn=sub_agent_system_prompt,
    )

    docker_client: DockerClient | None = None
    if args.docker_url:
        docker_client = DockerClient(args.docker_url)
        try:
            health = await docker_client.health_check()
            print(f"Docker service OK: {health}", flush=True)
        except Exception as e:
            print(f"{RED}Docker service unreachable: {e}{RESET}", flush=True)
            print(f"{YELLOW}Falling back to random rewards{RESET}", flush=True)
            docker_client = None

    _cache_index, scenario_map = _load_scenarios(args.cache_dir)
    available = list(scenario_map.keys())

    print(f"\n{BOLD}Benchmark config:{RESET}")
    print(f"  Mode:           {'LocalBackend (TP=' + str(args.tp) + ')' if args.use_backend else 'External vLLM'}")
    print(f"  Rewards:        {'Docker (real)' if docker_client else 'Random'}")
    print(f"  Rollouts:       {args.num_rollouts}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Max steps:      {args.max_steps}")
    print(f"  Model:          {args.model}")
    if not args.use_backend:
        print(f"  vLLM URL:       {args.vllm_url}")
    else:
        print(f"  TP size:        {args.tp}")
    if docker_client:
        print(f"  Docker URL:     {args.docker_url}")
    print(f"  Cached repos:   {len(available)} ({', '.join(available)})")
    print(flush=True)

    if args.use_backend:
        from art.local.backend import LocalBackend

        with LocalBackend() as backend:
            model = art.TrainableModel(
                name=config.model_name,
                project="rlm-bench",
                base_model=config.base_model,
                _internal_config=art.dev.InternalModelConfig(
                    engine_args=art.dev.EngineArgs(
                        tensor_parallel_size=config.tensor_parallel_size,
                        gpu_memory_utilization=0.7,
                    ),
                    init_args=art.dev.InitArgs(
                        load_in_4bit=False,
                    ),
                ),
            )
            await model.register(backend)
            print(f"{GREEN}LocalBackend ready (TP={args.tp}). vLLM managed internally.{RESET}\n")
            await _run_benchmark(model, scenario_map, config, args, docker_client)
    else:
        model = art.Model(
            name="bench",
            project="rlm-bench",
            inference_base_url=config.vllm_url,
            inference_api_key="dummy",
            inference_model_name=config.base_model,
        )
        await _run_benchmark(model, scenario_map, config, args, docker_client)


if __name__ == "__main__":
    asyncio.run(main())
