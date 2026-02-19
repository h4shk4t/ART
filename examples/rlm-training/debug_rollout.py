"""Interactive single-task rollout runner for debugging and development.

Run a single rollout against one R2E-Gym task, with:
  - Full trajectory logging (always, regardless of sample_rate)
  - Verbose console output showing each REPL step
  - Optional human-in-the-loop mode (pause between steps)

Usage:
    # Run with default config
    uv run python debug_rollout.py --instance-id "django__abc123def456"

    # Run with human-in-the-loop (pause between steps)
    uv run python debug_rollout.py --instance-id "django__abc123def456" --interactive

    # Run with custom config
    uv run python debug_rollout.py --instance-id "django__abc123def456" \
        --max-steps 20 --model "Qwen/Qwen3-14B"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import openai
from openai.types.chat.chat_completion import Choice

import art

from config import ExperimentConfig
from docker_client import DockerClient, get_modified_files
from health import HealthMonitor
from prompts import r2e_rlm_system_prompt, sub_agent_system_prompt
from repl import LocalREPL, extract_python_code
from rewards import binary_test_reward
from trajectory_logger import TrajectoryLogger, TrajectorySession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug_rollout")

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def _banner(step: int, total: int, label: str) -> str:
    return f"\n{'='*60}\n  Step {step}/{total}: {label}\n{'='*60}"


async def debug_rollout(
    config: ExperimentConfig,
    instance_id: str,
    interactive: bool = False,
) -> float:
    """Run one rollout with verbose output and trajectory logging."""

    # Always log in debug mode
    traj_logger = TrajectoryLogger(
        log_dir=config.log_dir,
        experiment_name=f"debug_{config.experiment_name}",
        sample_rate=1.0,
    )

    # Load cache index
    index_path = Path(config.repo_cache_dir) / "index.json"
    if not index_path.exists():
        logger.error("Cache index not found at %s", index_path)
        sys.exit(1)

    with open(index_path) as f:
        cache_index = json.load(f)

    if instance_id not in cache_index:
        logger.error("Instance %r not in cache. Available: %s", instance_id, list(cache_index.keys())[:10])
        sys.exit(1)

    cache_entry = cache_index[instance_id]
    cache_src = cache_entry["cache_path"] if isinstance(cache_entry, dict) else cache_entry

    # Load dataset entry
    from datasets import load_dataset

    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
    entry = None
    for row in ds:
        repo = row.get("repo_name", "unknown")
        commit = row.get("commit_hash", "unknown")
        if f"{repo}__{commit[:12]}" == instance_id:
            entry = dict(row)
            break

    if entry is None:
        logger.error("Could not find dataset entry for %s", instance_id)
        sys.exit(1)

    problem_statement = entry.get("problem_statement", "No problem statement found.")
    docker_image = entry.get("docker_image", "")

    print(f"\n{BLUE}Instance:{RESET} {instance_id}")
    print(f"{BLUE}Docker image:{RESET} {docker_image}")
    print(f"{BLUE}Problem statement:{RESET}\n{problem_statement[:500]}...")

    # Setup
    health = HealthMonitor.from_config(config)
    docker_client = DockerClient(config.docker_service_url)

    model = art.Model(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
    )

    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0.0,
        metadata={"instance_id": instance_id, "docker_image": docker_image},
        metrics={"num_steps": 0, "finished": False, "num_sub_queries": 0},
    )

    system_prompt = config.system_prompt_fn() if config.system_prompt_fn else r2e_rlm_system_prompt()
    sub_prompt = config.sub_agent_system_prompt_fn() if config.sub_agent_system_prompt_fn else sub_agent_system_prompt()

    session = traj_logger.start_session(
        instance_id=instance_id,
        training_step=0,
        model_name=config.model_name,
        config_metadata=config.to_metadata(),
    )

    # Copy repo
    import tempfile

    work_dir = Path(tempfile.mkdtemp(prefix="rlm_debug_"))
    shutil.copytree(cache_src, str(work_dir), dirs_exist_ok=True)
    print(f"{DIM}Work dir: {work_dir}{RESET}")

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

    messages: list[dict | Choice] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Bug report:\n\n{problem_statement}\n\nFix this bug."},
    ]
    traj.messages_and_choices = messages

    try:
        for step in range(config.max_steps):
            print(_banner(step + 1, config.max_steps, "LLM Completion"))

            if interactive:
                input(f"{YELLOW}Press Enter to continue (Ctrl+C to abort)...{RESET}")

            step_t0 = time.time()

            def _to_dict(item):
                if isinstance(item, Choice):
                    return {"role": "assistant", "content": item.message.content or ""}
                return item

            messages_for_api = [_to_dict(m) for m in traj.messages_and_choices]

            try:
                client = model.openai_client()
                response = await client.chat.completions.create(
                    model=model.get_inference_name(),
                    messages=messages_for_api,
                    temperature=1.0,
                    max_completion_tokens=config.max_completion_tokens,
                )
            except openai.BadRequestError as e:
                print(f"{RED}API error: {e}{RESET}")
                continue

            choice = response.choices[0]
            traj.messages_and_choices.append(choice)
            assistant_text = choice.message.content or ""

            print(f"\n{GREEN}Assistant:{RESET}")
            print(assistant_text[:2000])
            if len(assistant_text) > 2000:
                print(f"{DIM}... ({len(assistant_text)} chars total){RESET}")

            code = extract_python_code(assistant_text)

            if code is None:
                print(f"\n{YELLOW}[No code block found]{RESET}")
                traj.messages_and_choices.append({
                    "role": "user",
                    "content": "Write Python code in a ```python or ```repl block to proceed.",
                })
                if session:
                    session.log_iteration(
                        iteration=step + 1,
                        prompt=messages_for_api,
                        response=assistant_text,
                        code=None,
                        iteration_time=time.time() - step_t0,
                    )
                continue

            print(f"\n{BLUE}Executing:{RESET}")
            print(code[:1000])

            exec_result = await asyncio.to_thread(repl.execute, code)
            elapsed = time.time() - step_t0

            print(f"\n{GREEN}Output ({exec_result.execution_time:.2f}s):{RESET}")
            print(exec_result.stdout[:2000])
            if exec_result.stderr:
                print(f"{RED}Stderr:{RESET}")
                print(exec_result.stderr[:1000])
            if exec_result.sub_agent_calls:
                for sc in exec_result.sub_agent_calls:
                    print(f"{DIM}  Sub-agent: {sc.prompt[:80]}... → {sc.response[:80]}...{RESET}")

            traj.metrics["num_steps"] = step + 1
            traj.metrics["num_sub_queries"] = len(traj.additional_histories)

            final_answer = None
            if repl.is_finished():
                final_answer = "Agent called finish()."
                traj.metrics["finished"] = True
                traj.messages_and_choices.append({
                    "role": "user",
                    "content": f"Output:\n```\n{exec_result.stdout}\n```\nAgent called finish().",
                })
                if session:
                    session.log_iteration(
                        iteration=step + 1,
                        prompt=messages_for_api,
                        response=assistant_text,
                        code=code,
                        exec_result=exec_result,
                        final_answer=final_answer,
                        iteration_time=elapsed,
                    )
                print(f"\n{GREEN}Agent called finish() at step {step + 1}{RESET}")
                break

            traj.messages_and_choices.append({
                "role": "user",
                "content": f"Output:\n```\n{exec_result.stdout}\n```",
            })
            if session:
                session.log_iteration(
                    iteration=step + 1,
                    prompt=messages_for_api,
                    response=assistant_text,
                    code=code,
                    exec_result=exec_result,
                    iteration_time=elapsed,
                )

        # Compute reward
        print(_banner(0, 0, "Reward Computation"))
        cache_path = Path(cache_src)
        modified = get_modified_files(work_dir, cache_path)
        print(f"Modified files: {list(modified.keys())}")

        reward = 0.0
        if modified:
            try:
                test_result = await docker_client.run_tests(
                    docker_image=docker_image,
                    modified_files=modified,
                )
                test_output = test_result.get("test_output", "")
                print(f"\n{BLUE}Test output:{RESET}")
                print(test_output[:3000])

                reward_fn = config.reward_fn or binary_test_reward
                reward = reward_fn(test_output, entry)
            except Exception as e:
                print(f"{RED}Test execution failed: {e}{RESET}")
                print(f"{YELLOW}Skipping reward (Docker service may not be running){RESET}")

        traj.reward = reward
        print(f"\n{'='*60}")
        print(f"  {GREEN if reward > 0 else RED}Reward: {reward}{RESET}")
        print(f"  Steps: {traj.metrics['num_steps']}")
        print(f"  Finished: {traj.metrics['finished']}")
        print(f"  Sub-queries: {traj.metrics['num_sub_queries']}")
        print(f"{'='*60}")

        if session:
            session.finalize(reward=reward, total_steps=traj.metrics.get("num_steps", 0))

    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)

    return reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug a single RLM rollout")
    parser.add_argument("--instance-id", required=True, help="Cache instance_id to test")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--model-name", default="r2e-rlm-debug")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--docker-url", default="http://docker-node:8000")
    parser.add_argument("--cache-dir", default="/data/repo-cache")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--interactive", action="store_true", help="Pause between steps")
    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_name="debug",
        model_name=args.model_name,
        base_model=args.model,
        vllm_url=args.vllm_url,
        docker_service_url=args.docker_url,
        repo_cache_dir=args.cache_dir,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        system_prompt_fn=r2e_rlm_system_prompt,
        reward_fn=binary_test_reward,
        sub_agent_system_prompt_fn=sub_agent_system_prompt,
    )

    reward = asyncio.run(debug_rollout(config, args.instance_id, interactive=args.interactive))
    sys.exit(0 if reward > 0 else 1)


if __name__ == "__main__":
    main()
