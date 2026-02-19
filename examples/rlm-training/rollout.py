"""Rollout function for distributed RLM training.

Runs a single rollout: copies cached repo to a temp dir, runs the REPL loop
with local file ops, then sends modified files to the Docker service for
test execution and reward computation.

All file operations during the REPL loop are local. Only run_tests()
(at reward time) hits the network.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from openai.types.chat.chat_completion import Choice

import art

from config import ExperimentConfig
from docker_client import DockerClient, get_modified_files
from health import HealthMonitor
from repl import LocalREPL, extract_python_code
from trajectory_logger import NullSession, TrajectoryLogger

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


@dataclass
class Scenario:
    """One bug-fixing task to roll out on."""

    instance_id: str
    docker_image: str
    problem_statement: str
    expected_output_json: str = ""
    test_cmd: str = "bash -lc 'cd /testbed && bash run_tests.sh 2>&1'"
    extra: dict[str, Any] | None = None

    @classmethod
    def from_dataset_entry(cls, ds: dict[str, Any]) -> Scenario:
        repo = ds.get("repo_name", "unknown")
        commit = ds.get("commit_hash", "unknown")
        instance_id = f"{repo}__{commit[:12]}"
        return cls(
            instance_id=instance_id,
            docker_image=ds.get("docker_image", ""),
            problem_statement=ds.get("problem_statement", ""),
            expected_output_json=ds.get("expected_output_json", ""),
            test_cmd=ds.get("test_cmd", cls.test_cmd),
            extra=ds,
        )


def _estimate_tokens(messages: list[dict], chars_per_token: int = CHARS_PER_TOKEN) -> int:
    return sum(len(m.get("content", "")) for m in messages) // chars_per_token


def _trim_context(
    messages: list[dict], budget: int = 28000, chars_per_token: int = CHARS_PER_TOKEN
) -> list[dict]:
    """Drop oldest mid-conversation turns to stay within token budget."""
    if _estimate_tokens(messages, chars_per_token) <= budget:
        return messages

    preserved = messages[:2]
    rest = messages[2:]
    note = {
        "role": "user",
        "content": "[Earlier conversation turns were trimmed to fit context window. "
        "Continue from the most recent output above.]",
    }

    while rest and _estimate_tokens(preserved + [note] + rest, chars_per_token) > budget:
        rest.pop(0)

    return preserved + [note] + rest


def _to_dict(item: Any) -> dict:
    if isinstance(item, Choice):
        return {"role": "assistant", "content": item.message.content or ""}
    return item


def _copy_from_cache(instance_id: str, cache_dir: str) -> tuple[Path, Path]:
    """Copy cached repo to a temp rollout directory.

    Returns (work_dir, cache_src) — work_dir is the mutable copy,
    cache_src is the original for diffing at reward time.
    """
    index_path = Path(cache_dir) / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    entry = index.get(instance_id)
    if not entry:
        raise FileNotFoundError(
            f"Instance {instance_id!r} not found in cache index at {index_path}"
        )

    src = entry["cache_path"] if isinstance(entry, dict) else entry
    cache_src = Path(src)

    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="rlm_rollout_"))
    shutil.copytree(str(cache_src), str(work_dir), dirs_exist_ok=True)
    return work_dir, cache_src


async def rollout(
    model: art.Model,
    scenario: Scenario,
    config: ExperimentConfig,
    docker_client: DockerClient,
    health: HealthMonitor,
    traj_logger: TrajectoryLogger,
    semaphore: asyncio.Semaphore,
    training_step: int = 0,
) -> art.Trajectory:
    """Run a single REPL rollout and return an ART trajectory.

    Steps:
    1. Copy cached repo to temp dir
    2. Build system prompt and REPL namespace
    3. Multi-step REPL loop (all local)
    4. Diff modified files, send to Docker service for tests
    5. Compute reward
    6. Cleanup temp dir
    """
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0.0,
        metadata={
            "instance_id": scenario.instance_id,
            "docker_image": scenario.docker_image,
        },
        metrics={
            "num_steps": 0,
            "finished": False,
            "num_sub_queries": 0,
        },
    )

    system_prompt = (
        config.system_prompt_fn() if config.system_prompt_fn else
        "You are an expert software engineer. Fix the bug by writing Python code."
    )
    sub_agent_prompt = (
        config.sub_agent_system_prompt_fn() if config.sub_agent_system_prompt_fn else ""
    )

    session = traj_logger.start_session(
        instance_id=scenario.instance_id,
        training_step=training_step,
        model_name=config.model_name,
        config_metadata=config.to_metadata(),
    )
    if session is None:
        session = NullSession()

    work_dir: Path | None = None
    try:
        async with semaphore:
            work_dir, cache_src = await asyncio.to_thread(
                _copy_from_cache, scenario.instance_id, config.repo_cache_dir
            )
            health.track_rollout(work_dir)

            repl = LocalREPL(
                work_dir=work_dir,
                model=model,
                trajectory=traj,
                health=health,
                sub_agent_system_prompt=sub_agent_prompt,
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
                step_t0 = time.time()

                messages_for_api = _trim_context(
                    [_to_dict(m) for m in traj.messages_and_choices],
                    budget=config.max_context_tokens,
                    chars_per_token=config.chars_per_token,
                )

                try:
                    async with traj.track_duration("llm_completion"):
                        client = model.openai_client()
                        response = await client.chat.completions.create(
                            model=model.get_inference_name(),
                            messages=messages_for_api,
                            temperature=1.0,
                            max_completion_tokens=config.max_completion_tokens,
                        )
                except openai.BadRequestError as e:
                    logger.warning("API error at step %d: %s", step, e)
                    traj.messages_and_choices.append({
                        "role": "user",
                        "content": "[Context too long — earlier turns were dropped. Please continue.]",
                    })
                    continue

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

                    session.log_iteration(
                        iteration=step + 1,
                        prompt=messages_for_api,
                        response=assistant_text,
                        code=None,
                        exec_result=None,
                        iteration_time=time.time() - step_t0,
                    )
                    continue

                consecutive_no_code = 0

                sub_before = len(traj.additional_histories)
                exec_result = await asyncio.to_thread(repl.execute, code)
                sub_after = len(traj.additional_histories)

                traj.metrics["num_steps"] = step + 1
                traj.metrics["num_sub_queries"] = sub_after

                observation = exec_result.stdout

                final_answer = None
                if repl.is_finished():
                    traj.metrics["finished"] = True
                    final_answer = "Agent called finish()."
                    traj.messages_and_choices.append({
                        "role": "user",
                        "content": f"Output:\n```\n{observation}\n```\nAgent called finish().",
                    })
                    session.log_iteration(
                        iteration=step + 1,
                        prompt=messages_for_api,
                        response=assistant_text,
                        code=code,
                        exec_result=exec_result,
                        final_answer=final_answer,
                        iteration_time=time.time() - step_t0,
                    )
                    break

                num_new = sub_after - sub_before
                sub_note = f"\n[{num_new} sub-agent(s) spawned this step]" if num_new > 0 else ""
                traj.messages_and_choices.append({
                    "role": "user",
                    "content": f"Output:\n```\n{observation}\n```{sub_note}",
                })

                session.log_iteration(
                    iteration=step + 1,
                    prompt=messages_for_api,
                    response=assistant_text,
                    code=code,
                    exec_result=exec_result,
                    iteration_time=time.time() - step_t0,
                )

            # Compute reward via Docker service
            modified = await asyncio.to_thread(
                get_modified_files, work_dir, cache_src
            )

            if modified and config.reward_fn:
                try:
                    test_result = await docker_client.run_tests(
                        docker_image=scenario.docker_image,
                        modified_files=modified,
                        test_cmd=scenario.test_cmd,
                    )
                    test_output = test_result.get("test_output", "")
                    reward = config.reward_fn(
                        test_output,
                        scenario.extra or {},
                    )
                    traj.reward = reward
                    traj.metrics["test_exit_code"] = test_result.get("exit_code", -1)
                except Exception as e:
                    logger.error("Reward computation failed for %s: %s", scenario.instance_id, e)
                    traj.reward = 0.0
            else:
                traj.reward = 0.0

            session.finalize(
                reward=traj.reward,
                total_steps=traj.metrics.get("num_steps", 0),
            )

    except Exception as e:
        traj.log(f"Rollout error: {e}")
        logger.exception("Rollout failed for %s", scenario.instance_id)
    finally:
        if work_dir is not None:
            health.unregister_rollout(work_dir)
            try:
                await asyncio.to_thread(shutil.rmtree, str(work_dir), True)
            except Exception:
                pass

    return traj.finish()
