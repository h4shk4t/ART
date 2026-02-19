"""Rollout function for R2E-Gym RLM-style agent.

Each rollout runs a multi-step REPL loop: the model writes Python code,
the code is exec()'d in a persistent REPL with Docker helpers, and the
output goes back to the model. The model can also spawn recursive sub-agents
via sub_query() to delegate codebase analysis.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import openai

import art

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv

from repl import DockerREPL, extract_python_code

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert software engineer. You will receive a bug report for a repository \
located at /testbed. Fix the bug by writing Python code in ```python or ```repl blocks.

This is an RLM-style environment: you can use the REPL helpers to explore the \
repo, and you can delegate analysis to sub-agents. Use sub-agents when the task \
benefits from parallel or deeper exploration.

Available REPL helpers (persistent state across steps):
- ls(path="/testbed", max_depth=2) -> str
- read(path, start=None, end=None) -> str
- grep(pattern, path="/testbed", file_pattern="*.py") -> str
- apply_patch(patch: str) -> str   # apply a unified diff
- run_tests(timeout=300) -> str
- bash(cmd, timeout=90) -> str
- finish() -> str                  # call when done
- sub_query(prompt, max_steps=8) -> str  # delegate analysis to a sub-agent
- llm_query(prompt, max_steps=8) -> str  # alias for sub_query (RLM prompt)
- llm_query_batched([prompt1, prompt2], max_steps=8) -> list[str]

Workflow:
1. Read the bug report. Identify the failing test and relevant source files.
2. Use grep/read to understand the root cause (be surgical — read specific line ranges).
3. If needed, delegate file exploration to sub-agents with sub_query/llm_query.
4. Write a minimal patch with apply_patch().
5. run_tests() to verify.
6. Call finish() once tests pass.

IMPORTANT: Keep your code short. Prefer targeted reads (read(path, start, end)) over \
reading entire files. Always call finish() when done — you only get reward if you do.
"""

CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 28000
MAX_COMPLETION_TOKENS = 4096


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // CHARS_PER_TOKEN


def _trim_context(messages: list[dict], budget: int = MAX_CONTEXT_TOKENS) -> list[dict]:
    """Drop oldest mid-conversation turns to stay within token budget.

    Keeps: system prompt (idx 0), initial bug report (idx 1), and the most
    recent turns. Drops from the middle outward.
    """
    if _estimate_tokens(messages) <= budget:
        return messages

    preserved_prefix = messages[:2]
    rest = messages[2:]

    trimmed_note = {
        "role": "user",
        "content": "[Earlier conversation turns were trimmed to fit context window. "
        "Continue from the most recent output above.]",
    }

    while rest and _estimate_tokens(preserved_prefix + [trimmed_note] + rest) > budget:
        rest.pop(0)

    return preserved_prefix + [trimmed_note] + rest


@dataclass
class R2EScenario:
    ds: dict[str, Any]
    max_steps: int = 20
    step_timeout: int = 90
    reward_timeout: int = 300


async def rollout(
    model: art.Model,
    scenario: R2EScenario,
    semaphore: asyncio.Semaphore,
) -> art.Trajectory:
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0.0,
        metadata={
            "instance_id": scenario.ds.get("instance_id", "unknown"),
            "repo": scenario.ds.get("repo_name", scenario.ds.get("repo", "unknown")),
        },
        metrics={
            "num_steps": 0,
            "finished": False,
            "num_sub_queries": 0,
        },
    )

    env: RepoEnv | None = None
    try:
        async with semaphore:
            env = await asyncio.to_thread(_create_env, scenario)
            task_instruction = await asyncio.to_thread(env.get_task_instruction)

            repl = DockerREPL(env, model, traj)

            traj.messages_and_choices = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Bug report:\n\n{task_instruction}\n\nFix this bug.",
                },
            ]

            consecutive_no_code = 0

            for step in range(scenario.max_steps):
                messages_for_api = _trim_context(
                    [_to_dict(m) for m in traj.messages_and_choices],
                    budget=MAX_CONTEXT_TOKENS,
                )

                try:
                    async with traj.track_duration("llm_completion"):
                        client = model.openai_client()
                        response = await client.chat.completions.create(
                            model=model.get_inference_name(),
                            messages=messages_for_api,
                            temperature=1.0,
                            max_completion_tokens=MAX_COMPLETION_TOKENS,
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
                    if consecutive_no_code >= 2:
                        traj.messages_and_choices.append({
                            "role": "user",
                            "content": "You must write Python code in a ```python or ```repl block. "
                            "If you're done, write: ```python\nfinish()\n```",
                        })
                    else:
                        traj.messages_and_choices.append({
                            "role": "user",
                            "content": "Write Python code in a ```python or ```repl block to proceed.",
                        })
                    continue
                else:
                    consecutive_no_code = 0

                sub_queries_before = len(traj.additional_histories)
                observation = await asyncio.to_thread(repl.execute, code)
                sub_queries_after = len(traj.additional_histories)

                traj.metrics["num_steps"] = step + 1
                traj.metrics["num_sub_queries"] = sub_queries_after

                if repl.is_finished():
                    traj.metrics["finished"] = True
                    traj.messages_and_choices.append({
                        "role": "user",
                        "content": f"Output:\n```\n{observation}\n```\nAgent called finish().",
                    })
                    break

                num_new_subs = sub_queries_after - sub_queries_before
                sub_note = ""
                if num_new_subs > 0:
                    sub_note = f"\n[{num_new_subs} sub-agent(s) spawned this step]"

                traj.messages_and_choices.append({
                    "role": "user",
                    "content": f"Output:\n```\n{observation}\n```{sub_note}",
                })

            reward = await asyncio.to_thread(env.compute_reward, scenario.reward_timeout)
            traj.reward = float(reward)

    except Exception as e:
        traj.log(f"Rollout error: {e}")
        logger.exception("Rollout failed for %s", scenario.ds.get("instance_id", "unknown"))
    finally:
        if env is not None:
            try:
                await asyncio.to_thread(env.close)
            except Exception:
                logger.exception("Failed to close environment")

    return traj.finish()


def _to_dict(item) -> dict:
    from openai.types.chat.chat_completion import Choice

    if isinstance(item, Choice):
        return {"role": "assistant", "content": item.message.content or ""}
    return item


def _create_env(scenario: R2EScenario) -> RepoEnv:
    env_args = EnvArgs(ds=scenario.ds)
    return RepoEnv(args=env_args, verbose=False)
