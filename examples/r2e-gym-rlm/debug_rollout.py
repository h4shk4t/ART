"""Debug rollouts for R2E-Gym RLM pipeline.

Runs a few rollouts and prints:
- system + user prompt snippets
- whether model produced code blocks
- REPL / sub-query usage metrics
- reward and finish status
"""
from __future__ import annotations

import asyncio
import inspect
import os
import re
from typing import Any

from datasets import load_dataset
from openai.types.chat.chat_completion import Choice

import art
USE_RLM_ENV = os.environ.get("USE_RLM_ENV", "1") == "1"
if USE_RLM_ENV:
    from rollout_rlm_env import R2ERLMScenario as R2EScenario, rollout  # type: ignore[import-not-found]
else:
    from rollout import R2EScenario, rollout  # type: ignore[import-not-found]
from repl import extract_python_code


class Config:
    model_name: str = "r2e-rlm-qwen3-14b-debug"
    project: str = "r2e-gym-rlm-debug"
    base_model: str = "OpenPipe/Qwen3-14B-Instruct"

    max_steps: int = 6
    max_concurrent: int = 1
    dataset_size: int = 1
    num_rollouts: int = 2
    require_wandb_api_key: bool = True


HELPER_NAMES = [
    "ls",
    "read",
    "grep",
    "apply_patch",
    "run_tests",
    "bash",
    "finish",
    "sub_query",
    "llm_query",
    "llm_query_batched",
]


def _count_helper_calls(code: str) -> dict[str, int]:
    counts = {name: 0 for name in HELPER_NAMES}
    for name in HELPER_NAMES:
        total = 0
        idx = 0
        while True:
            idx = code.find(name, idx)
            if idx == -1:
                break
            # Ensure a word boundary before name
            if idx > 0 and (code[idx - 1].isalnum() or code[idx - 1] == "_"):
                idx += len(name)
                continue
            j = idx + len(name)
            while j < len(code) and code[j].isspace():
                j += 1
            if j < len(code) and code[j] == "(":
                total += 1
            idx += len(name)
        counts[name] = total
    return counts


def _summarize_messages(messages: list[Any]) -> dict[str, Any]:
    system = None
    user = None
    first_assistant = None
    last_assistant = None
    assistant_choices = []
    helper_calls = {name: 0 for name in HELPER_NAMES}
    tool_call_counts: dict[str, int] = {}

    for item in messages:
        if isinstance(item, Choice):
            assistant_choices.append(item)
            if first_assistant is None:
                first_assistant = item.message.content or ""
            last_assistant = item.message.content or ""
            for tc in item.message.tool_calls or []:
                name = tc.function.name
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
        elif isinstance(item, dict):
            if item.get("role") == "system" and system is None:
                system = item.get("content", "")
            if item.get("role") == "user" and user is None:
                user = item.get("content", "")

    code_blocks = 0
    for c in assistant_choices:
        code = extract_python_code(c.message.content or "")
        if code is not None:
            code_blocks += 1
            counts = _count_helper_calls(code)
            for k, v in counts.items():
                helper_calls[k] += v

    return {
        "system_preview": (system or "")[:400],
        "user_preview": (user or "")[:400],
        "first_assistant_preview": (first_assistant or "")[:400],
        "last_assistant_preview": (last_assistant or "")[:400],
        "assistant_messages": len(assistant_choices),
        "assistant_code_blocks": code_blocks,
        "helper_calls": helper_calls,
        "tool_call_counts": tool_call_counts,
    }


async def run_r2e_rollouts() -> None:
    print("=== R2E-Gym RLM Debug ===")
    print(f"USE_RLM_ENV={USE_RLM_ENV}")
    config = Config()

    if config.require_wandb_api_key and not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set. Skipping R2E rollouts.")
        return

    backend = art.ServerlessBackend()
    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
    )
    await model.register(backend)

    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
    selected = ds.select(range(config.dataset_size))
    scenarios = [
        R2EScenario(ds=entry, max_steps=config.max_steps)
        for entry in selected
    ]

    semaphore = asyncio.Semaphore(config.max_concurrent)

    for i in range(config.num_rollouts):
        print(f"\n--- Rollout {i+1}/{config.num_rollouts} ---")
        traj = await rollout(model, scenarios[0], semaphore)

        summary = _summarize_messages(traj.messages_and_choices)
        print(f"reward={traj.reward}")
        print(f"metrics={traj.metrics}")
        print(f"additional_histories={len(traj.additional_histories)}")
        print(f"assistant_messages={summary['assistant_messages']}")
        print(f"assistant_code_blocks={summary['assistant_code_blocks']}")
        print(f"helper_calls={summary['helper_calls']}")
        print(f"tool_call_counts={summary['tool_call_counts']}")
        if traj.logs:
            print(f"logs={traj.logs}")

        print("\nSYSTEM PROMPT (preview):")
        print(summary["system_preview"])
        print("\nUSER PROMPT (preview):")
        print(summary["user_preview"])
        print("\nFIRST ASSISTANT (preview):")
        print(summary["first_assistant_preview"])
        print("\nLAST ASSISTANT (preview):")
        print(summary["last_assistant_preview"])

    print("\nR2E-Gym debug complete.")


async def smoke_test_rlm_env() -> None:
    print("\n=== Prime Intellect RLMEnv Smoke Test ===")
    try:
        import verifiers.envs.experimental.rlm_env as rlm_env  # type: ignore
    except Exception as e:
        print(f"Failed to import verifiers RLMEnv: {e}")
        return

    print(f"rlm_env module: {getattr(rlm_env, '__file__', 'unknown')}")

    # Try to find an env class
    env_cls = None
    for name in ["RLMEnv", "RLMEnvV2", "RLMEnvV3"]:
        if hasattr(rlm_env, name):
            env_cls = getattr(rlm_env, name)
            print(f"Found env class: {name}")
            break

    if env_cls is None:
        print("No RLMEnv class found in module.")
        return

    try:
        sig = inspect.signature(env_cls)
        print(f"RLMEnv signature: {sig}")
    except Exception as e:
        print(f"Could not inspect signature: {e}")

    from datasets import Dataset

    dummy_dataset = Dataset.from_list(
        [{"prompt": "Hello from RLMEnv smoke test", "info": {}}]
    )
    env = env_cls(
        dataset=dummy_dataset,
        repl_language="python",
        execution_backend="local",
        max_iterations=1,
    )
    state = {"prompt": "Hello from RLMEnv smoke test", "info": {}}  # type: ignore[dict-item]
    try:
        state = await env.setup_state(state)
        sys_prompt = state.get("rlm_system_prompt", "")
        fs_has_data = state.get("rlm_fs_has_data", None)
        print("RLMEnv setup_state OK")
        print(f"system_prompt_len={len(sys_prompt)}")
        print(f"fs_has_data={fs_has_data}")
    except Exception as e:
        print(f"RLMEnv setup_state failed: {e}")
    finally:
        try:
            await env._cleanup(state)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await env._teardown()  # type: ignore[attr-defined]
        except Exception:
            pass


async def main() -> None:
    await smoke_test_rlm_env()
    await run_r2e_rollouts()


if __name__ == "__main__":
    asyncio.run(main())
