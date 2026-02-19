"""RLMEnv-based rollout for R2E-Gym tasks.

Uses verifiers' RLMEnv with a sandbox docker image and bash REPL.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any

from openai.types.chat.chat_completion import Choice
from datasets import Dataset

import art
import re
from verifiers.utils.worker_utils import get_free_port

try:
    import verifiers as vf
    from verifiers.envs.experimental.rlm_env import RLMEnv
except Exception as e:  # pragma: no cover
    raise ImportError(
        "verifiers is required for RLMEnv rollout. Ensure it is installed."
    ) from e


class PatchedRLMEnv(RLMEnv):
    """Patch token id None values, ensure pip, and run tests before cleanup."""

    async def on_sandbox_ready(self, state, sandbox_id):
        executor = self._executor
        setup_cmds = [
            "python -m ensurepip --default-pip 2>/dev/null || python -m ensurepip 2>/dev/null || true",
            "ln -sf /r2e_tests /testbed/r2e_tests 2>/dev/null || true",
        ]
        for cmd in setup_cmds:
            try:
                await executor._execute_sandbox_command(
                    sandbox_id, f"bash -lc '{cmd}'", timeout=60
                )
            except Exception:
                pass

    @vf.cleanup(priority=10)
    async def run_tests_before_cleanup(self, state):
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id or state.get("error"):
            return
        if self.execution_backend != "sandbox":
            return
        try:
            result = await self._executor._execute_sandbox_command(
                sandbox_id,
                "bash -lc 'cd /testbed && bash run_tests.sh 2>&1'",
                timeout=300,
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            state["_test_output"] = stdout or stderr
            state["_test_exit_code"] = getattr(result, "exit_code", None)
        except Exception as e:
            state["_test_output"] = ""
            state["_test_error"] = str(e)

    async def add_model_response(self, state, prompt_messages, response):
        try:
            if (
                hasattr(response, "prompt_token_ids")
                and response.prompt_token_ids is None
            ):
                response.prompt_token_ids = []
        except Exception:
            pass
        try:
            if (
                hasattr(response, "choices")
                and response.choices
                and hasattr(response.choices[0], "token_ids")
                and response.choices[0].token_ids is None
            ):
                response.choices[0].token_ids = []
        except Exception:
            pass
        return await super().add_model_response(state, prompt_messages, response)


@dataclass
class R2ERLMScenario:
    ds: dict[str, Any]
    max_steps: int = 20
    reward_timeout: int = 300


def _get_docker_image(ds: dict[str, Any]) -> str:
    if "docker_image" in ds and ds["docker_image"]:
        return ds["docker_image"]
    if "image_name" in ds and ds["image_name"]:
        return ds["image_name"]
    raise ValueError("No docker image found in dataset entry")


def _get_repo_name(ds: dict[str, Any]) -> str:
    return ds.get("repo_name") or ds.get("repo") or "unknown"


def _make_context_dir(task_instruction: str, instance_id: str) -> str:
    base = tempfile.mkdtemp(prefix=f"rlm_ctx_{instance_id}_")
    task_path = os.path.join(base, "TASK.txt")
    with open(task_path, "w", encoding="utf-8") as f:
        f.write("R2E-Gym task\n")
        f.write("Repo is located at /testbed in the container.\n\n")
        f.write(task_instruction)
        f.write("\n")
    return base


def _build_prompt(task_instruction: str, backend: str) -> str:
    location_hint = (
        "Repo is at /testbed. Use call_bash_repl to explore /testbed."
        if backend == "sandbox"
        else "Your working directory is the filesystem root; start with `pwd` and `ls`."
    )
    return (
        "Bug report:\n\n"
        f"{task_instruction}\n\n"
        f"{location_hint} "
        "Use call_bash_repl for iterative exploration, and llm_batch for sub-LLM help.\n\n"
        "Mandatory: call llm_batch at least once before finalizing. "
        "Start by summarizing the bug report with llm_batch."
    )


def _choice_to_message(choice: Choice) -> dict:
    content = choice.message.content or ""
    tool_calls = choice.message.tool_calls or []
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [tc.model_dump(mode="json") for tc in tool_calls]
    return msg


def _build_messages_and_choices(state: dict) -> list:
    messages_and_choices: list = []
    full_messages: list[dict] = []
    for step in state.get("trajectory", []):
        prompt = step["prompt"]
        if (
            isinstance(prompt, list)
            and len(prompt) >= len(full_messages)
            and prompt[: len(full_messages)] == full_messages
        ):
            new_msgs = prompt[len(full_messages) :]
        else:
            new_msgs = prompt if isinstance(prompt, list) else []
        messages_and_choices.extend(new_msgs)

        response = step.get("response")
        if response and getattr(response, "choices", None):
            choice = response.choices[0]
            messages_and_choices.append(choice)
            full_messages = list(prompt) + [_choice_to_message(choice)]
        else:
            full_messages = list(prompt)
    return messages_and_choices


def _parse_pytest_log(log: str) -> dict[str, str]:
    if not log or "short test summary info" not in log:
        return {}
    summary = log.split("short test summary info")[1].strip()
    result = {}
    for line in summary.split("\n"):
        if "PASSED" in line:
            result[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            result[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            result[name.split(" - ")[0]] = "ERROR"
    return result


def _decolor(d: dict) -> dict:
    strip = lambda s: re.sub(r"\u001b\[\d+m", "", s)
    return {strip(k): v for k, v in d.items()}


def _compute_reward_from_logs(log_output: str, ds: dict[str, Any]) -> float:
    parse = _decolor(_parse_pytest_log(log_output))

    expected_json = ds.get("expected_output_json")
    if not expected_json:
        return 0.0

    expected: dict = json.loads(expected_json)
    expected = _decolor(expected)
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

    if len(parse) != len(expected):
        return 0.0

    for k in parse.keys():
        if not k:
            continue
        if k not in expected:
            return 0.0
        if parse[k] != expected[k]:
            return 0.0

    return 1.0


async def rollout(
    model: art.Model,
    scenario: R2ERLMScenario,
    semaphore: asyncio.Semaphore,
) -> art.Trajectory:
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0.0,
        metadata={
            "instance_id": scenario.ds.get("instance_id", "unknown"),
            "repo": _get_repo_name(scenario.ds),
        },
        metrics={
            "num_steps": 0,
            "finished": False,
            "num_sub_queries": 0,
        },
    )

    env: RLMEnv | None = None
    context_dir: str | None = None
    context_dir_owned = False
    state: dict | None = None
    try:
        async with semaphore:
            docker_image = _get_docker_image(scenario.ds)
            dummy_dataset = Dataset.from_list([{"prompt": "RLMEnv placeholder"}])
            backend = os.environ.get("RLM_ENV_BACKEND", "local").lower()
            if backend not in {"local", "sandbox"}:
                raise ValueError(f"Invalid RLM_ENV_BACKEND: {backend}")
            interception_url = os.environ.get("RLM_INTERCEPTION_URL")
            if backend == "sandbox" and not os.environ.get("PRIME_API_KEY") and not interception_url:
                raise ValueError(
                    "PRIME_API_KEY or RLM_INTERCEPTION_URL is required for RLMEnv sandbox backend"
                )

            interception_port = get_free_port()
            env = PatchedRLMEnv(
                dataset=dummy_dataset,
                repl_language="bash",
                execution_backend=backend,
                interception_port=interception_port,
                interception_url=interception_url,
                root_prompt_verbosity="heavy",
                sub_prompt_verbosity="medium",
                include_sub_llm_in_trajectory=True,
                max_iterations=scenario.max_steps,
                sandbox_docker_image=docker_image,
                score_rollouts=False,
            )

            task_instruction = (
                scenario.ds.get("problem_statement")
                or scenario.ds.get("issue_description")
                or scenario.ds.get("task")
                or ""
            )

            prompt = _build_prompt(task_instruction, backend)
            instance_id = scenario.ds.get("instance_id", "unknown")
            context_override = os.environ.get("RLM_CONTEXT_DIR")
            if context_override:
                context_dir = context_override
            else:
                context_dir = _make_context_dir(task_instruction, str(instance_id))
                context_dir_owned = True

            input_row = {
                "prompt": prompt,
                "info": {"context_dir": context_dir},
            }

            client = model.openai_client()
            state = await env.rollout(
                input_row,
                client,
                model.get_inference_name(),
                sampling_args={"temperature": 0.7, "max_completion_tokens": 512},
            )

            state_error = state.get("error")
            traj.log(
                "RLMEnv state debug: "
                f"error={state_error} "
                f"stop_condition={state.get('stop_condition')} "
                f"prompt_too_long={state.get('prompt_too_long')} "
                f"final_env_response={state.get('final_env_response') is not None} "
                f"info={state.get('info')} "
                f"trajectory_len={len(state.get('trajectory', []))} "
                f"prompt_type={type(state.get('prompt'))} "
                f"prompt_preview={str(state.get('prompt'))[:200]} "
                f"raw_prompt_preview={str(state.get('raw_prompt'))[:200]}"
            )
            test_output = state.get("_test_output", "")
            test_exit = state.get("_test_exit_code")
            test_error = state.get("_test_error")
            traj.log(
                f"Test run: exit_code={test_exit} "
                f"error={test_error} "
                f"output_len={len(test_output)} "
                f"output_preview={test_output[:500]}"
            )
            if test_output:
                traj.reward = float(_compute_reward_from_logs(test_output, scenario.ds))
            else:
                traj.reward = 0.0

            messages_and_choices = _build_messages_and_choices(state)
            traj.messages_and_choices = messages_and_choices

            metrics = state.get("metrics") or {}
            traj.metrics["num_steps"] = len(state.get("trajectory", []))
            traj.metrics["finished"] = bool(state.get("is_completed", False))
            root_tool_calls = state.get("root_tool_calls", {}) or {}
            llm_batch_calls = int(root_tool_calls.get("llm_batch", 0))
            traj.metrics["num_sub_queries"] = int(
                metrics.get(
                    "sub_llm_call_count",
                    state.get("sub_llm_call_count", llm_batch_calls),
                )
            )
            traj.metrics["repl_call_count"] = int(
                metrics.get("repl_call_count", state.get("repl_call_count", 0))
            )
            traj.metrics["root_tool_call_count"] = int(
                metrics.get(
                    "root_tool_call_count", state.get("root_tool_call_count", 0)
                )
            )
            traj.metrics["llm_batch_calls"] = llm_batch_calls
            traj.metrics["is_sandbox_backend"] = backend == "sandbox"
            traj.metrics["fs_has_data"] = bool(state.get("rlm_fs_has_data", False))

    except Exception as e:
        traj.log(f"RLMEnv rollout error: {e}\n{traceback.format_exc()}")
    finally:
        if env is not None and state is not None:
            try:
                await env._cleanup(state)
            except Exception:
                pass
        if env is not None:
            try:
                await env._teardown()
            except Exception:
                pass
        if context_dir_owned and context_dir and os.path.isdir(context_dir):
            shutil.rmtree(context_dir, ignore_errors=True)

    return traj.finish()
