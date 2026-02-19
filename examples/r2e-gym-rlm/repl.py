"""Docker REPL wrapper with helper functions for R2E-Gym environments.

Implements the RLM (Recursive Language Model) execution model:
- The model writes Python code that is exec()'d in a **persistent** REPL
- Helper functions (ls, read, grep, bash, etc.) call into Docker containers
- sub_query() spawns a recursive sub-agent with its **own** multi-step REPL
  loop and codebase access, enabling hierarchical exploration
- All sub-agent conversations are stored as additional_histories for joint
  GRPO training across the full recursive execution tree
"""

from __future__ import annotations

import asyncio
import io
import re
import traceback
from contextlib import redirect_stdout

import nest_asyncio
from openai.types.chat.chat_completion import Choice

import art
from art.trajectories import History

from r2egym.agenthub.environment.env import RepoEnv

nest_asyncio.apply()

MAX_OUTPUT_CHARS = 3000
DEFAULT_MAX_DEPTH = 2
DEFAULT_SUB_STEPS = 8

SUB_AGENT_SYSTEM_PROMPT = """\
You are a coding assistant helping analyze a repository to answer a question.

You have access to a Python REPL with the following helper functions:

- ls(path="/testbed", max_depth=2) -> str: List files.
- read(path, start=None, end=None) -> str: Read a file (or line range).
- grep(pattern, path="/testbed", file_pattern="*.py") -> str: Search for a pattern.
- bash(cmd, timeout=90) -> str: Run a bash command.
- sub_query(prompt, max_steps=5) -> str: Delegate a sub-question to another agent.

Write Python code in a ```python or ```repl fenced code block. Use print() to see results.

When you have gathered enough information, respond with your final answer as \
plain text (NO code block). This signals that you are done.
"""


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n"
        + text[-half:]
    )


def extract_python_code(text: str) -> str | None:
    """Extract the first ```python ... ``` or ```repl ... ``` block from text."""
    match = re.search(r"```(?:python|repl)\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _messages_from_choices(messages_and_choices: list) -> list[dict]:
    """Convert a mixed list of dicts and Choice objects to plain message dicts."""
    messages = []
    for item in messages_and_choices:
        if isinstance(item, Choice):
            messages.append({"role": "assistant", "content": item.message.content or ""})
        else:
            messages.append(item)
    return messages


class DockerREPL:
    """Wraps a RepoEnv to expose helper functions as globals for exec().

    The REPL state (all Python variables) persists across execute() calls,
    matching how a real interactive Python session works. This lets the model
    build up state incrementally: defining variables in step 1 and using them
    in step 5.

    For sub-agents (spawned by sub_query), a **separate** DockerREPL is created
    with its own globals dict but the **same** Docker container, so sub-agents
    can independently navigate the codebase without polluting the parent's state.
    """

    def __init__(
        self,
        env: RepoEnv,
        model: art.Model,
        trajectory: art.Trajectory,
        *,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self.env = env
        self.runtime = env.runtime
        self._model = model
        self._trajectory = trajectory
        self._finished = False
        self._depth = depth
        self._max_depth = max_depth

        # Persistent globals dict — state carries across execute() calls
        self._globals: dict = {
            "ls": self.ls,
            "read": self.read,
            "grep": self.grep,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
            "bash": self.bash,
            "finish": self.finish,
            "sub_query": self.sub_query,
            # RLM-compatible aliases (used by official prompts)
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
        }

    def is_finished(self) -> bool:
        return self._finished

    # ------------------------------------------------------------------
    # Helper functions (exposed as globals to exec'd code)
    # ------------------------------------------------------------------

    def ls(self, path: str = "/testbed", max_depth: int = 2) -> str:
        """List files in the container."""
        output, _ = self.runtime.run(
            f"find {path} -maxdepth {max_depth} -type f", timeout=30
        )
        return _truncate(output)

    def read(
        self, path: str, start: int | None = None, end: int | None = None
    ) -> str:
        """Read a file (or line range) from the container."""
        if start is not None and end is not None:
            cmd = f"sed -n '{start},{end}p' {path} | cat -n"
        elif start is not None:
            cmd = f"sed -n '{start},$p' {path} | cat -n"
        else:
            cmd = f"cat -n {path}"
        output, _ = self.runtime.run(cmd, timeout=30)
        return _truncate(output)

    def grep(
        self,
        pattern: str,
        path: str = "/testbed",
        file_pattern: str = "*.py",
    ) -> str:
        """Grep for a pattern in the container."""
        output, _ = self.runtime.run(
            f"grep -rn --include='{file_pattern}' '{pattern}' {path}",
            timeout=30,
        )
        return _truncate(output)

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch to the repo."""
        output, error_code = self.runtime.apply_patch(patch)
        return _truncate(f"{output}\nExit code: {error_code}")

    def run_tests(self, timeout: int = 300) -> str:
        """Run the test suite in the container."""
        output, error_code = self.runtime.run_tests(timeout=timeout)
        return _truncate(f"{output}\nExit code: {error_code}")

    def bash(self, cmd: str, timeout: int = 90) -> str:
        """Run an arbitrary bash command in the container."""
        output, error_code = self.runtime.run(cmd, timeout=timeout)
        return _truncate(f"{output}\nExit code: {error_code}")

    def finish(self) -> str:
        """Signal that the agent is done."""
        self._finished = True
        return "Task marked as finished."

    # ------------------------------------------------------------------
    # Recursive sub-agent (the core RLM mechanism)
    # ------------------------------------------------------------------

    def sub_query(self, prompt: str, max_steps: int = DEFAULT_SUB_STEPS) -> str:
        """Spawn a recursive sub-agent with its own REPL and codebase access.

        The sub-agent gets:
        - Its own persistent Python REPL (separate globals from parent)
        - The same Docker container (same codebase access)
        - A multi-step loop: it writes code, sees output, writes more code...
        - The ability to spawn further sub-agents (up to max_depth)

        The sub-agent's full conversation is stored in
        trajectory.additional_histories for joint GRPO training.

        Returns the sub-agent's final text answer.
        """
        if self._depth >= self._max_depth:
            return (
                f"[Maximum recursion depth ({self._max_depth}) reached. "
                f"Cannot spawn sub-agent. Please answer directly.]"
            )

        # Handle async: we may be in a thread (via asyncio.to_thread) with no
        # running loop, or inside a nested loop (recursive sub_query).
        try:
            loop = asyncio.get_running_loop()
            # Already in a loop (recursive case) — nest_asyncio allows this
            return loop.run_until_complete(
                self._run_sub_agent(prompt, max_steps)
            )
        except RuntimeError:
            # No running loop (thread case) — create one
            return asyncio.run(self._run_sub_agent(prompt, max_steps))

    async def _run_sub_agent(self, prompt: str, max_steps: int) -> str:
        """Run a multi-step sub-agent with its own REPL.

        The sub-agent loop mirrors the root rollout loop: the model writes
        Python code, sees output, and iterates. When it responds with plain
        text (no code block), that's its final answer.
        """
        # Sub-agent gets its own REPL (separate state, same Docker container)
        sub_repl = DockerREPL(
            self.env,
            self._model,
            self._trajectory,  # shares trajectory for additional_histories
            depth=self._depth + 1,
            max_depth=self._max_depth,
        )
        # Remove finish/apply_patch/run_tests from sub-agent — only root
        # should modify the repo and signal completion
        del sub_repl._globals["finish"]
        del sub_repl._globals["apply_patch"]
        del sub_repl._globals["run_tests"]

        messages_and_choices: list = [
            {"role": "system", "content": SUB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        final_answer = "[Sub-agent did not produce a final answer]"

        for _ in range(max_steps):
            client = self._model.openai_client()
            response = await client.chat.completions.create(
                model=self._model.get_inference_name(),
                messages=_messages_from_choices(messages_and_choices),  # type: ignore[arg-type]
                temperature=1.0,
                max_completion_tokens=4000,
            )
            choice = response.choices[0]
            messages_and_choices.append(choice)

            assistant_text = choice.message.content or ""
            code = extract_python_code(assistant_text)

            if code is None:
                # No code block → this is the final answer
                final_answer = assistant_text
                break

            # Execute code in sub-agent's own REPL
            observation = sub_repl.execute(code)
            messages_and_choices.append({
                "role": "user",
                "content": f"Output:\n```\n{observation}\n```",
            })
        else:
            # Ran out of steps — use last assistant text as answer
            last_text = ""
            for item in reversed(messages_and_choices):
                if isinstance(item, Choice):
                    last_text = item.message.content or ""
                    break
            final_answer = last_text or final_answer

        # Store the full sub-agent conversation for joint GRPO training.
        # This is the key insight: every sub-call's policy decisions get
        # gradient signal from the root trajectory's binary reward.
        history = History(messages_and_choices=messages_and_choices)
        self._trajectory.additional_histories.append(history)

        return _truncate(final_answer)

    def llm_query(self, prompt: str, max_steps: int = DEFAULT_SUB_STEPS) -> str:
        """Alias for sub_query to match official RLM prompts."""
        return self.sub_query(prompt, max_steps=max_steps)

    def llm_query_batched(
        self, prompts: list[str], max_steps: int = DEFAULT_SUB_STEPS
    ) -> list[str]:
        """Simple batched sub-query helper (sequential for safety)."""
        if not isinstance(prompts, list):
            raise ValueError("llm_query_batched expects a list of prompts.")
        return [self.sub_query(p, max_steps=max_steps) for p in prompts]

    # ------------------------------------------------------------------
    # Code execution (persistent REPL)
    # ------------------------------------------------------------------

    def execute(self, code: str) -> str:
        """Execute model-generated Python code in the persistent REPL.

        The globals dict persists across calls, so variables defined in step N
        are available in step N+1 — just like a real Python REPL. This lets
        the model build up state incrementally (e.g., accumulating results from
        multiple file reads into a data structure).

        Captures stdout and returns it. Tracebacks are returned as strings.
        """
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self._globals)  # noqa: S102
        except Exception:
            buf.write(traceback.format_exc())
        return _truncate(buf.getvalue())
