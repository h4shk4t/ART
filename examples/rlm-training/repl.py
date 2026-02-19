"""Local REPL with file operation helpers for RLM training rollouts.

Mirrors the interface of ART/examples/r2e-gym-rlm/repl.py (DockerREPL)
but all file operations run locally on the training node via subprocess,
not inside Docker containers. Only run_tests() hits the network.

Key differences from DockerREPL:
  - File ops use subprocess.run(cmd, cwd=work_dir) instead of env.runtime.run()
  - bash() goes through health.safe_subprocess() for timeout + process group isolation
  - execute() returns ExecResult with structured data for the trajectory logger
  - Sub-agents (llm_query) create a child LocalREPL sharing the same work_dir
"""

from __future__ import annotations

import asyncio
import io
import re
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nest_asyncio
from openai.types.chat.chat_completion import Choice

import art
from art.trajectories import History

from health import HealthMonitor, safe_subprocess

nest_asyncio.apply()

# Safe builtins — blocks dangerous operations that could hang or escape the sandbox.
# Ported from rlm/rlm/environments/local_repl.py.
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked — these can hang, escape the sandbox, or cause recursion
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


@dataclass
class SubAgentCall:
    """Record of a single sub-agent invocation, for trajectory logging."""

    prompt: str
    response: str
    execution_time: float = 0.0


@dataclass
class ExecResult:
    """Structured result from executing code in the REPL."""

    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    sub_agent_calls: list[SubAgentCall] = field(default_factory=list)


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n"
        + text[-half:]
    )


def extract_python_code(text: str) -> str | None:
    """Extract the first ```python or ```repl block from text."""
    match = re.search(r"```(?:python|repl)\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _messages_from_choices(messages_and_choices: list) -> list[dict]:
    """Convert a mixed list of dicts and Choice objects to plain message dicts."""
    result = []
    for item in messages_and_choices:
        if isinstance(item, Choice):
            result.append({"role": "assistant", "content": item.message.content or ""})
        else:
            result.append(item)
    return result


class LocalREPL:
    """Exec-based REPL with local file operations for RLM training.

    All file operations (ls, read, grep, bash, apply_patch) run locally on
    the training node against work_dir. The REPL state persists across
    execute() calls. Sub-agents get their own globals but share the same
    work_dir and trajectory (for joint GRPO training).
    """

    def __init__(
        self,
        work_dir: str | Path,
        model: art.Model,
        trajectory: art.Trajectory,
        health: HealthMonitor,
        *,
        sub_agent_system_prompt: str = "",
        max_output_chars: int = 3000,
        max_sub_agent_steps: int = 8,
        max_completion_tokens: int = 4000,
        depth: int = 0,
        max_depth: int = 2,
    ) -> None:
        self.work_dir = Path(work_dir)
        self._model = model
        self._trajectory = trajectory
        self._health = health
        self._finished = False
        self._depth = depth
        self._max_depth = max_depth
        self._max_output_chars = max_output_chars
        self._max_sub_agent_steps = max_sub_agent_steps
        self._max_completion_tokens = max_completion_tokens
        self._sub_agent_system_prompt = sub_agent_system_prompt

        self._pending_sub_calls: list[SubAgentCall] = []
        self._exec_lock = threading.Lock()

        self._tool_bindings: dict[str, Any] = {
            "ls": self.ls,
            "read": self.read,
            "grep": self.grep,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
            "bash": self.bash,
            "finish": self.finish,
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
        }

        self._globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__repl__",
            **self._tool_bindings,
        }

    def is_finished(self) -> bool:
        return self._finished

    # ------------------------------------------------------------------
    # Local file operations
    # ------------------------------------------------------------------

    def ls(self, path: str = ".", max_depth: int = 2) -> str:
        """List files in the local work directory."""
        target = self._resolve_path(path)
        output, _ = safe_subprocess(
            f"find {target} -maxdepth {max_depth} -type f",
            cwd=self.work_dir,
            timeout=30,
        )
        return self._truncate(output)

    def read(
        self, path: str, start: int | None = None, end: int | None = None
    ) -> str:
        """Read a file (or line range) from the local work directory."""
        target = self._resolve_path(path)
        if start is not None and end is not None:
            cmd = f"sed -n '{start},{end}p' {target} | cat -n"
        elif start is not None:
            cmd = f"sed -n '{start},$p' {target} | cat -n"
        else:
            cmd = f"cat -n {target}"
        output, _ = safe_subprocess(cmd, cwd=self.work_dir, timeout=30)
        return self._truncate(output)

    def grep(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*.py",
    ) -> str:
        """Search for a pattern in the local work directory."""
        target = self._resolve_path(path)
        output, _ = safe_subprocess(
            f"grep -rn --include='{file_pattern}' '{pattern}' {target}",
            cwd=self.work_dir,
            timeout=30,
        )
        return self._truncate(output)

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff patch to the local repo copy."""
        patch = textwrap.dedent(patch).strip() + "\n"
        try:
            result = subprocess.run(
                ["patch", "-p1", "--no-backup-if-mismatch"],
                input=patch,
                capture_output=True,
                text=True,
                cwd=self.work_dir,
                timeout=30,
            )
            output = result.stdout + result.stderr
            return self._truncate(f"{output}\nExit code: {result.returncode}")
        except Exception as e:
            return f"Patch failed: {e}"

    def run_tests(self, timeout: int = 300) -> str:
        """Test execution happens at reward time via the remote Docker service.

        The local repo copy doesn't have the full test environment installed,
        so running tests locally would produce misleading results. The real
        test suite runs in a Docker container after the REPL loop finishes.
        """
        return (
            "Note: Full test execution runs at reward time via Docker. "
            "The local copy lacks the installed test environment. "
            "Use bash('python -m pytest <specific_test> -x') for quick "
            "syntax checks, but results may differ from the real test suite."
        )

    def bash(self, cmd: str, timeout: int = 90) -> str:
        """Run a bash command in the local work directory."""
        output, code = self._health.safe_subprocess(
            cmd, cwd=self.work_dir, timeout=timeout
        )
        return self._truncate(f"{output}\nExit code: {code}")

    def finish(self) -> str:
        """Signal that the agent is done."""
        self._finished = True
        return "Task marked as finished."

    # ------------------------------------------------------------------
    # Recursive sub-agents
    # ------------------------------------------------------------------

    def llm_query(self, prompt: str, max_steps: int | None = None) -> str:
        """Spawn a recursive sub-agent with its own REPL loop.

        The sub-agent shares the same work_dir (read-only for sub-agents:
        no finish, apply_patch, or run_tests). Its conversation is stored
        in trajectory.additional_histories for joint GRPO training.
        """
        if max_steps is None:
            max_steps = self._max_sub_agent_steps

        if self._depth >= self._max_depth:
            return (
                f"[Maximum recursion depth ({self._max_depth}) reached. "
                f"Cannot spawn sub-agent. Please answer directly.]"
            )

        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(
                self._run_sub_agent(prompt, max_steps)
            )
        except RuntimeError:
            return asyncio.run(self._run_sub_agent(prompt, max_steps))

    def llm_query_batched(
        self, prompts: list[str], max_steps: int | None = None
    ) -> list[str]:
        """Run multiple sub-agent queries (sequential for safety)."""
        if not isinstance(prompts, list):
            raise ValueError("llm_query_batched expects a list of prompts.")
        return [self.llm_query(p, max_steps=max_steps) for p in prompts]

    async def _run_sub_agent(self, prompt: str, max_steps: int) -> str:
        """Multi-step sub-agent loop with its own REPL."""
        t0 = time.time()

        sub_repl = LocalREPL(
            work_dir=self.work_dir,
            model=self._model,
            trajectory=self._trajectory,
            health=self._health,
            sub_agent_system_prompt=self._sub_agent_system_prompt,
            max_output_chars=self._max_output_chars,
            max_sub_agent_steps=self._max_sub_agent_steps,
            max_completion_tokens=self._max_completion_tokens,
            depth=self._depth + 1,
            max_depth=self._max_depth,
        )
        del sub_repl._globals["finish"]
        del sub_repl._globals["apply_patch"]
        del sub_repl._globals["run_tests"]

        messages_and_choices: list = [
            {"role": "system", "content": self._sub_agent_system_prompt},
            {"role": "user", "content": prompt},
        ]

        final_answer = "[Sub-agent did not produce a final answer]"

        for _ in range(max_steps):
            client = self._model.openai_client()
            response = await client.chat.completions.create(
                model=self._model.get_inference_name(),
                messages=_messages_from_choices(messages_and_choices),
                temperature=1.0,
                max_completion_tokens=self._max_completion_tokens,
            )
            choice = response.choices[0]
            messages_and_choices.append(choice)

            assistant_text = choice.message.content or ""
            code = extract_python_code(assistant_text)

            if code is None:
                final_answer = assistant_text
                break

            observation = sub_repl.execute(code)
            messages_and_choices.append({
                "role": "user",
                "content": f"Output:\n```\n{observation.stdout}\n```",
            })
        else:
            last_text = ""
            for item in reversed(messages_and_choices):
                if isinstance(item, Choice):
                    last_text = item.message.content or ""
                    break
            final_answer = last_text or final_answer

        history = History(messages_and_choices=messages_and_choices)
        self._trajectory.additional_histories.append(history)

        elapsed = time.time() - t0
        call = SubAgentCall(
            prompt=prompt,
            response=final_answer,
            execution_time=elapsed,
        )
        self._pending_sub_calls.append(call)

        return self._truncate(final_answer)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecResult:
        """Execute model-generated Python in the persistent REPL.

        Thread-safe: uses a lock around stdout/stderr redirection so
        concurrent rollouts (via asyncio.to_thread) don't interleave output.
        After execution, tool bindings are restored so the model can't
        permanently overwrite helpers like ls, read, llm_query, etc.
        """
        self._pending_sub_calls = []
        t0 = time.time()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        stderr_text = ""

        with self._exec_lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = stdout_buf
                sys.stderr = stderr_buf
                exec(code, self._globals)  # noqa: S102
            except Exception:
                stderr_text = traceback.format_exc()
                stdout_buf.write(stderr_text)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        self._restore_scaffold()

        elapsed = time.time() - t0
        stdout = self._truncate(stdout_buf.getvalue())
        if stderr_buf.getvalue():
            stderr_text = stderr_text or stderr_buf.getvalue()

        return ExecResult(
            stdout=stdout,
            stderr=stderr_text,
            execution_time=elapsed,
            sub_agent_calls=list(self._pending_sub_calls),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_scaffold(self) -> None:
        """Re-bind all tool functions after exec() so the model can't
        permanently overwrite helpers like ls, read, llm_query, etc."""
        for name, fn in self._tool_bindings.items():
            self._globals[name] = fn

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to work_dir. Prevents directory traversal."""
        resolved = (self.work_dir / path).resolve()
        if not str(resolved).startswith(str(self.work_dir.resolve())):
            return str(self.work_dir)
        return str(resolved)

    def _truncate(self, text: str) -> str:
        return _truncate(text, limit=self._max_output_chars)
