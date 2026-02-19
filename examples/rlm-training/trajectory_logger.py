"""Trajectory logger compatible with the RLM visualizer's JSONL format.

Emits .jsonl files that can be directly uploaded to the RLM visualizer
(rlm/visualizer/). Each file contains one metadata line followed by one
line per REPL iteration.

Extended metadata fields (training_step, instance_id, reward, experiment_name)
are ignored by the current visualizer but enable future UIs to visualize
trajectory evolution across training steps.

Usage:
    logger = TrajectoryLogger(log_dir="logs", experiment_name="exp1")

    # At start of rollout
    session = logger.start_session(
        instance_id="django__abc123",
        training_step=42,
        model_name="Qwen/Qwen3-14B",
        config_metadata={...},
    )

    # After each REPL step
    session.log_iteration(
        iteration=1,
        prompt=messages,
        response=assistant_text,
        code="print(ls())",
        exec_result=exec_result,
        final_answer=None,
        iteration_time=1.23,
    )

    # At end of rollout
    session.finalize(reward=1.0, total_steps=5)
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repl import ExecResult, SubAgentCall


@dataclass
class LoggedCodeBlock:
    """A single code execution block, matching the visualizer's CodeBlock type."""

    code: str
    result: LoggedREPLResult


@dataclass
class LoggedREPLResult:
    """Execution result matching the visualizer's REPLResult type."""

    stdout: str = ""
    stderr: str = ""
    locals: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    rlm_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None


def _sub_call_to_rlm_chat(call: SubAgentCall, model_name: str = "") -> dict[str, Any]:
    """Convert a SubAgentCall to the visualizer's RLMChatCompletion format."""
    return {
        "root_model": model_name,
        "prompt": call.prompt,
        "response": call.response,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "execution_time": call.execution_time,
    }


def exec_result_to_repl_result(
    result: ExecResult,
    model_name: str = "",
    final_answer: str | None = None,
) -> LoggedREPLResult:
    """Convert an ExecResult from the REPL to LoggedREPLResult for logging."""
    rlm_calls = [
        _sub_call_to_rlm_chat(c, model_name) for c in result.sub_agent_calls
    ]
    return LoggedREPLResult(
        stdout=result.stdout,
        stderr=result.stderr,
        locals={},
        execution_time=result.execution_time,
        rlm_calls=rlm_calls,
        final_answer=final_answer,
    )


class TrajectorySession:
    """Represents a single rollout's trajectory being logged.

    Collects iteration records and writes them to a JSONL file.
    Thread-safe for a single rollout (not shared across rollouts).
    """

    def __init__(self, path: Path, model_name: str) -> None:
        self._path = path
        self._model_name = model_name
        self._iterations: list[dict[str, Any]] = []

    def log_metadata(
        self,
        instance_id: str,
        training_step: int,
        model_name: str,
        config_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write the metadata line (first line of the JSONL file)."""
        meta = {
            "type": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "root_model": model_name,
            "max_depth": config_metadata.get("max_sub_agent_depth", 2) if config_metadata else 2,
            "max_iterations": config_metadata.get("max_steps", 15) if config_metadata else 15,
            "backend": "local",
            "backend_kwargs": None,
            "environment_type": "local_repl",
            "environment_kwargs": None,
            "other_backends": None,
            # Extended fields for future training-evolution UI
            "instance_id": instance_id,
            "training_step": training_step,
            "experiment_name": config_metadata.get("experiment_name", "") if config_metadata else "",
        }
        self._write_line(meta)

    def log_iteration(
        self,
        iteration: int,
        prompt: list[dict[str, str]],
        response: str,
        code: str | None,
        exec_result: ExecResult | None = None,
        final_answer: str | list[str] | None = None,
        iteration_time: float | None = None,
    ) -> None:
        """Log a single REPL iteration."""
        code_blocks = []
        if code is not None and exec_result is not None:
            repl_result = exec_result_to_repl_result(
                exec_result, self._model_name, final_answer=None
            )
            code_blocks.append({
                "code": code,
                "result": asdict(repl_result),
            })

        record = {
            "type": "iteration",
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "response": response,
            "code_blocks": code_blocks,
            "final_answer": final_answer,
            "iteration_time": iteration_time,
        }
        self._iterations.append(record)
        self._write_line(record)

    def finalize(self, reward: float, total_steps: int) -> None:
        """Write a summary line at the end (extended, ignored by current visualizer)."""
        summary = {
            "type": "summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reward": reward,
            "total_steps": total_steps,
            "total_iterations": len(self._iterations),
        }
        self._write_line(summary)

    def _write_line(self, obj: dict[str, Any]) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(obj, default=str) + "\n")


class TrajectoryLogger:
    """Factory for creating per-rollout TrajectorySession objects.

    Controls sampling rate and directory layout:
        {log_dir}/{experiment_name}/step_{training_step}/
            {instance_id}_{timestamp}_{random}.jsonl
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "default",
        sample_rate: float = 0.0,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._experiment_name = experiment_name
        self._sample_rate = sample_rate

    @classmethod
    def from_config(cls, config: Any) -> TrajectoryLogger:
        """Create from an ExperimentConfig."""
        return cls(
            log_dir=config.log_dir,
            experiment_name=config.experiment_name,
            sample_rate=config.log_trajectory_sample_rate,
        )

    def should_log(self) -> bool:
        """Decide whether to log this rollout based on sample_rate."""
        if self._sample_rate <= 0:
            return False
        if self._sample_rate >= 1.0:
            return True
        return random.random() < self._sample_rate

    def start_session(
        self,
        instance_id: str,
        training_step: int,
        model_name: str,
        config_metadata: dict[str, Any] | None = None,
    ) -> TrajectorySession | None:
        """Start a new logging session for one rollout.

        Returns None if this rollout was not selected for logging
        (based on sample_rate).
        """
        if not self.should_log():
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand = f"{random.randint(0, 0xFFFF):04x}"
        safe_id = instance_id.replace("/", "_").replace(" ", "_")

        step_dir = self._log_dir / self._experiment_name / f"step_{training_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{safe_id}_{ts}_{rand}.jsonl"
        path = step_dir / filename

        session = TrajectorySession(path, model_name)
        session.log_metadata(
            instance_id=instance_id,
            training_step=training_step,
            model_name=model_name,
            config_metadata=config_metadata,
        )
        return session


class NullSession:
    """No-op session for when logging is disabled. Drop-in for TrajectorySession."""

    def log_iteration(self, **kwargs: Any) -> None:
        pass

    def finalize(self, **kwargs: Any) -> None:
        pass
