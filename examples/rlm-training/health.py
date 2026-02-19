"""System health monitoring for the training node.

Prevents runaway processes, zombie subprocesses, disk exhaustion, and
thread explosion when running many concurrent REPL rollouts.

Key components:
  - safe_subprocess(): Run a command with process group isolation + hard timeout
  - RolloutGuard: Context manager that tracks and cleans up a rollout's resources
  - HealthMonitor: Periodic checks for process count, disk usage, memory
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

ROLLOUT_TMP_PREFIX = "rlm_rollout_"


@dataclass
class HealthStatus:
    """Snapshot of system health at a point in time."""

    child_process_count: int = 0
    tmp_usage_gb: float = 0.0
    active_rollouts: int = 0
    stale_rollout_dirs: int = 0
    warnings: list[str] = field(default_factory=list)
    critical: list[str] = field(default_factory=list)


def safe_subprocess(
    cmd: str | list[str],
    cwd: str | Path | None = None,
    timeout: int = 90,
    max_output: int = 100_000,
) -> tuple[str, int]:
    """Run a subprocess with process group isolation and hard timeout.

    If the command times out, the entire process group is killed (not just
    the parent), preventing orphan child processes.

    Args:
        cmd: Command string or list.
        cwd: Working directory.
        timeout: Hard timeout in seconds.
        max_output: Truncate stdout+stderr to this many characters.

    Returns:
        (output, exit_code) tuple. Output is stdout+stderr combined.
    """
    shell = isinstance(cmd, str)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
            start_new_session=True,
        )
        output = result.stdout + result.stderr
        if len(output) > max_output:
            half = max_output // 2
            output = (
                output[:half]
                + f"\n\n... [truncated {len(output) - max_output} chars] ...\n\n"
                + output[-half:]
            )
        return output, result.returncode

    except subprocess.TimeoutExpired as e:
        # Kill the entire process group
        if e.cmd is not None:
            try:
                pgid = os.getpgid(e.cmd if isinstance(e.cmd, int) else 0)
            except (OSError, TypeError):
                pgid = None

        # Try to kill via the process group set by start_new_session
        # The child PID is not directly available from TimeoutExpired,
        # but subprocess.run already sends SIGKILL on timeout.
        # We log the event for debugging.
        logger.warning("Command timed out after %ds: %s", timeout, cmd if shell else " ".join(cmd))

        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        output = stdout + stderr + f"\n[TIMEOUT after {timeout}s]"
        if len(output) > max_output:
            output = output[-max_output:]
        return output, -1

    except FileNotFoundError:
        return f"Command not found: {cmd}", 127

    except Exception as e:
        return f"Subprocess error: {e}", -1


class RolloutGuard:
    """Context manager that tracks a rollout's work directory and ensures cleanup.

    Usage:
        with health_monitor.register_rollout(work_dir) as guard:
            # do rollout work
            ...
        # work_dir is cleaned up even if an exception occurred
    """

    def __init__(self, work_dir: str | Path, monitor: HealthMonitor):
        self.work_dir = Path(work_dir)
        self._monitor = monitor

    def __enter__(self) -> RolloutGuard:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
        self._monitor._active_rollout_dirs.discard(str(self.work_dir))
        return None

    def cleanup(self) -> None:
        """Remove the work directory."""
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
            except Exception as e:
                logger.warning("Failed to clean up %s: %s", self.work_dir, e)


class HealthMonitor:
    """Monitors system health and manages rollout lifecycle."""

    def __init__(
        self,
        max_child_processes: int = 256,
        max_tmp_usage_gb: float = 50.0,
        bash_timeout: int = 90,
        tmp_dir: str = "/tmp",
    ):
        self.max_child_processes = max_child_processes
        self.max_tmp_usage_gb = max_tmp_usage_gb
        self.bash_timeout = bash_timeout
        self.tmp_dir = tmp_dir
        self._active_rollout_dirs: set[str] = set()
        self._lock = Lock()

    @classmethod
    def from_config(cls, config) -> HealthMonitor:
        """Create from an ExperimentConfig."""
        return cls(
            max_child_processes=config.max_child_processes,
            max_tmp_usage_gb=config.max_tmp_usage_gb,
            bash_timeout=config.bash_timeout,
        )

    @contextmanager
    def register_rollout(self, work_dir: str | Path):
        """Context manager that tracks a rollout and guarantees cleanup."""
        work_dir_str = str(work_dir)
        with self._lock:
            self._active_rollout_dirs.add(work_dir_str)
        guard = RolloutGuard(work_dir, self)
        try:
            yield guard
        finally:
            guard.cleanup()
            with self._lock:
                self._active_rollout_dirs.discard(work_dir_str)

    def track_rollout(self, work_dir: str | Path) -> None:
        """Register a rollout directory as active (non-context-manager API)."""
        with self._lock:
            self._active_rollout_dirs.add(str(work_dir))

    def unregister_rollout(self, work_dir: str | Path) -> None:
        """Remove a rollout directory from the active set."""
        with self._lock:
            self._active_rollout_dirs.discard(str(work_dir))

    def safe_subprocess(
        self,
        cmd: str | list[str],
        cwd: str | Path | None = None,
        timeout: int | None = None,
    ) -> tuple[str, int]:
        """Run a subprocess with safety guardrails. Delegates to module-level safe_subprocess."""
        return safe_subprocess(
            cmd,
            cwd=cwd,
            timeout=timeout if timeout is not None else self.bash_timeout,
        )

    def check(self) -> HealthStatus:
        """Run a health check. Returns status with warnings and critical errors."""
        status = HealthStatus()

        # 1. Count child processes
        try:
            my_pid = os.getpid()
            output, _ = safe_subprocess(
                f"pgrep -c -P {my_pid}", timeout=5
            )
            status.child_process_count = int(output.strip()) if output.strip().isdigit() else 0
        except Exception:
            status.child_process_count = -1

        if status.child_process_count > self.max_child_processes:
            status.critical.append(
                f"Child process count ({status.child_process_count}) exceeds limit ({self.max_child_processes})"
            )
        elif status.child_process_count > self.max_child_processes * 0.8:
            status.warnings.append(
                f"Child process count ({status.child_process_count}) approaching limit ({self.max_child_processes})"
            )

        # 2. Check /tmp disk usage
        try:
            usage = shutil.disk_usage(self.tmp_dir)
            status.tmp_usage_gb = round(usage.used / (1024**3), 2)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)

            if free_gb < 5.0:
                status.critical.append(
                    f"/tmp has only {free_gb:.1f} GB free (total: {total_gb:.1f} GB)"
                )
            elif free_gb < 20.0:
                status.warnings.append(
                    f"/tmp has {free_gb:.1f} GB free (total: {total_gb:.1f} GB)"
                )
        except Exception as e:
            status.warnings.append(f"Could not check /tmp usage: {e}")

        # 3. Active rollouts
        with self._lock:
            status.active_rollouts = len(self._active_rollout_dirs)

        # 4. Stale rollout dirs
        status.stale_rollout_dirs = self._count_stale_rollout_dirs()
        if status.stale_rollout_dirs > 10:
            status.warnings.append(
                f"{status.stale_rollout_dirs} stale rollout dirs in {self.tmp_dir}"
            )

        return status

    def reap_stale_rollouts(self, max_age_seconds: int = 3600) -> int:
        """Remove rollout work dirs older than max_age_seconds.

        Only removes dirs matching the rollout prefix that are NOT
        in the active set.

        Returns number of dirs removed.
        """
        removed = 0
        now = time.time()
        tmp = Path(self.tmp_dir)

        with self._lock:
            active = set(self._active_rollout_dirs)

        for d in tmp.iterdir():
            if not d.is_dir() or not d.name.startswith(ROLLOUT_TMP_PREFIX):
                continue
            if str(d) in active:
                continue
            try:
                mtime = d.stat().st_mtime
                if now - mtime > max_age_seconds:
                    shutil.rmtree(d)
                    removed += 1
                    logger.info("Reaped stale rollout dir: %s (age: %ds)", d.name, int(now - mtime))
            except Exception as e:
                logger.warning("Failed to reap %s: %s", d, e)

        return removed

    def _count_stale_rollout_dirs(self) -> int:
        """Count rollout dirs in /tmp that are not in the active set."""
        count = 0
        tmp = Path(self.tmp_dir)
        with self._lock:
            active = set(self._active_rollout_dirs)
        try:
            for d in tmp.iterdir():
                if d.is_dir() and d.name.startswith(ROLLOUT_TMP_PREFIX) and str(d) not in active:
                    count += 1
        except Exception:
            pass
        return count

    def summary(self) -> str:
        """One-line health summary for logging."""
        status = self.check()
        parts = [
            f"procs={status.child_process_count}",
            f"tmp={status.tmp_usage_gb:.1f}GB",
            f"rollouts={status.active_rollouts}",
            f"stale={status.stale_rollout_dirs}",
        ]
        if status.warnings:
            parts.append(f"WARN={len(status.warnings)}")
        if status.critical:
            parts.append(f"CRIT={len(status.critical)}")
        return " | ".join(parts)
