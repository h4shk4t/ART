"""Experiment configuration for distributed RLM training.

Every tunable for a training run lives here. Change one field = one experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExperimentConfig:
    # -- Identity --
    experiment_name: str = "default"

    # -- Model --
    model_name: str = "r2e-rlm-qwen3-14b"
    project: str = "rlm-training"
    base_model: str = "Qwen/Qwen3-14B"
    vllm_url: str = "http://localhost:8000/v1"

    # -- System prompt --
    # Import and assign a function from prompts.py, e.g.:
    #   from prompts import r2e_rlm_system_prompt
    #   config.system_prompt_fn = r2e_rlm_system_prompt
    system_prompt_fn: Callable[[], str] | None = None

    # -- Reward --
    # Import and assign a function from rewards.py, e.g.:
    #   from rewards import binary_test_reward
    #   config.reward_fn = binary_test_reward
    reward_fn: Callable[[str, dict[str, Any]], float] | None = None

    # -- Rollout --
    max_steps: int = 15
    max_completion_tokens: int = 4096
    max_sub_agent_steps: int = 8
    max_sub_agent_depth: int = 2
    max_output_chars: int = 3000
    max_context_tokens: int = 28000
    chars_per_token: int = 4

    # -- Training --
    num_epochs: int = 3
    groups_per_step: int = 8
    rollouts_per_group: int = 4
    max_concurrent: int = 32
    learning_rate: float = 5e-6

    # -- Infrastructure --
    docker_service_url: str = "http://docker-node:8000"
    repo_cache_dir: str = "/data/repo-cache"

    # -- Health limits --
    max_child_processes: int = 256
    max_tmp_usage_gb: float = 50.0
    bash_timeout: int = 90
    health_check_interval: int = 60

    # -- Trajectory logging --
    log_dir: str = "logs"
    log_trajectory_sample_rate: float = 0.0

    # -- Sub-agent prompt (for recursive sub-agents) --
    sub_agent_system_prompt_fn: Callable[[], str] | None = None

    def config_hash(self) -> str:
        """Deterministic hash of serializable config fields for reproducibility."""
        serializable = {
            k: v
            for k, v in self.__dict__.items()
            if not callable(v) and v is not None
        }
        raw = json.dumps(serializable, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_metadata(self) -> dict[str, Any]:
        """Export config as a flat dict for logging/visualization metadata."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not callable(v) and v is not None
        }
