# Rollout Module Walkthrough

## Overview

`rollout.py` implements the core REPL rollout loop for training. `debug_rollout.py`
provides a single-task interactive runner for development and debugging.

## Architecture

```
rollout()
├── 1. Copy cached repo to /tmp/rlm_rollout_{uuid}/
├── 2. Build LocalREPL with file ops + sub-agent support
├── 3. Multi-step REPL loop (all local)
│   ├── LLM completion (local vLLM)
│   ├── Extract code from response
│   ├── exec(code) in persistent REPL
│   ├── Log iteration to trajectory logger
│   └── Check if agent called finish()
├── 4. Diff modified files against cache
├── 5. Send patches to Docker service → run tests → compute reward
└── 6. Cleanup temp dir
```

## Key Components

### `Scenario`

Represents one bug-fixing task:

```python
@dataclass
class Scenario:
    instance_id: str          # e.g. "django/django__abc123def456"
    docker_image: str         # e.g. "r2egym/django:latest"
    problem_statement: str    # The bug report
    expected_output_json: str # Expected test results
    test_cmd: str             # Test command for Docker
    extra: dict | None        # Full dataset entry

    @classmethod
    def from_dataset_entry(cls, ds: dict) -> Scenario: ...
```

### `rollout()`

The main function called by `train.py` during gathering. Signature:

```python
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
```

**Network calls during the REPL loop: zero.** All file operations are local.
The only network call is at the end, when modified files are sent to the Docker
service for test execution.

### Context Window Management

`_trim_context()` keeps the conversation within `max_context_tokens`:
- Preserves the system prompt and initial bug report
- Drops the oldest mid-conversation turns first
- Inserts a note explaining that earlier turns were trimmed

### Reward Computation

After the REPL loop:
1. `get_modified_files()` diffs `work_dir` against the cached original
2. Modified files are sent to the Docker service as a tar
3. Test output is parsed by the configured `reward_fn`
4. If no files were modified or no reward function is set, reward = 0.0

## debug_rollout.py

Interactive runner for testing individual rollouts:

```bash
uv run python debug_rollout.py \
    --instance-id "django/django__abc123def456" \
    --max-steps 20 \
    --interactive  # pause between steps
```

Features:
- **Always logs** trajectories (sample_rate=1.0) to `logs/debug_{experiment}/`
- **Colored console output** showing each step's code, output, and sub-agents
- **Interactive mode** (`--interactive`) pauses between steps for human review
- **Graceful Docker failure** — if Docker service isn't running, skips reward
  computation and prints a warning

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--instance-id` | (required) | Cache instance ID |
| `--max-steps` | 15 | Max REPL steps |
| `--model` | Qwen/Qwen3-14B | Base model |
| `--model-name` | r2e-rlm-debug | ART model name |
| `--vllm-url` | http://localhost:8000/v1 | vLLM server URL |
| `--docker-url` | http://docker-node:8000 | Docker service URL |
| `--cache-dir` | /data/repo-cache | Repo cache directory |
| `--log-dir` | logs | Trajectory log directory |
| `--interactive` | false | Pause between steps |

## Customization

### Changing the REPL loop behavior

The rollout loop is intentionally straightforward — you can modify it to:

1. **Add step-level metrics**: Track which tools the model calls most, average
   code length, etc.
2. **Early stopping**: Break the loop if reward > 0 after an intermediate
   `run_tests()` call.
3. **Dynamic temperature**: Increase temperature after repeated failures.

### Changing the reward computation

Wire a different function in `ExperimentConfig.reward_fn`. The function
receives `(test_output: str, ds: dict) -> float`. See `docs/adding-rewards.md`.

### Changing the nudge behavior

When the model doesn't produce a code block, the rollout sends a "nudge" message.
Modify the nudge text in the `code is None` branch of the loop.
