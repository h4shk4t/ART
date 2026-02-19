# Trajectory Logging Walkthrough

## Overview

`trajectory_logger.py` emits `.jsonl` files compatible with the existing RLM
visualizer (`rlm/visualizer/`). Each file captures one rollout's full trajectory:
every REPL iteration, the model's responses, executed code, and sub-LLM calls.

## Quick Start

### Enable logging

In your `ExperimentConfig`:

```python
config = ExperimentConfig(
    log_dir="logs",
    experiment_name="baseline_v1",
    log_trajectory_sample_rate=0.1,  # log 10% of rollouts
)
```

### View logged trajectories

```bash
cd rlm && make visualizer
```

Then upload a `.jsonl` file from `logs/baseline_v1/step_*/`.

## Architecture

```
TrajectoryLogger
├── should_log()          # Bernoulli sampling based on sample_rate
├── start_session()       # Creates a JSONL file + writes metadata line
│   └── TrajectorySession
│       ├── log_iteration()   # Writes one line per REPL step
│       └── finalize()        # Writes summary line with reward
└── NullSession               # No-op drop-in when logging is disabled
```

## File Layout

```
logs/
└── {experiment_name}/
    ├── step_0/
    │   ├── django__abc123_20260219_143000_a1b2.jsonl
    │   └── flask__def456_20260219_143005_c3d4.jsonl
    ├── step_1/
    │   └── ...
    └── step_N/
        └── ...
```

This layout enables future UIs to compare trajectories for the same instance
across training steps, visualizing how the model's strategy evolves.

## JSONL Format

Each `.jsonl` file has three types of lines:

### 1. Metadata (first line)

```json
{
  "type": "metadata",
  "timestamp": "2026-02-19T14:30:00Z",
  "root_model": "Qwen/Qwen3-14B",
  "max_depth": 2,
  "max_iterations": 15,
  "backend": "local",
  "environment_type": "local_repl",
  "instance_id": "django__abc123def456",
  "training_step": 42,
  "experiment_name": "baseline_v1"
}
```

The visualizer reads `root_model`, `max_depth`, `max_iterations`. The extended
fields (`instance_id`, `training_step`, `experiment_name`) are for future use.

### 2. Iteration (one per REPL step)

```json
{
  "type": "iteration",
  "iteration": 1,
  "timestamp": "2026-02-19T14:30:01Z",
  "prompt": [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "Bug report: ..."}
  ],
  "response": "Let me examine the code:\n```python\nprint(ls())\n```",
  "code_blocks": [
    {
      "code": "print(ls())",
      "result": {
        "stdout": "./src/main.py\n./src/utils.py\n",
        "stderr": "",
        "locals": {},
        "execution_time": 0.05,
        "rlm_calls": [
          {
            "root_model": "Qwen/Qwen3-14B",
            "prompt": "What does the function do?",
            "response": "It returns the sum of a and b.",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "execution_time": 0.8
          }
        ]
      }
    }
  ],
  "final_answer": null,
  "iteration_time": 1.23
}
```

### 3. Summary (last line)

```json
{
  "type": "summary",
  "timestamp": "2026-02-19T14:31:00Z",
  "reward": 1.0,
  "total_steps": 5,
  "total_iterations": 5
}
```

The current visualizer ignores `summary` lines. They're for future dashboards.

## Sampling

Setting `sample_rate=0.1` means ~10% of rollouts are logged. This is important
because:

- Full logging at 32 concurrent rollouts generates substantial I/O
- During training, you typically want a representative sample, not every rollout
- For debugging, set `sample_rate=1.0` (or use `debug_rollout.py` which always logs)

## Integration Points

### In rollout.py

```python
logger = TrajectoryLogger.from_config(config)
session = logger.start_session(
    instance_id=scenario.instance_id,
    training_step=step,
    model_name=config.model_name,
    config_metadata=config.to_metadata(),
)
if session is None:
    session = NullSession()

for step_num in range(max_steps):
    # ... model completion ...
    session.log_iteration(
        iteration=step_num + 1,
        prompt=messages,
        response=assistant_text,
        code=code,
        exec_result=exec_result,
        final_answer=final_answer if repl.is_finished() else None,
        iteration_time=elapsed,
    )

session.finalize(reward=reward, total_steps=step_num + 1)
```

### Custom metadata

Add any fields to `config_metadata` — they'll be written to the metadata line:

```python
session = logger.start_session(
    ...,
    config_metadata={
        **config.to_metadata(),
        "hypothesis": "longer context window improves patch quality",
        "git_sha": "abc123",
    },
)
```

## Customization

### Adding fields to iterations

Extend `log_iteration()` with keyword arguments. Unknown fields are written
to the JSONL and ignored by the current visualizer.

### Token tracking

The visualizer expects `prompt_tokens` and `completion_tokens` in `rlm_calls`.
These are set to 0 by default. To track real token usage, modify
`_sub_call_to_rlm_chat()` to extract from the OpenAI completion response:

```python
def _sub_call_to_rlm_chat(call: SubAgentCall, model_name: str = "") -> dict:
    return {
        "root_model": model_name,
        "prompt": call.prompt,
        "response": call.response,
        "prompt_tokens": call.prompt_tokens,      # add to SubAgentCall
        "completion_tokens": call.completion_tokens,  # add to SubAgentCall
        "execution_time": call.execution_time,
    }
```

### Disabling logging entirely

```python
config = ExperimentConfig(log_trajectory_sample_rate=0.0)
```

No files are created, no I/O overhead.
