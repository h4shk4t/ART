# Training Loop Walkthrough

## Overview

`train.py` implements the GRPO training loop using ART's `LocalBackend`.
It orchestrates the full pipeline: loading data, running parallel rollouts,
computing rewards via the Docker service, and training the model.

## Quick Start

```bash
cd ART/examples/rlm-training

# Basic training run
uv run python train.py

# Custom experiment
uv run python train.py \
    --experiment-name "longer_context" \
    --max-steps 25 \
    --max-context-tokens 32000 \
    --groups-per-step 4 \
    --log-sample-rate 0.1
```

## Architecture

```
train.py
├── Parse args → ExperimentConfig
├── Start HealthMonitor background thread
├── Initialize LocalBackend + TrainableModel
├── Load R2E-Gym-Lite dataset → Scenario list
├── Docker service health check
├── Resume from checkpoint (if exists)
└── Training loop:
    ├── gather_trajectory_groups()
    │   └── N groups × M rollouts (parallel)
    │       └── rollout() → art.Trajectory
    ├── backend.train(model, groups)
    ├── Aggregate metrics (reward, steps, finished rate)
    ├── model.log(groups, metrics)
    └── model.merge_state() for resume
```

## Training Step Breakdown

Each training step:

1. **Gather**: `groups_per_step` scenarios × `rollouts_per_group` rollouts run
   concurrently (bounded by `max_concurrent` semaphore). Each rollout is a full
   REPL loop producing an `art.Trajectory`.

2. **Train**: `backend.train()` computes GRPO advantages within each group
   (same scenario, different rollouts) and updates the model.

3. **Log**: Metrics are logged to W&B and ART's trajectory store.

4. **Checkpoint**: `model.merge_state()` saves the current step for resume.

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment-name` | `default` | Experiment identifier |
| `--model-name` | `r2e-rlm-qwen3-14b` | ART model name |
| `--project` | `rlm-training` | ART project name |
| `--base-model` | `Qwen/Qwen3-14B` | HuggingFace base model |
| `--vllm-url` | `http://localhost:8000/v1` | Local vLLM server |
| `--docker-url` | `http://docker-node:8000` | Remote Docker service |
| `--cache-dir` | `/data/repo-cache` | Pre-cached repo directory |
| `--num-epochs` | `3` | Dataset epochs |
| `--groups-per-step` | `8` | Scenarios per training step |
| `--rollouts-per-group` | `4` | Rollouts per scenario |
| `--max-concurrent` | `32` | Max parallel rollouts |
| `--max-steps` | `15` | Max REPL steps per rollout |
| `--max-context-tokens` | `28000` | Context window budget |
| `--learning-rate` | `5e-6` | GRPO learning rate |
| `--dataset-size` | `None` | Limit dataset (for testing) |
| `--log-dir` | `logs` | Trajectory log directory |
| `--log-sample-rate` | `0.0` | Fraction of rollouts to log |

## Resume Support

Training automatically resumes from the last completed step:

```python
state = model.read_state()
initial_step = state["step"] + 1 if state and "step" in state else 0
```

The state is stored in `.art/{model_name}/state.json`. If you need to restart
from scratch, delete this file.

## Health Monitoring

A background thread runs `HealthMonitor.check()` every `health_check_interval`
seconds (default 60). It logs warnings for:

- High child process count (approaching `max_child_processes`)
- High `/tmp` disk usage (approaching `max_tmp_usage_gb`)
- Stale rollout directories (older than 1 hour)

Critical issues are logged as errors but don't stop training.

## Experiment Workflow

### 1. Define hypothesis

"Using a more detailed system prompt with explicit chain-of-thought instructions
will increase the solve rate."

### 2. Create experiment config

```bash
uv run python train.py \
    --experiment-name "cot_prompt_v1" \
    --log-sample-rate 0.1 \
    --dataset-size 500
```

In code, swap the prompt function:

```python
# In train.py or a wrapper script
from prompts import r2e_cot_system_prompt
config.system_prompt_fn = r2e_cot_system_prompt
```

### 3. Monitor training

- **W&B**: Real-time reward curves, loss, step time
- **Trajectories**: `logs/cot_prompt_v1/step_*/` — upload to RLM visualizer

### 4. Compare experiments

Each experiment gets its own `experiment_name` and `config_hash`. The logged
metrics include all config values, making it easy to compare runs in W&B.

## Customization

### Using a different reward function

```python
from rewards import partial_test_reward
config.reward_fn = partial_test_reward
```

### Adjusting GRPO parameters

The `backend.train()` call accepts additional GRPO parameters:

```python
result = await backend.train(
    model, groups,
    learning_rate=config.learning_rate,
    advantage_balance=0.1,    # bias towards positive advantages
    scale_rewards=True,        # normalize by reward std dev
)
```

See `LocalBackend.train()` for the full parameter list.

### Adding evaluation

Add periodic evaluation by checking `batch.step % eval_steps == 0` and
running `art.gather_trajectories()` on a held-out set.
