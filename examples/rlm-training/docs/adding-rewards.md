# Adding a New Reward Function

This walkthrough shows how to add a custom reward function (e.g., process-based rewards) to the training pipeline.

## What you need to change

| File | Change | Required? |
|------|--------|-----------|
| `rewards.py` | Add your new reward function | Yes |
| `config.py` | No changes needed (just assign your function at runtime) | No |
| `train.py` / `debug_rollout.py` | Set `config.reward_fn = your_function` | Yes |

## Step 1: Write the reward function in `rewards.py`

Every reward function has the same signature:

```python
def my_reward(log_output: str, ds: dict[str, Any]) -> float:
```

**Arguments:**
- `log_output`: Raw stdout from running `/run_tests.sh` inside the Docker container. This typically contains pytest output with test results.
- `ds`: The full dataset entry dict from R2E-Gym-Lite. Available fields include:
  - `instance_id`: e.g., `"django__django-16229"`
  - `expected_output_json`: JSON string mapping test names to expected statuses
  - `problem_statement` / `issue_description` / `task`: the bug report text
  - `docker_image` / `image_name`: the Docker image for this task
  - `repo_name` / `repo`: repository name

**Return:** A float reward. Typically 0.0-1.0, but any float works (GRPO normalizes within a group).

### Example: Process-based reward

A process-based reward gives partial credit for intermediate progress, not just final test pass/fail:

```python
def process_based_reward(log_output: str, ds: dict[str, Any]) -> float:
    """Reward based on process signals, not just outcome.
    
    Scoring:
      - 0.1: Agent called finish() (showed intent to complete)
      - 0.2: Tests ran without crashing (valid patch applied)
      - 0.3: At least one test changed status from the baseline
      - 1.0: All tests match expected output (full solve)
    """
    score = 0.0
    
    # Check if tests ran at all (log has pytest output)
    if "short test summary info" in log_output:
        score = 0.2
    elif "error" in log_output.lower() or "Traceback" in log_output:
        return 0.1  # Tests crashed but at least something ran
    
    # Check test results
    parsed = _decolor_dict(_parse_pytest_log(log_output))
    expected_json = ds.get("expected_output_json")
    if not expected_json:
        return score
    
    expected = json.loads(expected_json)
    expected = _decolor_dict(expected)
    
    # Partial credit for matching tests
    parsed_norm = _normalize_test_keys(parsed)
    expected_norm = _normalize_test_keys(expected)
    
    if expected_norm:
        matching = sum(
            1 for k, v in expected_norm.items()
            if parsed_norm.get(k) == v
        )
        match_ratio = matching / len(expected_norm)
        
        if match_ratio == 1.0:
            score = 1.0
        elif match_ratio > 0:
            score = 0.3 + 0.5 * match_ratio  # 0.3 to 0.8 range
    
    return score
```

Note: You can use the existing helper functions (`_parse_pytest_log`, `_decolor_dict`, `_normalize_test_keys`) that are already defined in `rewards.py`.

## Step 2: Use it in your experiment

In `debug_rollout.py` or `train.py`:

```python
from config import ExperimentConfig
from rewards import process_based_reward

config = ExperimentConfig(
    experiment_name="process-reward-v1",
    reward_fn=process_based_reward,
)
```

Or override at runtime:

```python
config = ExperimentConfig()
config.reward_fn = process_based_reward
```

## Step 3: Test it standalone

You can test your reward function without running a full rollout:

```python
from rewards import process_based_reward
import json

# Simulate test output
fake_log = """
=========================== short test summary info ============================
PASSED tests/test_models.py::TestUser::test_create
FAILED tests/test_models.py::TestUser::test_delete - AssertionError
========= 1 passed, 1 failed =========
"""

fake_ds = {
    "expected_output_json": json.dumps({
        "TestUser.test_create": "PASSED",
        "TestUser.test_delete": "PASSED",
    })
}

reward = process_based_reward(fake_log, fake_ds)
print(f"Reward: {reward}")  # Should be partial credit
```

## Advanced: Reward that uses trajectory info

If you need access to the trajectory itself (e.g., penalize long rollouts), you can extend the reward function signature. The rollout code in `rollout.py` calls:

```python
reward = config.reward_fn(test_log_output, scenario.ds)
```

To add trajectory-aware rewards, you could wrap the function:

```python
def trajectory_aware_reward(log_output, ds, *, num_steps=0, finished=False):
    base = binary_test_reward(log_output, ds)
    # Bonus for finishing early
    if base > 0 and num_steps < 10:
        base += 0.1
    # Penalty for not calling finish()
    if not finished:
        base *= 0.5
    return min(base, 1.0)

# In rollout.py, you'd call it with extra kwargs
```

This would require a small change in `rollout.py` to pass extra arguments. See the rollout code for the exact call site.
