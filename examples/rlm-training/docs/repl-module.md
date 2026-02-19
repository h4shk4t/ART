# REPL Module Walkthrough

## Overview

`repl.py` implements `LocalREPL` — a persistent Python REPL that exposes file
operation helpers as globals to model-generated code. It mirrors the interface
of the Docker-based `DockerREPL` (in `ART/examples/r2e-gym-rlm/repl.py`) but
runs all file operations locally on the training node.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  LocalREPL (persistent globals dict)            │
│                                                 │
│  File ops:  ls, read, grep, apply_patch, bash   │
│     └─> subprocess on /tmp/rollout_{uuid}/      │
│                                                 │
│  Sub-agents:  llm_query, llm_query_batched      │
│     └─> child LocalREPL + model completion loop │
│     └─> histories stored for joint GRPO         │
│                                                 │
│  Control:  finish, run_tests                    │
│  Execution: exec(code, self._globals)           │
└─────────────────────────────────────────────────┘
```

## Key Components

### `LocalREPL`

The main class. Created once per rollout with:
- `work_dir` — path to the local repo copy (`/tmp/rlm_rollout_{uuid}/`)
- `model` — `art.Model` for sub-agent completions
- `trajectory` — `art.Trajectory` where sub-agent histories are stored
- `health` — `HealthMonitor` for subprocess safety

### `ExecResult`

Structured return from `execute()`:
- `stdout` — captured print output (truncated)
- `stderr` — traceback text if an exception occurred
- `execution_time` — wall-clock seconds
- `sub_agent_calls` — list of `SubAgentCall` records from that step

### `SubAgentCall`

Record of a single `llm_query` invocation:
- `prompt` — what was asked
- `response` — the sub-agent's final answer
- `execution_time` — how long the sub-agent loop took

## How It Works

### Code Execution

The model writes Python code. `execute()` runs it via `exec(code, self._globals)`.
The globals dict persists across calls, so variables defined in step 1 are
available in step 5. This is the same mechanism as `DockerREPL`.

### File Operations

All file ops delegate to `safe_subprocess()` from `health.py`, which provides:
- Process group isolation (no orphan processes)
- Hard timeouts (default 30s for file ops, 90s for bash)
- Output truncation

The `_resolve_path()` method prevents directory traversal — all paths are
resolved relative to `work_dir` and checked to stay within it.

### Sub-Agents (llm_query)

When the model calls `llm_query(prompt)`:
1. A child `LocalREPL` is created with its own globals but the same `work_dir`
2. Dangerous operations (`finish`, `apply_patch`, `run_tests`) are removed
3. The child runs a multi-step loop: model writes code → exec → observe → repeat
4. When the model responds without a code block, that's the final answer
5. The full sub-agent conversation is stored in `trajectory.additional_histories`
   for joint GRPO training

Recursion depth is bounded by `max_depth` (default 2).

### run_tests()

During the REPL loop, `run_tests()` runs whatever local test script exists.
The real reward-computing test execution happens after the loop, via the Docker
service. This is a key architectural difference from the Docker-based approach.

## Customization Points

### Changing file operation behavior

To add a new REPL tool (e.g., `search_symbols`):

1. Add a method to `LocalREPL`:

```python
def search_symbols(self, name: str) -> str:
    """Find symbol definitions using ctags or similar."""
    output, _ = safe_subprocess(
        f"grep -rn 'def {name}\\|class {name}' .",
        cwd=self.work_dir,
        timeout=30,
    )
    return self._truncate(output)
```

2. Register it in `__init__`:

```python
self._globals["search_symbols"] = self.search_symbols
```

3. Document it in the system prompt (see `docs/adding-prompts.md`).

### Changing truncation limits

`max_output_chars` controls how much output is shown to the model. Default is
3000 characters. You can set this per-experiment in `ExperimentConfig`:

```python
config = ExperimentConfig(
    max_output_chars=5000,  # more context for the model
)
```

### Changing sub-agent behavior

- `max_depth` — how deep sub-agents can recurse (default 2)
- `max_sub_agent_steps` — max REPL steps per sub-agent (default 8)
- `max_completion_tokens` — per-completion token limit (default 4000)
- `sub_agent_system_prompt` — the system prompt for sub-agents

All of these are wired through `ExperimentConfig`.

## Differences from DockerREPL

| Aspect | DockerREPL | LocalREPL |
|--------|-----------|-----------|
| File ops | `runtime.run()` (Docker exec) | `subprocess.run(cwd=work_dir)` |
| Test execution | In-container | Placeholder (real tests via Docker service) |
| Process safety | Docker isolation | `safe_subprocess()` with process groups |
| Return type | `str` | `ExecResult` (structured) |
| Path root | `/testbed` | `.` (relative to work_dir) |
