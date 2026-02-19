# Health Monitoring

This walkthrough explains the system health monitoring and how to tune it for your node.

## Why health monitoring?

When running 32+ concurrent REPL rollouts on a training node, each of which can:
- Spawn `bash()` subprocesses (find, grep, wc, etc.)
- Fork recursive sub-agents (each with their own REPL loop)
- Create temp files in `/tmp`

Things that can go wrong without monitoring:
- A `bash("find / -name '*.py'")` hangs forever, leaking a process
- A crashed rollout leaves its `/tmp/rlm_rollout_*` directory behind, filling disk
- Too many concurrent subprocesses overwhelm the kernel
- `/tmp` fills up, crashing everything including the training loop

## Components

### `safe_subprocess(cmd, cwd, timeout)`

Every `bash()` call in the REPL goes through this instead of raw `subprocess.run`. It:

1. Creates a **new process group** (`start_new_session=True`) so the entire process tree can be killed
2. Enforces a **hard timeout** (default 90s from config)
3. **Truncates output** to prevent memory blowup from commands that dump huge output
4. Returns `(output, exit_code)` -- never raises on timeout (returns exit code -1)

### `RolloutGuard` (context manager)

Wraps each rollout's lifecycle. Guarantees cleanup of the work directory even if the rollout crashes:

```python
with health_monitor.register_rollout(work_dir) as guard:
    # do rollout work...
    # if anything raises, work_dir is still cleaned up
```

### `HealthMonitor`

Periodic health checker with these metrics:

| Metric | What it checks | Warning threshold | Critical threshold |
|--------|---------------|-------------------|-------------------|
| Child processes | `pgrep -c -P {pid}` | 80% of limit | 100% of limit |
| /tmp free space | `shutil.disk_usage` | < 20 GB free | < 5 GB free |
| Active rollouts | Internal tracking | (informational) | (informational) |
| Stale rollout dirs | `/tmp/rlm_rollout_*` not in active set | > 10 dirs | (triggers reap) |

## What you need to change

| What | File | Change |
|------|------|--------|
| Subprocess timeout | `config.py` | `bash_timeout = 90` (seconds) |
| Max child processes | `config.py` | `max_child_processes = 256` |
| /tmp usage limit | `config.py` | `max_tmp_usage_gb = 50.0` |
| Health check frequency | `config.py` | `health_check_interval = 60` (seconds) |
| Stale dir max age | `health.py` | `reap_stale_rollouts(max_age_seconds=3600)` |
| Output truncation | `health.py` | `max_output` param in `safe_subprocess()` |

## Tuning for your node

### Process limits

With 32 concurrent rollouts, each potentially running 1-2 `bash()` calls plus sub-agent LLM requests:

```
Estimated max processes = 32 rollouts * (1 bash + 1 sub-agent) = ~96
```

The default limit of 256 has plenty of headroom. If you increase `max_concurrent` to 64+, bump this accordingly.

### Disk usage

Each repo copy is ~2-10 MB. With 32 concurrent rollouts:

```
Estimated /tmp usage = 32 * 10 MB = ~320 MB
```

Very modest. The main risk is stale directories from crashed rollouts accumulating over hours. The reaper handles this.

### Timeout tuning

Most `bash()` calls (find, grep, cat, wc) complete in <5 seconds. The 90-second default is generous. If models start running expensive commands (e.g., `bash("python -m pytest tests/ -v")`), you might want to increase it for specific use cases, or just let `run_tests()` handle test execution through the Docker service.

## Integration points

- **`repl.py`**: `bash()` calls `health_monitor.safe_subprocess()` instead of raw `subprocess.run`
- **`rollout.py`**: Each rollout wrapped in `health_monitor.register_rollout(work_dir)`
- **`train.py`**: Background task runs `health_monitor.check()` every `health_check_interval` seconds, logs the summary, and calls `reap_stale_rollouts()` on critical status

## Inspecting health at runtime

The health monitor logs a one-line summary every check interval:

```
INFO health: procs=45 | tmp=2.3GB | rollouts=28 | stale=0
```

If warnings or critical issues are detected:

```
WARNING health: procs=210 | tmp=45.2GB | rollouts=30 | stale=12 | WARN=1
ERROR health: CRITICAL: Child process count (260) exceeds limit (256)
```

## Testing standalone

```python
from health import HealthMonitor

monitor = HealthMonitor(max_child_processes=256, max_tmp_usage_gb=50.0)
status = monitor.check()
print(f"Processes: {status.child_process_count}")
print(f"/tmp used: {status.tmp_usage_gb} GB")
print(f"Warnings: {status.warnings}")
print(f"Critical: {status.critical}")
print(f"Summary: {monitor.summary()}")
```
