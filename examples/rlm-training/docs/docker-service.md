# Docker Test Service

This walkthrough explains how the Docker service and client work, and how to customize test execution.

## Architecture

```
Training Node                          Docker Node
┌────────────────────┐                ┌──────────────────────────┐
│  rollout.py        │                │  docker_service.py       │
│    ↓               │                │    (FastAPI on :8000)    │
│  docker_client.py  │── POST /test ─→│    ↓                    │
│    (httpx)         │                │  docker create + start   │
│                    │                │  docker cp patches.tar   │
│                    │                │  docker exec run_tests   │
│                    │←── response ───│  docker rm -f            │
└────────────────────┘                └──────────────────────────┘
```

## The /test endpoint flow

1. **Receive request**: `{docker_image, patch_tar_b64, test_cmd, timeout}`
2. **Create container**: `docker create {image}` + `docker start`
3. **Apply patches**: Decode the base64 tar, `docker cp` it in, `tar xf` inside `/testbed`
4. **Run tests**: `docker exec {container} bash -c "{test_cmd}"` with timeout
5. **Return**: test stdout+stderr, exit code, elapsed time
6. **Cleanup**: `docker rm -f` (always, even on error)

Containers are fully ephemeral -- created per request, destroyed after.

## Running the service

On the Docker Node:

```bash
# Install deps
pip install fastapi uvicorn

# Start the service
python docker_service.py --host 0.0.0.0 --port 8000

# Or with custom log level
python docker_service.py --log_level debug
```

Verify it's running:

```bash
curl http://docker-node:8000/health
# {"status": "ok"}
```

## How patches are sent

The rollout code on the Training Node:

1. Compares the modified work directory against the cached original (`get_modified_files()`)
2. Only sends files that changed (usually <100KB)
3. Packs them into a tar, base64-encodes it
4. Sends in the POST body

This means: no matter how large the repo is (100MB+), we only send the diff over the network.

## What you need to change

| What | File | Change |
|------|------|--------|
| Service URL | `config.py` | `docker_service_url = "http://your-docker-node:8000"` |
| Test command | `docker_client.py` | `test_cmd` parameter in `run_tests()` |
| Test timeout | `config.py` or call site | `timeout` parameter (default 300s) |
| Retry count | `docker_client.py` | `MAX_RETRIES` constant (default 2) |
| Network timeout | `docker_client.py` | `DEFAULT_TIMEOUT` constant (default 600s) |

## Customizing test execution

### Different test commands

The default test command is:

```bash
bash -lc 'cd /testbed && bash run_tests.sh 2>&1'
```

You can override this per-request:

```python
result = await docker_client.run_tests(
    docker_image=image,
    modified_files=files,
    test_cmd="cd /testbed && python -m pytest tests/ -x -v 2>&1",
    timeout=120,
)
```

### Adding setup steps before tests

If you need to install dependencies or run setup before tests, modify the test command:

```python
test_cmd = (
    "bash -lc '"
    "cd /testbed && "
    "pip install -e . 2>/dev/null && "
    "bash run_tests.sh 2>&1"
    "'"
)
```

### Changing how patches are applied

The current flow: tar is extracted into `/testbed`, overwriting files. If you need a different patching strategy (e.g., git apply), modify `docker_service.py`'s `/test` endpoint where it runs `tar xf`.

## Testing the service locally

You can test the full flow without the training pipeline:

```python
import asyncio
from docker_client import DockerClient

async def test():
    client = DockerClient("http://localhost:8000")

    # Health check
    status = await client.health_check()
    print(f"Service status: {status}")

    # Run a test with no patches (baseline)
    result = await client.run_tests(
        docker_image="namanjain12/aiohttp_final:f0d74880deec...",
        modified_files={},
    )
    print(f"Exit code: {result['exit_code']}")
    print(f"Output: {result['test_output'][:500]}")

asyncio.run(test())
```

## Monitoring

The service logs every request with container ID, image name, and elapsed time:

```
2026-02-19 15:30:00 docker_service INFO Created container a1b2c3d4e5f6 from namanjain12/django_final:...
2026-02-19 15:30:45 docker_service INFO Container a1b2c3d4e5f6: tests completed in 44.8s (exit=0)
2026-02-19 15:30:45 docker_service INFO Destroyed container a1b2c3d4e5f6
```

If the Docker Node is under heavy load, you can monitor container count:

```bash
watch 'docker ps | wc -l'
```
