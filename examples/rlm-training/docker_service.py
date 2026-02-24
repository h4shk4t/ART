#!/usr/bin/env python3
"""FastAPI server for remote test execution. Runs on the Docker Node.

Provides a single /test endpoint that:
  1. Creates an ephemeral container from the task's Docker image
  2. Applies patched files (received as a base64-encoded tar)
  3. Runs the test command
  4. Returns test output
  5. Destroys the container

Usage:
    pip install fastapi uvicorn
    python docker_service.py
    python docker_service.py --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import base64
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("docker_service")
app = FastAPI(title="RLM Docker Test Service")


class TestRequest(BaseModel):
    docker_image: str
    patch_tar_b64: str
    test_cmd: str = "bash -lc 'cd /testbed && bash run_tests.sh 2>&1'"
    timeout: int = 300


class TestResponse(BaseModel):
    test_output: str
    exit_code: int | None
    elapsed_seconds: float
    container_id: str


@app.get("/ping")
async def ping():
    """Lightweight liveness check -- no subprocess, instant response."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        docker_ok = proc.returncode == 0
    except Exception:
        docker_ok = False

    return {
        "status": "ok" if docker_ok else "docker_unavailable",
    }


async def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a subprocess without blocking the event loop."""
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


@app.post("/test", response_model=TestResponse)
async def run_test(req: TestRequest):
    """Create container, apply patches, run tests, return output, destroy container."""
    container_id = None
    t0 = time.time()

    try:
        # 1. Create and start container with a long-running command to keep it alive.
        #    The image's default CMD may exit immediately, so we override with
        #    "sleep infinity" and run all operations via docker exec.
        result = await _run(
            ["docker", "run", "-d", "--entrypoint", "sleep", req.docker_image, "infinity"],
            timeout=60,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"docker run failed: {result.stderr.strip()}",
            )
        container_id = result.stdout.strip()
        logger.info("Created container %s from %s", container_id[:12], req.docker_image)

        # 3. Apply patches (decode tar and copy into container)
        if req.patch_tar_b64:
            patch_bytes = base64.b64decode(req.patch_tar_b64)
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=True) as tmp:
                tmp.write(patch_bytes)
                tmp.flush()
                result = await _run(
                    ["docker", "cp", tmp.name, f"{container_id}:/tmp/_patches.tar"],
                    timeout=30,
                )
                if result.returncode != 0:
                    raise HTTPException(
                        status_code=500,
                        detail=f"docker cp (tar) failed: {result.stderr.strip()}",
                    )

            # Extract patches into /testbed
            result = await _run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    "cd /testbed && tar xf /tmp/_patches.tar && rm /tmp/_patches.tar",
                ],
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "tar extract warning (may be ok): %s", result.stderr.strip()
                )

        # 4. Run tests
        try:
            result = await _run(
                ["docker", "exec", container_id, "bash", "-c", req.test_cmd],
                timeout=req.timeout,
            )
            test_output = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            test_output = f"Test execution timed out after {req.timeout}s"
            exit_code = -1

        elapsed = time.time() - t0
        logger.info(
            "Container %s: tests completed in %.1fs (exit=%s)",
            container_id[:12],
            elapsed,
            exit_code,
        )

        return TestResponse(
            test_output=test_output,
            exit_code=exit_code,
            elapsed_seconds=round(elapsed, 2),
            container_id=container_id[:12],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /test")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if container_id:
            try:
                await _run(["docker", "rm", "-f", container_id], timeout=30)
            except Exception:
                pass
            logger.info("Destroyed container %s", container_id[:12])


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="RLM Docker Test Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--log_level", default="info", help="Log level (debug, info, warning)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
