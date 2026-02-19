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


@app.get("/health")
async def health():
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=10
    )
    return {
        "status": "ok" if result.returncode == 0 else "docker_unavailable",
    }


@app.post("/test", response_model=TestResponse)
async def run_test(req: TestRequest):
    """Create container, apply patches, run tests, return output, destroy container."""
    container_id = None
    t0 = time.time()

    try:
        # 1. Create container
        result = subprocess.run(
            ["docker", "create", req.docker_image],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"docker create failed: {result.stderr.strip()}",
            )
        container_id = result.stdout.strip()
        logger.info("Created container %s from %s", container_id[:12], req.docker_image)

        # 2. Start container
        result = subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"docker start failed: {result.stderr.strip()}",
            )

        # 3. Apply patches (decode tar and copy into container)
        if req.patch_tar_b64:
            patch_bytes = base64.b64decode(req.patch_tar_b64)
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=True) as tmp:
                tmp.write(patch_bytes)
                tmp.flush()
                result = subprocess.run(
                    ["docker", "cp", tmp.name, f"{container_id}:/tmp/_patches.tar"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    raise HTTPException(
                        status_code=500,
                        detail=f"docker cp (tar) failed: {result.stderr.strip()}",
                    )

            # Extract patches into /testbed
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    "cd /testbed && tar xf /tmp/_patches.tar && rm /tmp/_patches.tar",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "tar extract warning (may be ok): %s", result.stderr.strip()
                )

        # 4. Run tests
        try:
            result = subprocess.run(
                ["docker", "exec", container_id, "bash", "-c", req.test_cmd],
                capture_output=True,
                text=True,
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
        # 5. Always destroy container
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=30,
            )
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
