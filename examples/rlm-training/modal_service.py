#!/usr/bin/env python3
"""Modal-based Docker test service — cloud replacement for docker_service.py.

Deploys as a Modal ASGI app with the SAME URL paths (/test, /health, /ping)
so docker_client.py needs ZERO changes. Just swap the --docker-url.

Uses Modal Sandbox API to run tests in ephemeral containers pulled directly
from Docker Hub (e.g. namanjain12/aiohttp_final:commit_hash).

Setup (on your training server):
    1. cd /home/colligo/experiments/RLM/ART/examples/rlm-training
    2. uv run python -m modal token set --token-id <ID> --token-secret <SECRET>
    3. uv run python -m modal deploy modal_service.py
    4. Copy the URL from the output
    5. Update --docker-url in your launch scripts

Image caching:
    - Modal caches Docker image layers after first pull (~30-60s cold start)
    - Subsequent sandbox creates from same image are ~2-3s
    - Pre-warm images by running: uv run python modal_service.py --warm
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import tarfile
import time

import modal

logger = logging.getLogger("modal_service")

# --- Modal App ---
app = modal.App("rlm-docker-test-service")

# Image for the web endpoint container (needs fastapi, pydantic)
service_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "pydantic",
)

# Shared volume for passing large scripts (avoids ARG_MAX limit)
script_vol = modal.Volume.from_name("rlm-test-scripts-vol", create_if_missing=True)


# ---------------------------------------------------------------------------
# FastAPI app (served via modal.asgi_app for single-URL compatibility)
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI(title="RLM Docker Test Service (Modal)")


class TestRequest(BaseModel):
    docker_image: str
    patch_tar_b64: str = ""
    test_cmd: str = "bash -lc 'cd /testbed && bash run_tests.sh 2>&1'"
    timeout: int = 300


class TestResponse(BaseModel):
    test_output: str
    exit_code: int | None
    elapsed_seconds: float
    container_id: str


@web_app.get("/ping")
async def ping():
    return {"status": "ok"}


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@web_app.post("/test", response_model=TestResponse)
async def run_test(req: TestRequest):
    """Run tests in a Modal Sandbox container."""
    t0 = time.time()

    if not req.docker_image:
        raise HTTPException(status_code=400, detail="docker_image is required")

    try:
        # Build a Modal Image from the Docker Hub image.
        image = modal.Image.from_registry(
            req.docker_image,
            add_python="3.11",
            setup_dockerfile_commands=["RUN apt-get update -qq 2>/dev/null || true"],
        )

        # Decode the patch tar into individual files
        patch_files: dict[str, bytes] = {}
        if req.patch_tar_b64:
            raw = base64.b64decode(req.patch_tar_b64)
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            patch_files[member.name] = f.read()

        # Build a shell script that:
        #   1. Writes all patch files into /testbed
        #   2. Runs the test command
        script_lines = ["#!/bin/bash", "set -e", "cd /testbed"]
        for path, content in patch_files.items():
            b64_content = base64.b64encode(content).decode("ascii")
            script_lines.append(f"mkdir -p \"$(dirname '{path}')\"")
            script_lines.append(
                f"echo '{b64_content}' | base64 -d > '{path}'"
            )
        script_lines.append("set +e")
        script_lines.append(req.test_cmd)

        full_script = "\n".join(script_lines)
        script_bytes = full_script.encode("utf-8")

        # Determine if we need Volume-based transfer (ARG_MAX is 65536)
        use_volume = len(script_bytes) > 50000  # leave margin

        if use_volume:
            # Write script to shared Volume, sandbox reads it
            script_hash = hashlib.sha256(script_bytes).hexdigest()[:16]
            script_filename = f"run_{script_hash}_{int(time.time()*1000)}.sh"

            # Upload script to volume
            with script_vol.batch_upload() as batch:
                batch.put(io.BytesIO(script_bytes), script_filename)

            # Sandbox CMD: read script from volume and execute it
            sandbox_cmd = f"bash /scripts/{script_filename}"

            sandbox = modal.Sandbox.create(
                "bash", "-c", sandbox_cmd,
                image=image,
                timeout=req.timeout + 30,
                app=app,
                cpu=2.0,
                memory=4096,
                volumes={"/scripts": script_vol},
            )
        else:
            # Small script — pass directly as CMD argument
            sandbox = modal.Sandbox.create(
                "bash", "-c", full_script,
                image=image,
                timeout=req.timeout + 30,
                app=app,
                cpu=2.0,
                memory=4096,
            )

        sandbox_id = sandbox.object_id or "unknown"

        try:
            sandbox.wait()
        except modal.exception.SandboxTimeoutError:
            elapsed = time.time() - t0
            # Cleanup volume script if used
            if use_volume:
                try:
                    script_vol.remove_file(script_filename)
                except Exception:
                    pass
            return TestResponse(
                test_output=f"Test execution timed out after {req.timeout}s",
                exit_code=-1,
                elapsed_seconds=round(elapsed, 2),
                container_id=sandbox_id,
            )

        # Read output
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        test_output = stdout + stderr
        exit_code = sandbox.returncode

        elapsed = time.time() - t0

        # Cleanup volume script
        if use_volume:
            try:
                script_vol.remove_file(script_filename)
            except Exception:
                pass

        return TestResponse(
            test_output=test_output,
            exit_code=exit_code,
            elapsed_seconds=round(elapsed, 2),
            container_id=sandbox_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - t0
        logger.exception("Sandbox error for image %s", req.docker_image)
        raise HTTPException(
            status_code=500,
            detail=f"Modal sandbox error: {str(e)[:500]}",
        ) from e


# ---------------------------------------------------------------------------
# Deploy the FastAPI app as a single Modal function
# ---------------------------------------------------------------------------
@app.function(
    image=service_image,
    timeout=900,
    scaledown_window=300,  # keep warm for 5 min to avoid cold starts
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def fastapi_app():
    return web_app


# ---------------------------------------------------------------------------
# Image pre-warming utility
# ---------------------------------------------------------------------------
def warm_images():
    """Pre-pull common Docker images into Modal's cache.

    Run: uv run python modal_service.py --warm
    This triggers Modal to build/cache the images so first test runs are fast.
    """
    import json
    from pathlib import Path

    index_path = Path("repo-cache/index.json")
    if not index_path.exists():
        print("No repo-cache/index.json found. Nothing to warm.")
        return

    with open(index_path) as f:
        index = json.load(f)

    # Get unique Docker images
    images = list({entry["docker_image"] for entry in index.values() if entry.get("docker_image")})
    print(f"Found {len(images)} unique Docker images to pre-warm")

    for i, img in enumerate(images):
        print(f"[{i+1}/{len(images)}] Pre-building Modal image for: {img}")
        try:
            modal_img = modal.Image.from_registry(
                img,
                add_python="3.11",
                setup_dockerfile_commands=["RUN apt-get update -qq 2>/dev/null || true"],
            )
            # Force the build by attaching it to a function
            with app.run():
                modal_img.build()  # noqa: this triggers the cache
            print(f"  ✓ Cached successfully")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nDone! {len(images)} images pre-warmed in Modal's cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal Docker Test Service")
    parser.add_argument("--warm", action="store_true", help="Pre-warm Docker images in Modal's cache")
    args = parser.parse_args()

    if args.warm:
        warm_images()
    else:
        print("Usage:")
        print("  Deploy:  uv run python -m modal deploy modal_service.py")
        print("  Warm:    uv run python modal_service.py --warm")
        print("  Serve:   uv run python -m modal serve modal_service.py  (dev mode)")
