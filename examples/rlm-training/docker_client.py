"""Async HTTP client for the remote Docker test service.

Runs on the Training Node. Wraps the single /test endpoint with retry logic
and tar creation from modified files.
"""

from __future__ import annotations

import base64
import io
import logging
import tarfile
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600.0
MAX_RETRIES = 2


class DockerClient:
    """Thin async client for the Docker test service."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health_check(self) -> dict[str, Any]:
        """Check if the Docker service is reachable and Docker is available."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def run_tests(
        self,
        docker_image: str,
        modified_files: dict[str, bytes],
        test_cmd: str = "bash -lc 'cd /testbed && bash run_tests.sh 2>&1'",
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Send modified files to Docker service and run tests.

        Args:
            docker_image: The Docker image to create the container from.
            modified_files: Dict mapping relative paths (from /testbed) to
                file contents as bytes. These are packed into a tar and
                applied on top of the container's /testbed.
            test_cmd: The test command to run inside the container.
            timeout: Max seconds for test execution.

        Returns:
            Dict with keys: test_output, exit_code, elapsed_seconds, container_id
        """
        patch_tar_b64 = _create_patch_tar_b64(modified_files)

        payload = {
            "docker_image": docker_image,
            "patch_tar_b64": patch_tar_b64,
            "test_cmd": test_cmd,
            "timeout": timeout,
        }

        last_error: Exception | None = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/test", json=payload
                    )
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Docker service request failed (attempt %d/%d): %s",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        e,
                    )
                    continue
            except httpx.HTTPStatusError as e:
                logger.error("Docker service error: %s %s", e.response.status_code, e.response.text)
                raise

        raise RuntimeError(
            f"Docker service unreachable after {MAX_RETRIES + 1} attempts: {last_error}"
        )


def _create_patch_tar_b64(modified_files: dict[str, bytes]) -> str:
    """Create a base64-encoded tar from modified file contents.

    The tar is structured so that extracting it in /testbed applies the patches:
        tar xf patches.tar  (from /testbed)
    So file paths in the tar are relative to /testbed.
    """
    if not modified_files:
        return ""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel_path, content in modified_files.items():
            info = tarfile.TarInfo(name=rel_path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return base64.b64encode(buf.getvalue()).decode("ascii")


_SKIP_PATTERNS = {
    "__pycache__",
    ".pyc",
    ".pyo",
    ".so",
    ".o",
    ".a",
    ".egg-info",
    ".egg",
    ".whl",
    ".git",
}


def _should_skip(rel_path: str) -> bool:
    """Return True for binary/build artifacts that shouldn't be sent to Docker."""
    parts = rel_path.split("/")
    for part in parts:
        for pattern in _SKIP_PATTERNS:
            if part == pattern or part.endswith(pattern):
                return True
    return False


def get_modified_files(
    work_dir: str | Path, cache_dir: str | Path
) -> dict[str, bytes]:
    """Diff a work directory against the cached original to find modified files.

    Compares file contents between work_dir and cache_dir. Returns a dict
    mapping relative paths to the new file contents for any files that were
    added or modified. Skips binary/build artifacts (.pyc, .so, __pycache__).
    """
    work_path = Path(work_dir)
    cache_path = Path(cache_dir)
    modified: dict[str, bytes] = {}

    for file in work_path.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(work_path)
        rel_str = str(rel)

        if _should_skip(rel_str):
            continue

        cache_file = cache_path / rel

        try:
            new_content = file.read_bytes()
        except (OSError, PermissionError):
            continue

        if not cache_file.exists():
            modified[rel_str] = new_content
        else:
            try:
                old_content = cache_file.read_bytes()
            except (OSError, PermissionError):
                continue
            if new_content != old_content:
                modified[rel_str] = new_content

    return modified
