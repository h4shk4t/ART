#!/usr/bin/env python3
"""Cache repo snapshots from Docker images via the remote Docker service.

Run this on the TRAINING NODE. Instead of requiring local Docker, it
calls the remote Docker service's /extract endpoint to download repos.

Usage:
    uv run python cache_repos_remote.py --docker-url https://4072-130-248-127-34.ngrok-free.app
    uv run python cache_repos_remote.py --docker-url https://... --max-entries 30
    uv run python cache_repos_remote.py --docker-url https://... --max-entries 30 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache repos via remote Docker service")
    parser.add_argument(
        "--docker-url",
        type=str,
        required=True,
        help="URL of the Docker service (e.g. https://...ngrok-free.app)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./repo-cache",
        help="Local directory to store cached repos",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to cache",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=30,
        help="Max entries to process",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip entries already in the cache index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be done without doing it",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout per extraction request (seconds)",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading R2E-Gym-Lite split={args.split}...")
    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split=args.split)
    entries = list(ds)
    if args.max_entries:
        entries = entries[: args.max_entries]
    print(f"Processing up to {len(entries)} entries")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index
    index: dict[str, dict] = {}
    index_path = cache_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"Loaded existing index with {len(index)} entries")

    # Check Docker service health
    base_url = args.docker_url.rstrip("/")
    headers = {}
    if "ngrok" in base_url:
        headers["ngrok-skip-browser-warning"] = "true"

    try:
        resp = httpx.get(f"{base_url}/health", headers=headers, timeout=10)
        print(f"Docker service: {resp.json()}")
    except Exception as e:
        print(f"ERROR: Docker service unreachable: {e}")
        return

    succeeded = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(entries):
        repo = entry.get("repo_name", "unknown")
        commit = entry.get("commit_hash", "unknown")
        instance_id = f"{repo}__{commit[:12]}"
        image = entry.get("docker_image", "")
        dest = cache_dir / instance_id

        prefix = f"[{i + 1}/{len(entries)}] {instance_id}"

        if args.skip_existing and instance_id in index and dest.exists():
            skipped += 1
            continue

        if args.dry_run:
            print(f"{prefix}: would cache from {image}")
            continue

        print(f"{prefix}: extracting from {image}...")
        t0 = time.time()

        try:
            resp = httpx.post(
                f"{base_url}/extract",
                json={"docker_image": image, "timeout": 120},
                headers=headers,
                timeout=args.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Decode and extract the tar
            tar_bytes = base64.b64decode(data["testbed_tar_b64"])
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(str(dest))

            elapsed = time.time() - t0
            file_count = data.get("file_count", sum(1 for _ in dest.rglob("*") if _.is_file()))
            size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6

            index[instance_id] = {
                "instance_id": instance_id,
                "repo_name": repo,
                "docker_image": image,
                "commit_hash": commit,
                "cache_path": str(dest),
                "file_count": file_count,
                "size_mb": round(size_mb, 1),
            }

            # Save index after each success (crash-safe)
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

            succeeded += 1
            print(f"  OK: {file_count} files, {size_mb:.1f} MB, {elapsed:.1f}s")

        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            # Clean up partial directory
            if dest.exists():
                import shutil
                shutil.rmtree(dest, ignore_errors=True)

    print(f"\nDone: {succeeded} cached, {skipped} skipped, {failed} failed")
    print(f"Index: {index_path} ({len(index)} entries)")

    if index:
        total_mb = sum(e.get("size_mb", 0) for e in index.values())
        print(f"Total cache size: {total_mb / 1000:.1f} GB")


if __name__ == "__main__":
    main()
