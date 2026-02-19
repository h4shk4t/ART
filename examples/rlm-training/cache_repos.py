#!/usr/bin/env python3
"""One-time script to cache repo snapshots from R2E-Gym-Lite Docker images.

Run this on the Docker Node (the machine with Docker access) BEFORE training.
For each dataset entry, it:
  1. Starts the Docker container
  2. Copies /testbed out to the local cache directory
  3. Destroys the container
  4. Writes an index.json mapping instance_id -> cache path

Usage:
    python cache_repos.py --cache_dir /data/repo-cache
    python cache_repos.py --cache_dir /data/repo-cache --split dev_10pr_v1 --max_entries 10
    python cache_repos.py --cache_dir /data/repo-cache --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def make_instance_id(entry: dict) -> str:
    """Derive a unique instance_id from dataset fields."""
    repo = entry.get("repo_name", "unknown")
    commit = entry.get("commit_hash", "unknown")
    return f"{repo}__{commit[:12]}"


def pull_image(image: str) -> bool:
    """Pull a Docker image. Returns True on success."""
    result = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode == 0


def extract_testbed(image: str, dest: Path, timeout: int = 120) -> bool:
    """Start a container, copy /testbed out, destroy container. Returns True on success."""
    container_id = None
    try:
        result = subprocess.run(
            ["docker", "create", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  ERROR: docker create failed: {result.stderr.strip()}")
            return False
        container_id = result.stdout.strip()

        dest.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["docker", "cp", f"{container_id}:/testbed/.", str(dest)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  ERROR: docker cp failed: {result.stderr.strip()}")
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"  ERROR: timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=30,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache R2E-Gym-Lite repos locally")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/repo-cache",
        help="Directory to store cached repos",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to cache (train, dev_10pr_v1, etc.)",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Max entries to process (for testing)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip entries that already have a cached copy",
    )
    parser.add_argument(
        "--skip_pull",
        action="store_true",
        default=False,
        help="Skip docker pull (images already local)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print what would be done without doing it",
    )
    parser.add_argument(
        "--cp_timeout",
        type=int,
        default=120,
        help="Timeout in seconds for docker cp",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading R2E-Gym-Lite split={args.split}...")
    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split=args.split)
    entries = list(ds)
    if args.max_entries:
        entries = entries[: args.max_entries]
    print(f"Processing {len(entries)} entries")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    index_path = cache_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"Loaded existing index with {len(index)} entries")

    succeeded = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(entries):
        instance_id = make_instance_id(entry)
        image = entry.get("docker_image", "")
        dest = cache_dir / instance_id

        prefix = f"[{i + 1}/{len(entries)}] {instance_id}"

        if args.skip_existing and instance_id in index and dest.exists():
            skipped += 1
            continue

        if args.dry_run:
            print(f"{prefix}: would cache from {image}")
            continue

        print(f"{prefix}: caching from {image}...")
        t0 = time.time()

        if not args.skip_pull:
            if not pull_image(image):
                print(f"  WARNING: pull failed, trying create anyway...")

        if dest.exists():
            shutil.rmtree(dest)

        if extract_testbed(image, dest, timeout=args.cp_timeout):
            elapsed = time.time() - t0
            file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
            size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6

            index[instance_id] = {
                "instance_id": instance_id,
                "repo_name": entry.get("repo_name", ""),
                "docker_image": image,
                "commit_hash": entry.get("commit_hash", ""),
                "cache_path": str(dest),
                "file_count": file_count,
                "size_mb": round(size_mb, 1),
            }

            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

            succeeded += 1
            print(f"  OK: {file_count} files, {size_mb:.1f} MB, {elapsed:.1f}s")
        else:
            failed += 1
            if dest.exists():
                shutil.rmtree(dest)

    print(f"\nDone: {succeeded} cached, {skipped} skipped, {failed} failed")
    print(f"Index: {index_path} ({len(index)} entries)")

    if index:
        total_mb = sum(e.get("size_mb", 0) for e in index.values())
        print(f"Total cache size: {total_mb / 1000:.1f} GB")


if __name__ == "__main__":
    main()
