#!/usr/bin/env python3
"""Cache repo snapshots using Modal — no local Docker required.

Uses Modal Volume as intermediary storage to avoid stdout buffer limits.
Flow: Sandbox tars /testbed → writes to Volume → client downloads from Volume.

Usage:
    uv run python cache_repos_modal.py --cache_dir repo-cache
    uv run python cache_repos_modal.py --cache_dir repo-cache --max_entries 50
    uv run python cache_repos_modal.py --cache_dir repo-cache --workers 4
    uv run python cache_repos_modal.py --dry_run
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

import modal

app = modal.App("rlm-cache-repos")

# Persistent volume for transferring tars between sandbox and client
cache_vol = modal.Volume.from_name("rlm-repo-cache-vol", create_if_missing=True)


def make_instance_id(entry: dict) -> str:
    repo = entry.get("repo_name", "unknown")
    commit = entry.get("commit_hash", "unknown")
    return f"{repo}__{commit[:12]}"


def extract_testbed_modal(image_name: str, instance_id: str, dest: Path, timeout: int = 300) -> bool:
    """Use Modal Sandbox + Volume to extract /testbed from a Docker image."""
    tar_path_in_vol = f"/vol/{instance_id}.tar.gz"
    try:
        image = modal.Image.from_registry(image_name)

        # Tar /testbed into the mounted volume (gzip for smaller transfer)
        sandbox = modal.Sandbox.create(
            "bash", "-c",
            f"tar czf {tar_path_in_vol} -C /testbed . 2>/dev/null",
            image=image,
            timeout=timeout,
            app=app,
            cpu=1.0,
            memory=2048,
            volumes={"/vol": cache_vol},
        )

        sandbox.wait()
        cache_vol.reload()

        if sandbox.returncode != 0:
            stderr = sandbox.stderr.read()
            print(f"  ERROR: sandbox failed (exit={sandbox.returncode}): {stderr[:200]}")
            return False

        # Download the tar from the volume
        try:
            tar_bytes = b""
            for chunk in cache_vol.read_file(f"{instance_id}.tar.gz"):
                tar_bytes += chunk
        except Exception as e:
            print(f"  ERROR: failed to read from volume: {e}")
            return False

        if not tar_bytes:
            print("  ERROR: empty tar from volume")
            return False

        # Extract locally
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(path=str(dest))

        # Clean up the volume file to save space
        try:
            cache_vol.remove_file(f"{instance_id}.tar.gz")
        except Exception:
            pass  # non-critical

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache R2E repos via Modal")
    parser.add_argument("--cache_dir", type=str, default="repo-cache")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--cp_timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for caching (note: sequential for now due to volume sync)")
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

    # Filter entries to process
    to_process = []
    skipped = 0
    for i, entry in enumerate(entries):
        instance_id = make_instance_id(entry)
        dest = cache_dir / instance_id
        if args.skip_existing and instance_id in index and dest.exists():
            skipped += 1
            continue
        if args.dry_run:
            image = entry.get("docker_image", "")
            print(f"[{i+1}/{len(entries)}] {instance_id}: would cache from {image}")
            continue
        to_process.append((i, entry))

    if args.dry_run:
        print(f"\nDry run: {len(to_process)} would be cached, {skipped} skipped")
        return

    print(f"\n{len(to_process)} to cache, {skipped} already cached\n")

    succeeded = 0
    failed = 0

    with app.run():
        for idx, (i, entry) in enumerate(to_process):
            instance_id = make_instance_id(entry)
            image = entry.get("docker_image", "")
            dest = cache_dir / instance_id
            prefix = f"[{i + 1}/{len(entries)}] {instance_id}"

            print(f"{prefix}: caching from {image}...")
            t0 = time.time()

            if dest.exists():
                shutil.rmtree(dest)

            if extract_testbed_modal(image, instance_id, dest, timeout=args.cp_timeout):
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

            # Progress update every 10 entries
            if (idx + 1) % 10 == 0:
                print(f"\n--- Progress: {succeeded} cached, {failed} failed, "
                      f"{len(to_process) - idx - 1} remaining ---\n")

    print(f"\nDone: {succeeded} cached, {skipped} skipped, {failed} failed")
    print(f"Index: {index_path} ({len(index)} entries)")

    if index:
        total_mb = sum(e.get("size_mb", 0) for e in index.values())
        print(f"Total cache size: {total_mb / 1000:.1f} GB")


if __name__ == "__main__":
    main()
