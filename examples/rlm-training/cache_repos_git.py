#!/usr/bin/env python3
"""Cache repo snapshots using git clone — FREE, no Docker or Modal needed.

For each dataset entry:
  1. git clone the repo (shallow, fast)
  2. git checkout the specific commit
  3. Save to repo-cache/

This is the fastest and cheapest way to cache repos on the training node.

Usage:
    uv run python cache_repos_git.py --cache_dir repo-cache
    uv run python cache_repos_git.py --cache_dir repo-cache --max_entries 100 --workers 8
    uv run python cache_repos_git.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


# Mapping from R2E-Gym repo_name (as it appears in the dataset) to GitHub org/repo.
# The fallback auto-guess tries {repo_name}/{repo_name}.git which works for many
# repos (e.g., numpy→numpy/numpy, datalad→datalad/datalad). Only repos where
# the GitHub org differs from the repo name need explicit entries here.
REPO_MAP: dict[str, str] = {
    # --- Repos where org != repo_name (MUST be listed) ---
    "aiohttp": "aio-libs/aiohttp",
    "coveragepy": "nedbat/coveragepy",
    "flask": "pallets/flask",
    "requests": "psf/requests",
    "scikit-learn": "scikit-learn/scikit-learn",
    "pytest": "pytest-dev/pytest",
    "matplotlib": "matplotlib/matplotlib",
    "pandas": "pandas-dev/pandas",
    "sphinx": "sphinx-doc/sphinx",
    "pylint": "pylint-dev/pylint",
    "xarray": "pydata/xarray",
    "tornado": "tornadoweb/tornado",
    "pydantic": "pydantic/pydantic",
    "fastapi": "fastapi/fastapi",
    "httpx": "encode/httpx",
    "starlette": "encode/starlette",
    "uvicorn": "encode/uvicorn",
    "boto3": "boto/boto3",
    "botocore": "boto/botocore",
    "jinja2": "pallets/jinja",
    "click": "pallets/click",
    "werkzeug": "pallets/werkzeug",
    "marshmallow": "marshmallow-code/marshmallow",
    "beautifulsoup4": "getanewsletter/BeautifulSoup4",
    "rich": "Textualize/rich",
    "black": "psf/black",
    "mypy": "python/mypy",
    "poetry": "python-poetry/poetry",
    "pip": "pypa/pip",
    "setuptools": "pypa/setuptools",
    "tox": "tox-dev/tox",
    "hypothesis": "HypothesisWorks/hypothesis",
    "attrs": "python-attrs/attrs",
    "cryptography": "pyca/cryptography",
    "pillow": "python-pillow/Pillow",
    "opencv": "opencv/opencv-python",
    "spacy": "explosion/spaCy",
    "transformers": "huggingface/transformers",
    "datasets": "huggingface/datasets",
    "accelerate": "huggingface/accelerate",
    "arrow": "apache/arrow",
    "pyarrow": "apache/arrow",
    "salt": "saltstack/salt",
    # --- Repos where org == repo_name (covered by fallback, but listed for clarity) ---
    "django": "django/django",
    "sympy": "sympy/sympy",
    "numpy": "numpy/numpy",
    "astropy": "astropy/astropy",
    "sqlalchemy": "sqlalchemy/sqlalchemy",
    "celery": "celery/celery",
    "scrapy": "scrapy/scrapy",
    "paramiko": "paramiko/paramiko",
    "fabric": "fabric/fabric",
    "ansible": "ansible/ansible",
    "dask": "dask/dask",
    "networkx": "networkx/networkx",
    "nltk": "nltk/nltk",
    "tqdm": "tqdm/tqdm",
    "datalad": "datalad/datalad",
}


def make_instance_id(entry: dict) -> str:
    repo = entry.get("repo_name", "unknown")
    commit = entry.get("commit_hash", "unknown")
    return f"{repo}__{commit[:12]}"


def get_github_url(repo_name: str) -> str | None:
    """Map R2E-Gym repo_name to a GitHub clone URL."""
    if repo_name in REPO_MAP:
        return f"https://github.com/{REPO_MAP[repo_name]}.git"
    # Try common patterns: org_name == repo_name
    return f"https://github.com/{repo_name}/{repo_name}.git"


def clone_and_checkout(repo_name: str, commit_hash: str, dest: Path, timeout: int = 120) -> bool:
    """Clone a repo and checkout the PARENT of the fix commit (= buggy version).

    R2E-Gym's commit_hash is the fix commit. We need the state just BEFORE
    the fix (commit_hash~1) so the model sees the buggy code.
    """
    github_url = get_github_url(repo_name)
    if not github_url:
        print(f"  ERROR: no GitHub URL mapping for {repo_name}")
        return False

    try:
        # Full clone (need full history for arbitrary commit checkout)
        result = subprocess.run(
            ["git", "clone", "--quiet", github_url, str(dest)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            # Try without .git suffix
            alt_url = github_url.replace(".git", "")
            result = subprocess.run(
                ["git", "clone", "--quiet", alt_url, str(dest)],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                print(f"  ERROR: git clone failed: {result.stderr.strip()[:200]}")
                return False

        # Checkout the PARENT of the fix commit (= the buggy version)
        # commit_hash~1 is the commit just before the fix was applied
        result = subprocess.run(
            ["git", "checkout", "--quiet", f"{commit_hash}~1"],
            capture_output=True, text=True, timeout=30,
            cwd=str(dest),
        )
        if result.returncode != 0:
            # Fallback: try checkout the fix commit itself if parent fails
            # (e.g., merge commits, initial commits)
            print(f"  WARN: parent checkout failed, trying {commit_hash[:12]} directly")
            result = subprocess.run(
                ["git", "checkout", "--quiet", commit_hash],
                capture_output=True, text=True, timeout=30,
                cwd=str(dest),
            )
            if result.returncode != 0:
                print(f"  ERROR: git checkout {commit_hash[:12]} failed: {result.stderr.strip()[:200]}")
                return False

        # Remove .git directory to save space (we only need the source files)
        git_dir = dest / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

        return True

    except subprocess.TimeoutExpired:
        print(f"  ERROR: timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache R2E repos via git clone (FREE)")
    parser.add_argument("--cache_dir", type=str, default="repo-cache")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--clone_timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel git clone workers")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading R2E-Gym-Lite split={args.split}...")
    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split=args.split)
    entries = list(ds)
    if args.max_entries:
        entries = entries[: args.max_entries]
    print(f"Processing {len(entries)} entries")

    # Discover unique repos and find any unmapped ones
    repo_names = {e.get("repo_name", "unknown") for e in entries}
    unmapped = repo_names - set(REPO_MAP.keys())
    if unmapped:
        print(f"WARNING: {len(unmapped)} repo(s) not in REPO_MAP (will try auto-mapping): {unmapped}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    index_path = cache_dir / "index.json"
    index_lock = Lock()
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"Loaded existing index with {len(index)} entries")

    to_process = []
    skipped = 0
    for i, entry in enumerate(entries):
        instance_id = make_instance_id(entry)
        dest = cache_dir / instance_id
        if args.skip_existing and instance_id in index and dest.exists():
            skipped += 1
            continue
        if args.dry_run:
            print(f"[{i+1}/{len(entries)}] {instance_id}: would clone")
            continue
        to_process.append((i, entry))

    if args.dry_run:
        print(f"\nDry run: {len(to_process)} would be cached, {skipped} skipped")
        return

    print(f"\n{len(to_process)} to cache, {skipped} already cached")
    print(f"Using {args.workers} parallel workers\n")

    succeeded = 0
    failed = 0
    t_start = time.time()

    def _do_one(item):
        i, entry = item
        instance_id = make_instance_id(entry)
        repo_name = entry.get("repo_name", "unknown")
        commit_hash = entry.get("commit_hash", "unknown")
        dest = cache_dir / instance_id

        print(f"[{i + 1}/{len(entries)}] {instance_id}: cloning {repo_name}@{commit_hash[:12]}...")
        t0 = time.time()

        if dest.exists():
            shutil.rmtree(dest)

        ok = clone_and_checkout(repo_name, commit_hash, dest, timeout=args.clone_timeout)

        if ok:
            elapsed = time.time() - t0
            file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
            size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6

            entry_data = {
                "instance_id": instance_id,
                "repo_name": repo_name,
                "docker_image": entry.get("docker_image", ""),
                "commit_hash": entry.get("commit_hash", ""),
                "cache_path": str(dest),
                "file_count": file_count,
                "size_mb": round(size_mb, 1),
            }

            with index_lock:
                index[instance_id] = entry_data
                with open(index_path, "w") as f:
                    json.dump(index, f, indent=2)

            print(f"  OK [{instance_id}]: {file_count} files, {size_mb:.1f} MB, {elapsed:.1f}s")
            return True
        else:
            if dest.exists():
                shutil.rmtree(dest)
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_do_one, item): item for item in to_process}
        for future in as_completed(futures):
            try:
                if future.result():
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  EXCEPTION: {e}")

            done = succeeded + failed
            if done % 25 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed * 3600
                remaining = (len(to_process) - done) / (done / elapsed) if done else 0
                print(f"\n--- Progress: {done}/{len(to_process)} "
                      f"({succeeded} ok, {failed} fail) | "
                      f"{rate:.0f}/hr | ETA {remaining/60:.0f}min ---\n")

    total_elapsed = time.time() - t_start
    print(f"\nDone in {total_elapsed/60:.1f}min: "
          f"{succeeded} cached, {skipped} skipped, {failed} failed")
    print(f"Index: {index_path} ({len(index)} entries)")

    if index:
        total_mb = sum(e.get("size_mb", 0) for e in index.values())
        print(f"Total cache size: {total_mb / 1000:.1f} GB")


if __name__ == "__main__":
    main()
