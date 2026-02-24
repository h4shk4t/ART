#!/usr/bin/env python3
"""Verify git-based cache against existing Docker-based cache.

For each entry in the existing cache index, clones the repo via git
(checking out commit_hash~1), then compares the file trees to see if
the git version matches the Docker /testbed extraction.

This tells us whether commit_hash~1 gives us the correct buggy version.

Usage:
    uv run python verify_cache.py --cache_dir repo-cache --max_entries 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# Copy REPO_MAP from cache_repos_git.py
REPO_MAP: dict[str, str] = {
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


def get_github_url(repo_name: str) -> str:
    if repo_name in REPO_MAP:
        return f"https://github.com/{REPO_MAP[repo_name]}.git"
    return f"https://github.com/{repo_name}/{repo_name}.git"


def get_relative_files(directory: Path) -> dict[str, int]:
    """Get all files relative to directory, with their sizes."""
    files = {}
    skip_dirs = {".git", "__pycache__", ".venv", "node_modules", ".tox"}
    for root, dirs, filenames in os.walk(directory):
        # Prune skipped directories in-place (prevents descending into them)
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            if fname.endswith(".pyc"):
                continue
            fpath = Path(root) / fname
            try:
                rel = str(fpath.relative_to(directory))
                files[rel] = fpath.stat().st_size
            except (PermissionError, OSError):
                pass
    return files


def compare_trees(docker_dir: Path, git_dir: Path) -> dict:
    """Compare file trees between Docker-cached and git-cached repos."""
    docker_files = get_relative_files(docker_dir)
    git_files = get_relative_files(git_dir)

    docker_only = set(docker_files) - set(git_files)
    git_only = set(git_files) - set(docker_files)
    common = set(docker_files) & set(git_files)

    # Check for content differences in common files
    diff_files = []
    same_files = 0
    for f in sorted(common):
        docker_path = docker_dir / f
        git_path = git_dir / f
        try:
            if docker_path.read_bytes() == git_path.read_bytes():
                same_files += 1
            else:
                diff_files.append(f)
        except (PermissionError, OSError):
            diff_files.append(f)

    return {
        "docker_file_count": len(docker_files),
        "git_file_count": len(git_files),
        "common": len(common),
        "same_content": same_files,
        "diff_content": diff_files,
        "docker_only": sorted(docker_only)[:10],  # cap output
        "git_only": sorted(git_only)[:10],
        "docker_only_count": len(docker_only),
        "git_only_count": len(git_only),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify git cache against Docker cache")
    parser.add_argument("--cache_dir", type=str, default="repo-cache")
    parser.add_argument("--max_entries", type=int, default=5,
                        help="Max entries to verify (each requires a full git clone)")
    parser.add_argument("--clone_timeout", type=int, default=180)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.json"

    if not index_path.exists():
        print(f"No index at {index_path}")
        return

    with open(index_path) as f:
        index = json.load(f)

    print(f"Found {len(index)} entries in existing cache")
    print(f"Will verify up to {args.max_entries} entries\n")
    print("=" * 80)

    # Load dataset to get commit hashes
    from datasets import load_dataset
    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
    ds_map = {}
    for entry in ds:
        repo = entry.get("repo_name", "unknown")
        commit = entry.get("commit_hash", "unknown")
        iid = f"{repo}__{commit[:12]}"
        ds_map[iid] = entry

    verified = 0
    for instance_id, cache_entry in list(index.items()):
        if verified >= args.max_entries:
            break

        docker_dir = cache_dir / instance_id
        if not docker_dir.exists():
            print(f"\n⊘ {instance_id}: cache dir missing, skipping")
            continue

        ds_entry = ds_map.get(instance_id)
        if not ds_entry:
            print(f"\n⊘ {instance_id}: not found in dataset, skipping")
            continue

        repo_name = ds_entry["repo_name"]
        commit_hash = ds_entry["commit_hash"]

        print(f"\n{'─' * 80}")
        print(f"Verifying: {instance_id}")
        print(f"  Repo: {repo_name} | Commit (fix): {commit_hash[:12]}")
        print(f"  Docker cached at: {docker_dir}")

        # Clone to temp dir
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"verify_{instance_id}_"))
        github_url = get_github_url(repo_name)
        print(f"  Cloning {github_url}...")

        try:
            result = subprocess.run(
                ["git", "clone", "--quiet", github_url, str(tmp_dir / "repo")],
                capture_output=True, text=True, timeout=args.clone_timeout,
            )
            if result.returncode != 0:
                print(f"  ✗ Clone failed: {result.stderr.strip()[:200]}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            git_repo = tmp_dir / "repo"

            # Test THREE versions:
            # 1. commit_hash~1 (parent = our approach = should be buggy)
            # 2. commit_hash (fix commit = what we want to AVOID)
            # 3. Docker cache (ground truth)
            for label, checkout_ref in [
                ("commit~1 (parent=buggy)", f"{commit_hash}~1"),
                ("commit (fix=BAD)", commit_hash),
            ]:
                result = subprocess.run(
                    ["git", "checkout", "--quiet", checkout_ref],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(git_repo),
                )
                if result.returncode != 0:
                    print(f"  ✗ Checkout {label} failed: {result.stderr.strip()[:100]}")
                    continue

                cmp = compare_trees(docker_dir, git_repo)
                match_pct = (cmp["same_content"] / cmp["common"] * 100) if cmp["common"] else 0

                status = "✓" if match_pct > 90 else "△" if match_pct > 50 else "✗"
                print(f"\n  {status} {label}:")
                print(f"    Files: docker={cmp['docker_file_count']}, git={cmp['git_file_count']}, "
                      f"common={cmp['common']}")
                print(f"    Content match: {cmp['same_content']}/{cmp['common']} "
                      f"({match_pct:.1f}%)")
                if cmp["diff_content"]:
                    print(f"    Different content ({len(cmp['diff_content'])} files): "
                          f"{cmp['diff_content'][:5]}")
                if cmp["docker_only_count"]:
                    print(f"    Docker-only ({cmp['docker_only_count']}): {cmp['docker_only']}")
                if cmp["git_only_count"]:
                    print(f"    Git-only ({cmp['git_only_count']}): {cmp['git_only']}")

        except subprocess.TimeoutExpired:
            print(f"  ✗ Timed out")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        verified += 1

    print(f"\n{'=' * 80}")
    print(f"Verified {verified} entries")
    print("\nKey:")
    print("  ✓ = >90% file content match (safe to use)")
    print("  △ = 50-90% match (Docker image has extra setup)")
    print("  ✗ = <50% match (wrong version)")
    print("\nIf commit~1 (parent) matches better than commit (fix),")
    print("then the git-based approach with commit~1 is correct.")


if __name__ == "__main__":
    main()
