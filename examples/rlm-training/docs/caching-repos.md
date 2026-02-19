# Caching Repos from R2E-Gym-Lite

This walkthrough explains how the repo cache works and how to use it.

## Why cache?

During training, each rollout needs a local copy of the repository to operate on. Without caching, you'd need Docker access on the training node to extract files. With caching:

- Repos are extracted once from Docker images on the Docker Node
- Copied to the Training Node's local storage
- Each rollout does a fast `cp -r` from cache (~50ms) instead of a network fetch

## Dataset overview

R2E-Gym-Lite (`R2E-Gym/R2E-Gym-Lite`) has these splits:

| Split | Entries | Description |
|-------|---------|-------------|
| `train` | 4578 | Full training set |
| `dev_10pr_v1` | 100 | Small dev set (good for testing) |
| `dev_100pr_v1` | 1000 | Medium dev set |

Each entry has a unique Docker image (`namanjain12/{repo}_final:{commit_hash}`). There are 10 unique repos but 4578 unique commits.

### Key dataset fields

| Field | Example | Used for |
|-------|---------|----------|
| `repo_name` | `django` | Instance ID prefix |
| `docker_image` | `namanjain12/django_final:abc123...` | Container to extract /testbed from |
| `commit_hash` | `abc123def456...` | Instance ID suffix |
| `problem_statement` | `[ISSUE] Unable to...` | Bug report shown to the model |
| `expected_output_json` | `{"TestFoo.test_bar": "PASSED"}` | Reward computation |

Note: The dataset does not have an `instance_id` field. We derive one as `{repo_name}__{commit_hash[:12]}`.

## Running the cache script

### Prerequisites

- Docker access on the machine you run this on
- The R2E-Gym Docker images must be pullable (they're on Docker Hub)
- Enough disk space (estimate ~2-5 MB per entry, so ~10-25 GB total for the full train set)

### Quick start (small test)

```bash
# Cache just 3 entries to verify everything works
python cache_repos.py --cache_dir /data/repo-cache --max_entries 3
```

### Full cache

```bash
# Cache the entire training set
python cache_repos.py --cache_dir /data/repo-cache --split train

# Or just the dev set for initial experiments
python cache_repos.py --cache_dir /data/repo-cache --split dev_10pr_v1
```

### Resume after interruption

The script writes `index.json` after each successful extraction. With `--skip_existing` (on by default), it skips entries already in the index:

```bash
# Safe to re-run -- picks up where it left off
python cache_repos.py --cache_dir /data/repo-cache
```

### Dry run

See what would be cached without doing it:

```bash
python cache_repos.py --cache_dir /data/repo-cache --dry_run
```

## What gets created

```
/data/repo-cache/
  index.json                          # Maps instance_id -> metadata
  aiohttp__f0d74880deec/              # Extracted /testbed contents
    setup.py
    aiohttp/
    tests/
    ...
  django__abc123def456/
    manage.py
    django/
    tests/
    ...
```

### index.json format

```json
{
  "aiohttp__f0d74880deec": {
    "instance_id": "aiohttp__f0d74880deec",
    "repo_name": "aiohttp",
    "docker_image": "namanjain12/aiohttp_final:f0d74880deec...",
    "commit_hash": "f0d74880deec...",
    "cache_path": "/data/repo-cache/aiohttp__f0d74880deec",
    "file_count": 342,
    "size_mb": 4.2
  }
}
```

## Transferring to Training Node

If the Docker Node and Training Node are different machines:

```bash
# Option 1: rsync
rsync -avz /data/repo-cache/ training-node:/data/repo-cache/

# Option 2: tar + scp
cd /data && tar czf repo-cache.tar.gz repo-cache/
scp repo-cache.tar.gz training-node:/data/
ssh training-node "cd /data && tar xzf repo-cache.tar.gz"
```

## What you need to change

| What | File | Change |
|------|------|--------|
| Cache directory path | `config.py` | `repo_cache_dir = "/data/repo-cache"` |
| Dataset split | `cache_repos.py` CLI | `--split dev_10pr_v1` |
| Instance ID derivation | `cache_repos.py` | `make_instance_id()` function |
| Docker cp timeout | `cache_repos.py` CLI | `--cp_timeout 120` |
