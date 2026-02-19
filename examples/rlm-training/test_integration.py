"""Integration tests for the distributed RLM training pipeline.

Tests all fixes from the pipeline audit:
  1. Cache index loading (dict structure with cache_path)
  2. Scaffold restoration (model can't overwrite tools)
  3. Safe builtins (eval/exec/input blocked)
  4. Thread-safe output capture
  5. run_tests() returns informative no-op
  6. Reward parsing edge cases
  7. Binary file filtering in get_modified_files
  8. RolloutGuard tracking
  9. _copy_from_cache returns (work_dir, cache_src) tuple
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

# ---- Test helpers ----

_passed = 0
_failed = 0


def _make_test_repl():
    """Create a minimal LocalREPL for testing (no model/trajectory needed)."""
    from repl import LocalREPL, _SAFE_BUILTINS
    from health import HealthMonitor

    work_dir = tempfile.mkdtemp(prefix="test_repl_")
    os.makedirs(os.path.join(work_dir, "src"), exist_ok=True)
    Path(os.path.join(work_dir, "src", "main.py")).write_text("x = 1\n")

    repl = LocalREPL.__new__(LocalREPL)
    repl.work_dir = Path(work_dir)
    repl._health = HealthMonitor()
    repl._max_output_chars = 3000
    repl._finished = False
    repl._pending_sub_calls = []
    repl._depth = 0
    repl._max_depth = 2
    repl._max_sub_agent_steps = 8
    repl._max_completion_tokens = 4000
    repl._sub_agent_system_prompt = ""

    import threading as _threading
    repl._exec_lock = _threading.Lock()

    repl._tool_bindings = {
        "ls": repl.ls,
        "read": repl.read,
        "grep": repl.grep,
        "apply_patch": repl.apply_patch,
        "run_tests": repl.run_tests,
        "bash": repl.bash,
        "finish": repl.finish,
        "llm_query": lambda prompt, max_steps=8: "stub",
        "llm_query_batched": lambda prompts, max_steps=8: ["stub"],
    }
    repl._globals = {
        "__builtins__": _SAFE_BUILTINS.copy(),
        "__name__": "__repl__",
        **repl._tool_bindings,
    }

    return repl


def test(name: str):
    """Decorator to register and run a test."""
    def wrapper(fn):
        global _passed, _failed
        try:
            fn()
            print(f"  PASS: {name}")
            _passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            _failed += 1
        return fn
    return wrapper


# ===========================================================================
# 1. Cache index loading
# ===========================================================================
print("\n=== Cache Index Loading ===")


@test("_copy_from_cache handles dict index entries")
def _():
    from rollout import _copy_from_cache

    with tempfile.TemporaryDirectory() as cache_dir:
        # Create a cached repo
        repo_dir = Path(cache_dir) / "repos" / "django__abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "main.py").write_text("x = 1\n")

        # Create index.json with dict entries (as cache_repos.py produces)
        index = {
            "django__abc123": {
                "instance_id": "django__abc123",
                "cache_path": str(repo_dir),
                "repo_name": "django",
                "file_count": 1,
            }
        }
        with open(Path(cache_dir) / "index.json", "w") as f:
            json.dump(index, f)

        work_dir, cache_src = _copy_from_cache("django__abc123", cache_dir)
        assert work_dir.exists()
        assert (work_dir / "main.py").exists()
        assert (work_dir / "main.py").read_text() == "x = 1\n"
        assert cache_src == repo_dir
        # Cleanup
        import shutil
        shutil.rmtree(str(work_dir))


@test("_copy_from_cache handles plain string index entries (backward compat)")
def _():
    from rollout import _copy_from_cache

    with tempfile.TemporaryDirectory() as cache_dir:
        repo_dir = Path(cache_dir) / "repos" / "flask__def456"
        repo_dir.mkdir(parents=True)
        (repo_dir / "app.py").write_text("y = 2\n")

        index = {"flask__def456": str(repo_dir)}
        with open(Path(cache_dir) / "index.json", "w") as f:
            json.dump(index, f)

        work_dir, cache_src = _copy_from_cache("flask__def456", cache_dir)
        assert work_dir.exists()
        assert (work_dir / "app.py").exists()
        assert cache_src == repo_dir
        import shutil
        shutil.rmtree(str(work_dir))


@test("_copy_from_cache raises for unknown instance_id")
def _():
    from rollout import _copy_from_cache

    with tempfile.TemporaryDirectory() as cache_dir:
        with open(Path(cache_dir) / "index.json", "w") as f:
            json.dump({}, f)
        try:
            _copy_from_cache("nonexistent", cache_dir)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass


# ===========================================================================
# 2. Scaffold restoration
# ===========================================================================
print("\n=== Scaffold Restoration ===")


@test("model can't permanently overwrite ls")
def _():
    from repl import LocalREPL
    repl = _make_test_repl()
    # Overwrite ls with a string
    repl.execute('ls = "overwritten"')
    # ls should be restored
    result = repl.execute('print(type(ls))')
    assert "method" in result.stdout or "function" in result.stdout, \
        f"ls not restored: {result.stdout}"


@test("model can't permanently overwrite llm_query")
def _():
    from repl import LocalREPL
    repl = _make_test_repl()
    repl.execute('llm_query = 42')
    result = repl.execute('print(callable(llm_query))')
    assert "True" in result.stdout, f"llm_query not restored: {result.stdout}"


@test("model can't permanently overwrite finish")
def _():
    from repl import LocalREPL
    repl = _make_test_repl()
    repl.execute('finish = None')
    result = repl.execute('print(callable(finish))')
    assert "True" in result.stdout, f"finish not restored: {result.stdout}"


# ===========================================================================
# 3. Safe builtins
# ===========================================================================
print("\n=== Safe Builtins ===")


@test("eval() is blocked")
def _():
    repl = _make_test_repl()
    result = repl.execute('x = eval("1+1")')
    assert "TypeError" in result.stderr or "NoneType" in result.stderr, \
        f"eval should be blocked: {result.stderr}"


@test("exec() is blocked")
def _():
    repl = _make_test_repl()
    result = repl.execute('exec("x = 1")')
    assert "TypeError" in result.stderr or "NoneType" in result.stderr, \
        f"exec should be blocked: {result.stderr}"


@test("input() is blocked")
def _():
    repl = _make_test_repl()
    result = repl.execute('x = input("prompt")')
    assert "TypeError" in result.stderr or "NoneType" in result.stderr, \
        f"input should be blocked: {result.stderr}"


@test("import still works")
def _():
    repl = _make_test_repl()
    result = repl.execute('import json\nprint(json.dumps({"a": 1}))')
    assert '"a": 1' in result.stdout


@test("print() works")
def _():
    repl = _make_test_repl()
    result = repl.execute('print("hello safe builtins")')
    assert "hello safe builtins" in result.stdout


# ===========================================================================
# 4. Thread-safe output capture
# ===========================================================================
print("\n=== Thread-safe Output Capture ===")


@test("concurrent executions don't interleave output")
def _():
    import time

    repl1 = _make_test_repl()
    repl2 = _make_test_repl()

    results = [None, None]

    def run1():
        results[0] = repl1.execute('print("AAA" * 100)')

    def run2():
        results[1] = repl2.execute('print("BBB" * 100)')

    t1 = threading.Thread(target=run1)
    t2 = threading.Thread(target=run2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert "AAA" in results[0].stdout
    assert "BBB" in results[1].stdout
    assert "BBB" not in results[0].stdout, "Output interleaved!"
    assert "AAA" not in results[1].stdout, "Output interleaved!"


# ===========================================================================
# 5. run_tests() no-op
# ===========================================================================
print("\n=== run_tests() No-Op ===")


@test("run_tests returns informative message")
def _():
    repl = _make_test_repl()
    result = repl.execute('print(run_tests())')
    assert "Docker" in result.stdout or "reward time" in result.stdout, \
        f"Expected Docker/reward message: {result.stdout}"


# ===========================================================================
# 6. Reward parsing edge cases
# ===========================================================================
print("\n=== Reward Parsing ===")


@test("binary_test_reward with matching results")
def _():
    from rewards import binary_test_reward
    # Parser extracts parts after first :: and joins with .
    # So "tests/test_main.py::TestFoo::test_one" -> "TestFoo.test_one"
    # Expected JSON must use same format as what the parser produces
    log = """
=== short test summary info ===
PASSED tests/test_main.py::TestFoo::test_one
PASSED tests/test_main.py::TestFoo::test_two
=== 2 passed ===
"""
    ds = {"expected_output_json": json.dumps({
        "tests/test_main.py::TestFoo::test_one": "PASSED",
        "tests/test_main.py::TestFoo::test_two": "PASSED",
    })}
    # The parsed keys are "TestFoo.test_one", expected keys are the full paths.
    # These won't match unless we normalize both sides the same way.
    # In the real dataset, expected_output_json uses the same format as the
    # test output. Let's test with the actual parsed format:
    ds2 = {"expected_output_json": json.dumps({
        "TestFoo.test_one": "PASSED",
        "TestFoo.test_two": "PASSED",
    })}
    assert binary_test_reward(log, ds2) == 1.0


@test("binary_test_reward with mismatch")
def _():
    from rewards import binary_test_reward
    log = """
=== short test summary info ===
FAILED tests/test_main.py::TestFoo::test_one - AssertionError
=== 1 failed ===
"""
    ds = {"expected_output_json": json.dumps({
        "TestFoo.test_one": "PASSED",
    })}
    assert binary_test_reward(log, ds) == 0.0


@test("binary_test_reward with malformed JSON")
def _():
    from rewards import binary_test_reward
    ds = {"expected_output_json": "not valid json {{{"}
    assert binary_test_reward("some output", ds) == 0.0


@test("binary_test_reward with empty expected and empty parsed")
def _():
    from rewards import binary_test_reward
    ds = {"expected_output_json": "{}"}
    assert binary_test_reward("no test summary here", ds) == 1.0


@test("binary_test_reward with no expected_output_json")
def _():
    from rewards import binary_test_reward
    assert binary_test_reward("any output", {}) == 0.0


@test("_parse_pytest_log handles multiple summary headers")
def _():
    from rewards import _parse_pytest_log
    log = """
=== short test summary info ===
PASSED old/test.py::test_old
=== some separator ===
=== short test summary info ===
PASSED new/test.py::test_new
=== 1 passed ===
"""
    result = _parse_pytest_log(log)
    # Should use the last occurrence
    assert "test_new" in str(result)


@test("_parse_pytest_log with empty log")
def _():
    from rewards import _parse_pytest_log
    assert _parse_pytest_log("") == {}
    assert _parse_pytest_log(None) == {}


@test("partial_test_reward gives fractional score")
def _():
    from rewards import partial_test_reward
    log = """
=== short test summary info ===
PASSED tests/test.py::test_a
FAILED tests/test.py::test_b - error
=== 1 passed, 1 failed ===
"""
    # Use parsed format: parts after first :: joined with .
    ds = {"expected_output_json": json.dumps({
        "test_a": "PASSED",
        "test_b": "PASSED",
    })}
    reward = partial_test_reward(log, ds)
    assert reward == 0.5, f"Expected 0.5, got {reward}"


# ===========================================================================
# 7. Binary file filtering
# ===========================================================================
print("\n=== Binary File Filtering ===")


@test("get_modified_files skips .pyc files")
def _():
    from docker_client import get_modified_files

    with tempfile.TemporaryDirectory() as work_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        # Create source file (should be included)
        (Path(work_dir) / "main.py").write_text("changed")
        (Path(cache_dir) / "main.py").write_text("original")
        # Create .pyc file (should be skipped)
        pycache = Path(work_dir) / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

        modified = get_modified_files(work_dir, cache_dir)
        assert "main.py" in modified
        assert not any(".pyc" in k for k in modified), f"Should skip .pyc: {list(modified.keys())}"
        assert not any("__pycache__" in k for k in modified)


@test("get_modified_files skips .so files")
def _():
    from docker_client import get_modified_files

    with tempfile.TemporaryDirectory() as work_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        (Path(work_dir) / "module.so").write_bytes(b"\x7fELF")
        modified = get_modified_files(work_dir, cache_dir)
        assert len(modified) == 0


@test("get_modified_files includes new source files")
def _():
    from docker_client import get_modified_files

    with tempfile.TemporaryDirectory() as work_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        (Path(work_dir) / "new_file.py").write_text("new code")
        modified = get_modified_files(work_dir, cache_dir)
        assert "new_file.py" in modified


# ===========================================================================
# 8. RolloutGuard tracking
# ===========================================================================
print("\n=== RolloutGuard Tracking ===")


@test("track_rollout / unregister_rollout work")
def _():
    from health import HealthMonitor
    hm = HealthMonitor()
    hm.track_rollout("/tmp/test_rollout_1")
    assert "/tmp/test_rollout_1" in hm._active_rollout_dirs
    hm.unregister_rollout("/tmp/test_rollout_1")
    assert "/tmp/test_rollout_1" not in hm._active_rollout_dirs


@test("register_rollout context manager tracks and cleans up")
def _():
    from health import HealthMonitor
    hm = HealthMonitor()

    with tempfile.TemporaryDirectory() as work_dir:
        with hm.register_rollout(work_dir) as guard:
            assert str(work_dir) in hm._active_rollout_dirs
        assert str(work_dir) not in hm._active_rollout_dirs


# ===========================================================================
# 9. Misc integration checks
# ===========================================================================
print("\n=== Misc Integration ===")


@test("Scenario.from_dataset_entry builds correct instance_id")
def _():
    from rollout import Scenario
    ds = {
        "repo_name": "django/django",
        "commit_hash": "abc123def456789",
        "docker_image": "img:latest",
        "problem_statement": "bug",
    }
    s = Scenario.from_dataset_entry(ds)
    assert s.instance_id == "django/django__abc123def456"
    assert s.docker_image == "img:latest"


@test("ExperimentConfig.to_metadata excludes callables")
def _():
    from config import ExperimentConfig
    cfg = ExperimentConfig(system_prompt_fn=lambda: "x")
    meta = cfg.to_metadata()
    assert "system_prompt_fn" not in meta
    assert "model_name" in meta


@test("_should_skip correctly identifies patterns")
def _():
    from docker_client import _should_skip
    assert _should_skip("__pycache__/main.cpython-311.pyc")
    assert _should_skip("src/__pycache__/mod.pyc")
    assert _should_skip("lib/module.so")
    assert _should_skip(".git/objects/abc")
    assert _should_skip("pkg.egg-info/PKG-INFO")
    assert not _should_skip("src/main.py")
    assert not _should_skip("README.md")
    assert not _should_skip("tests/test_foo.py")


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print(f"  Results: {_passed} passed, {_failed} failed")
print(f"{'='*60}")
sys.exit(1 if _failed > 0 else 0)
