"""Reward functions for R2E-Gym training rollouts.

Each function takes (test_log_output: str, dataset_entry: dict) -> float.
Swap via config.reward_fn.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return re.sub(r"\x1b\[\d+m", "", text)


def _decolor_dict(d: dict[str, str]) -> dict[str, str]:
    return {_strip_ansi(k): v for k, v in d.items()}


def _parse_pytest_log(log: str) -> dict[str, str]:
    """Extract test name -> status from pytest short test summary.

    Handles edge cases: multiple occurrences of the summary header,
    malformed lines, and lines without '::' separators.
    """
    if not log or "short test summary info" not in log:
        return {}
    # Use the last occurrence in case the header appears multiple times
    parts = log.split("short test summary info")
    summary = parts[-1].strip()
    result: dict[str, str] = {}
    for line in summary.split("\n"):
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if "PASSED" in line:
            test_parts = line.split("::")
            if len(test_parts) > 1:
                result[".".join(test_parts[1:])] = "PASSED"
        elif "FAILED" in line:
            test_parts = line.split("::")
            if len(test_parts) > 1:
                name = ".".join(test_parts[1:]).split(" - ")[0].strip()
                result[name] = "FAILED"
        elif "ERROR" in line:
            test_parts = line.split("::")
            if len(test_parts) > 1:
                name = ".".join(test_parts[1:]).split(" - ")[0].strip()
            else:
                name = line.split(" - ")[0].strip()
            if name:
                result[name] = "ERROR"
    return result


def _normalize_test_keys(d: dict[str, str]) -> dict[str, str]:
    """Strip trailing error descriptions from test keys for comparison."""
    return {k.split(" - ")[0]: v for k in sorted(d.keys()) if (v := d[k])}


def binary_test_reward(log_output: str, ds: dict[str, Any]) -> float:
    """Binary reward: 1.0 if all tests match expected output, 0.0 otherwise.

    Parses the pytest short test summary from log_output and compares against
    the expected_output_json field in the dataset entry.
    """
    parsed = _decolor_dict(_parse_pytest_log(log_output))
    expected_json = ds.get("expected_output_json")
    if not expected_json:
        return 0.0

    try:
        expected_raw: dict[str, str] = json.loads(expected_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    expected = _decolor_dict(expected_raw)

    parsed = _normalize_test_keys(parsed)
    expected = _normalize_test_keys(expected)

    if not parsed and not expected:
        return 1.0

    if len(parsed) != len(expected):
        return 0.0

    for key in parsed:
        if not key:
            continue
        if key not in expected:
            return 0.0
        if parsed[key] != expected[key]:
            return 0.0

    return 1.0


def partial_test_reward(log_output: str, ds: dict[str, Any]) -> float:
    """Partial reward: fraction of tests that match expected status.

    Useful for experiments where you want smoother reward signal.
    """
    parsed = _decolor_dict(_parse_pytest_log(log_output))
    expected_json = ds.get("expected_output_json")
    if not expected_json:
        return 0.0

    try:
        expected_raw: dict[str, str] = json.loads(expected_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    expected = _decolor_dict(expected_raw)

    parsed = _normalize_test_keys(parsed)
    expected = _normalize_test_keys(expected)

    if not expected:
        return 0.0

    matching = 0
    for key, status in expected.items():
        if not key:
            continue
        if parsed.get(key) == status:
            matching += 1

    return matching / len(expected)
