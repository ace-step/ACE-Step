#!/usr/bin/env python3
"""Run targeted commands with warnings enabled and flag deprecated usage."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMANDS: Tuple[Sequence[str], ...] = (
    (sys.executable, "-Wdefault", "infer.py", "--help"),
    (sys.executable, "-Wdefault", "trainer.py", "--help"),
    (sys.executable, "-Wdefault", "-m", "acestep.gui", "--help"),
)
WARNING_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"DeprecationWarning"),
    re.compile(r"\bwill be deprecated\b", re.IGNORECASE),
    re.compile(r"\bdeprecated\b", re.IGNORECASE),
)
IGNORE_SUBSTRINGS: Tuple[str, ...] = (
    "site-packages/spacy/",
    "site-packages/weasel/",
)


def collect_warnings(output: str) -> Iterable[str]:
    for line in output.splitlines():
        if any(ignore in line for ignore in IGNORE_SUBSTRINGS):
            continue
        if any(pattern.search(line) for pattern in WARNING_PATTERNS):
            yield line.strip()


def run_command(command: Sequence[str]) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "default")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit ACE-Step for deprecated usage emitted by key entry points."
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with status 1 if any deprecated usage is detected.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print the raw stdout/stderr for each command that was checked.",
    )
    args = parser.parse_args(argv)

    findings: List[Tuple[Sequence[str], str]] = []
    for command in DEFAULT_COMMANDS:
        code, stdout, stderr = run_command(command)
        command_display = " ".join(command)
        if code != 0:
            print(f"[ERROR] Command failed ({command_display})")
            if stdout:
                print(stdout)
            if stderr:
                print(stderr, file=sys.stderr)
            return code

        if args.show_output:
            if stdout:
                print(f"[STDOUT] {command_display}\n{stdout}")
            if stderr:
                print(f"[STDERR] {command_display}\n{stderr}", file=sys.stderr)

        for stream in (stdout, stderr):
            for warning in collect_warnings(stream):
                findings.append((command, warning))

    if findings:
        print("Deprecated usage detected:")
        for idx, (command, warning) in enumerate(findings, start=1):
            print(f"{idx}. {' '.join(command)} -> {warning}")
        return 1 if args.fail_on_warning else 0

    print("No deprecated usage detected across monitored commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
