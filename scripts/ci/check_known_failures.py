#!/usr/bin/env python3
"""Split a pytest run into known failures and new ones.

The repository inherited a set of failing backend tests (stale expectations,
tests behind a feature flag, tests that need an external credential or a
browser). A gate that simply ran pytest would therefore be red forever, so
those failures are listed in `known_test_failures.txt` and reported
separately here: the job still fails, but only for failures nobody has
recorded yet.

    python scripts/ci/check_known_failures.py \
        --summary pytest-summary.txt \
        --junit pytest-report.xml \
        --known scripts/ci/known_test_failures.txt

Exit codes:
    0  only recorded (known) failures, or a clean run
    1  at least one failure that is not in the known list
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SUMMARY_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+?)(?:\s+-\s+.*)?$")


def normalize(node_id: str) -> str:
    """pytest prints OS-native separators; compare on forward slashes."""
    return node_id.replace("\\", "/").strip()


def parse_summary(path: Path) -> list[str]:
    """The failed/error node ids from pytest's short summary."""
    failures: list[str] = []
    if not path.exists():
        return failures
    # utf-8-sig: a summary written by a Windows shell can carry a BOM, which
    # would otherwise hide the first failure from the anchored pattern.
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = SUMMARY_LINE.match(line.strip())
        if match:
            failures.append(normalize(match.group(2)))
    return failures


def parse_junit(path: Path) -> tuple[set[str], set[str]]:
    """(failed node ids, passed node ids) from pytest's junit xml."""
    failed: set[str] = set()
    passed: set[str] = set()
    if not path.exists():
        return failed, passed
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return failed, passed
    cases = root.iter("testcase")
    for case in cases:
        file_attr = (case.get("file") or "").replace("\\", "/")
        name = case.get("name") or ""
        node_id = f"{file_attr}::{name}" if file_attr else name
        broken = any(child.tag in ("failure", "error") for child in case)
        (failed if broken else passed).add(normalize(node_id))
    return failed, passed


def load_known(path: Path) -> list[str]:
    known: list[str] = []
    if not path.exists():
        return known
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        known.append(normalize(line))
    return known


def write_summary(title: str, body: str) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"### {title}\n\n{body}\n")
    print(f"### {title}\n\n{body}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="pytest output with the short summary")
    parser.add_argument("--junit", type=Path, default=None, help="pytest --junitxml report")
    parser.add_argument("--known", required=True, type=Path, help="known-failures manifest")
    args = parser.parse_args()

    known = load_known(args.known)
    failures = parse_summary(args.summary)
    junit_failed, junit_passed = (set(), set())
    if args.junit:
        junit_failed, junit_passed = parse_junit(args.junit)

    # The summary is authoritative; junit only adds "passed" knowledge.
    observed = list(dict.fromkeys(failures))
    known_set = set(known)
    still_failing = [node for node in observed if node in known_set]
    new_failures = [node for node in observed if node not in known_set]
    # A recorded failure that now passes (junit ids carry a different path
    # prefix, so match on the trailing part of the node id).
    passed_tails = {node.rsplit("::", 1)[-1] for node in junit_passed}
    unexpectedly_passing = [
        node for node in known
        if node not in known_set.intersection(observed) and node.rsplit("::", 1)[-1] in passed_tails
    ]

    if observed:
        body = [
            f"**{len(observed)} failing test(s): {len(still_failing)} known, {len(new_failures)} new.**",
            "",
        ]
        if still_failing:
            body.append("<details><summary>Known failures (recorded, not blocking)</summary>\n")
            body += [f"- `{node}`" for node in still_failing]
            body += ["", "</details>", ""]
        if new_failures:
            body.append("<details open><summary>New failures (blocking)</summary>\n")
            body += [f"- `{node}`" for node in new_failures]
            body += ["", "</details>", ""]
        if unexpectedly_passing:
            body.append(
                "<details><summary>Recorded failures that now pass (manifest can be trimmed)</summary>\n"
            )
            body += [f"- `{node}`" for node in unexpectedly_passing]
            body += ["", "</details>", ""]
        write_summary("Backend test results", "\n".join(body))
    else:
        write_summary("Backend test results", "All backend tests passed.")

    if new_failures:
        print(f"\n{len(new_failures)} new failure(s) not in {args.known}:", file=sys.stderr)
        for node in new_failures:
            print(f"  {node}", file=sys.stderr)
        return 1
    if unexpectedly_passing:
        print(
            f"\nNote: {len(unexpectedly_passing)} recorded failure(s) now pass; "
            f"trim {args.known} to keep it accurate.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
