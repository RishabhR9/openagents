#!/usr/bin/env python3
"""Judge a backend pytest run: known failures apart, and nothing hidden.

The repository inherited a set of failing backend tests (stale expectations,
tests behind a feature flag, tests that need an external credential or a
browser). A gate that simply ran pytest would therefore be red forever, so
those failures are listed in `known_test_failures.txt` and reported
separately: the job still fails, but only for failures nobody has recorded
yet.

The verdict is derived from BOTH the pytest output and pytest's own exit
status. Neither alone is enough:

  * The output alone cannot be trusted — a missing, empty or truncated file
    parses to "no failures", which would turn a crashed run into a green
    job. So an unreadable or unrecognisable summary is a failure of the
    gate, not a pass.
  * The exit status alone cannot be trusted either — pytest exits 1 for
    "tests failed", but also 2/3/4/5 for a run that never got as far as
    running tests (collection error, internal error, usage error, nothing
    collected). Those statuses are not a list of failing tests; they mean
    the run produced no usable result, and the gate fails on them.

    python scripts/ci/check_known_failures.py \
        --summary pytest-summary.txt \
        --junit pytest-report.xml \
        --known scripts/ci/known_test_failures.txt \
        --pytest-exit-code 1 \
        --pytest-rootdir workspace/backend

Exit codes:
    0  the run is green, or every failure it reported is a recorded one
    1  at least one failure that is not in the known list
    2  the run could not be judged (missing/empty/malformed summary, an
       unusable pytest exit status, or a summary that contradicts the
       exit status)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# "FAILED path::test" / "ERROR path::test" in pytest's short summary.
SUMMARY_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+?)(?:\s+-\s+.*)?$")
# The line pytest ends a run with: "22 failed, 2285 passed, ...", "no tests ran",
# "!!! Interrupted: 10 errors during collection !!!". Its presence is what
# proves the output is a pytest run at all. pytest decorates that line with
# "=" and, for an interrupted run, "!!!".
OUTCOME_LINE = re.compile(
    r"^[=\s!]*(?:"
    r"\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b"
    r"|no tests ran\b"
    r"|Interrupted:)",
    re.MULTILINE,
)

# pytest exit statuses (see _pytest.main.ExitCode).
EXIT_OK = 0
EXIT_TESTS_FAILED = 1
# 2 interrupted, 3 internal error, 4 usage error, 5 no tests collected.


@dataclass
class Decision:
    """What the checker concluded about a run."""

    status: str  # "ok" | "new-failures" | "unusable"
    exit_code: int  # what the CI job should exit with
    reasons: list[str] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)
    unexpectedly_passing: list[str] = field(default_factory=list)
    pytest_exit_code: int | None = None


def normalize(node_id: str) -> str:
    """pytest prints OS-native separators; compare on forward slashes."""
    return node_id.replace("\\", "/").strip()


def read_text(path: Path) -> str:
    """Read a text file whatever encoding it was written in.

    The summary is produced by a shell redirection: `>` writes UTF-16 under
    Windows PowerShell and UTF-8 under PowerShell 7, and an editor may add a
    UTF-8 BOM. The checker has to read all of them, or it will call a
    perfectly good run "malformed" depending on which shell ran it.
    """
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    return data.decode("utf-8", errors="replace")


def relativize(node_id: str, root: Path | None) -> str:
    """Make a node id repository-relative.

    pytest prints the path of the test file as it was given on the command
    line: relative when the suite is run as `pytest workspace/backend/tests/`,
    absolute when it is run with an absolute path. The manifest is written in
    the relative form, so an absolute id is made relative here rather than
    being reported as a new failure.
    """
    if root is None:
        return node_id
    head, sep, tail = node_id.partition("::")
    if not sep:
        return node_id
    candidate = Path(head)
    if not candidate.is_absolute():
        return node_id
    try:
        head = str(candidate.resolve().relative_to(root.resolve()))
    except ValueError:
        return node_id
    return normalize(f"{head}::{tail}")


def parse_summary(path: Path, root: Path | None = None) -> list[str]:
    """The failed/error node ids from pytest's short summary."""
    failures: list[str] = []
    if not path.exists():
        return failures
    for line in read_text(path).splitlines():
        match = SUMMARY_LINE.match(line.strip())
        if match:
            failures.append(relativize(normalize(match.group(2)), root))
    return failures


def _relative_file(file_attr: str, root: Path, pytest_rootdir: Path | None) -> str:
    """The junit `file` attribute as a repository-relative path.

    The attribute is relative to pytest's rootdir, not to the directory the
    run was started in — the suite's rootdir is `workspace/backend` (it holds
    pytest.ini), while the manifest is written relative to the repository
    root. `--pytest-rootdir` supplies that prefix so both sides speak the
    same dialect.
    """
    if not file_attr:
        return ""
    candidate = Path(file_attr)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root.resolve())
        except ValueError:
            return normalize(str(candidate))
        return normalize(str(candidate))
    if pytest_rootdir is not None:
        # The attribute is relative to pytest's rootdir — unless it already
        # is a repository path (rootdir == repo root). Disk, not assumption:
        # prepend only when the path exists under the rootdir and not
        # already under the repository root.
        under_root = root / candidate
        under_rootdir = root / pytest_rootdir / candidate
        if not under_root.exists() and under_rootdir.exists():
            candidate = under_rootdir
    try:
        candidate = candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return normalize(str(candidate))
    return normalize(str(candidate))


def _class_part(classname: str) -> str:
    """The class part of a junit classname ("tests.test_x.ClassA.ClassB")."""
    parts = classname.split(".") if classname else []
    classes: list[str] = []
    while parts and parts[-1][:1].isupper():
        classes.insert(0, parts.pop())
    return ".".join(classes)


def node_id_from_junit(
    case: ET.Element,
    root: Path,
    pytest_rootdir: Path | None = None,
) -> str:
    """The complete pytest node id of a junit testcase.

    pytest's default junit family ("xunit2") drops the file attribute, which
    leaves only the classname and the test name — enough to tell two tests
    apart by name but not by file. With `--junit-family=xunit1` the file is
    kept, and the node id can be rebuilt exactly: file, then the class, then
    the test name — the same shape pytest prints in its short summary.
    """
    file_part = _relative_file(case.get("file") or "", root, pytest_rootdir)
    class_part = _class_part(case.get("classname") or "")
    name = case.get("name") or ""
    parts = [part for part in (file_part, class_part, name) if part]
    return normalize("::".join(parts))


def parse_junit(
    path: Path,
    root: Path,
    pytest_rootdir: Path | None = None,
) -> tuple[set[str], set[str]]:
    """(failed node ids, passed node ids) from pytest's junit xml."""
    failed: set[str] = set()
    passed: set[str] = set()
    if not path.exists():
        return failed, passed
    try:
        root_xml = ET.parse(path).getroot()
    except ET.ParseError:
        return failed, passed
    for case in root_xml.iter("testcase"):
        node_id = node_id_from_junit(case, root, pytest_rootdir)
        if not node_id:
            continue
        broken = any(child.tag in ("failure", "error") for child in case)
        (failed if broken else passed).add(node_id)
    return failed, passed


def load_known(path: Path) -> list[str]:
    known: list[str] = []
    if not path.exists():
        return known
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        known.append(normalize(line))
    return known


def evaluate(
    *,
    summary_path: Path,
    known_path: Path,
    junit_path: Path | None = None,
    pytest_exit_code: int | None = None,
    root: Path | None = None,
    pytest_rootdir: Path | None = None,
) -> Decision:
    """Judge a pytest run. Pure: no printing, no exiting — the tests drive it."""
    root = root or default_root()

    def unusable(reason: str) -> Decision:
        return Decision(status="unusable", exit_code=2, reasons=[reason])

    if pytest_exit_code is None:
        return unusable("pytest's exit status was not provided, so the run cannot be judged")

    # The exit status is judged first: a status above 1 means the run itself
    # broke (interrupted, internal error, usage error, nothing collected), and
    # no summary can make that a pass.
    if pytest_exit_code not in (EXIT_OK, EXIT_TESTS_FAILED):
        return Decision(
            status="unusable",
            exit_code=2,
            reasons=[
                f"pytest exited {pytest_exit_code}, which is not a test failure "
                "(2 interrupted, 3 internal error, 4 usage error, 5 nothing collected). "
                "The run produced no usable result, so it cannot be judged."
            ],
            pytest_exit_code=pytest_exit_code,
        )

    if not summary_path.exists():
        return unusable(f"no pytest summary at {summary_path}")
    try:
        raw = read_text(summary_path)
    except OSError as error:
        return unusable(f"pytest summary could not be read: {error}")
    if not raw.strip():
        return unusable(f"pytest summary at {summary_path} is empty")
    if not OUTCOME_LINE.search(raw):
        return unusable(
            f"pytest summary at {summary_path} does not look like a pytest run "
            "(no outcome line) — it is truncated or was never written"
        )

    observed = list(dict.fromkeys(parse_summary(summary_path, root)))
    known = load_known(known_path)
    known_set = set(known)

    # A recorded failure that passed this time: the manifest can be trimmed.
    # Compared on the complete node id, so a test of the same name in another
    # file is never mistaken for it.
    passed: set[str] = set()
    if junit_path is not None:
        _, passed = parse_junit(junit_path, root, pytest_rootdir)
    unexpectedly_passing = [
        node for node in known
        if node not in observed and node in passed
    ]

    if pytest_exit_code == EXIT_OK:
        if observed:
            return Decision(
                status="unusable",
                exit_code=2,
                reasons=[
                    "pytest reported success but its summary lists failures — "
                    "the summary does not belong to this run"
                ],
                observed=observed,
                unexpectedly_passing=unexpectedly_passing,
                pytest_exit_code=pytest_exit_code,
            )
        return Decision(
            status="ok",
            exit_code=0,
            observed=[],
            unexpectedly_passing=unexpectedly_passing,
            pytest_exit_code=pytest_exit_code,
        )

    if not observed:
        return Decision(
            status="unusable",
            exit_code=2,
            reasons=[
                "pytest exited 1 but its summary lists no failing test — the "
                "summary is truncated or was written by something else"
            ],
            unexpectedly_passing=unexpectedly_passing,
            pytest_exit_code=pytest_exit_code,
        )

    still_failing = [node for node in observed if node in known_set]
    new_failures = [node for node in observed if node not in known_set]

    if new_failures:
        return Decision(
            status="new-failures",
            exit_code=1,
            reasons=[
                f"{len(new_failures)} failure(s) nobody has recorded yet"
            ],
            observed=observed,
            known_failures=still_failing,
            new_failures=new_failures,
            unexpectedly_passing=unexpectedly_passing,
            pytest_exit_code=pytest_exit_code,
        )

    return Decision(
        status="ok",
        exit_code=0,
        reasons=[
            f"pytest exited {pytest_exit_code}; all {len(observed)} failure(s) are recorded"
        ],
        observed=observed,
        known_failures=still_failing,
        new_failures=[],
        unexpectedly_passing=unexpectedly_passing,
        pytest_exit_code=pytest_exit_code,
    )


def default_root() -> Path:
    """The repository root (this file lives in <root>/scripts/ci/)."""
    return Path(__file__).resolve().parents[2]


def write_summary(title: str, body: str) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"### {title}\n\n{body}\n")
    print(f"### {title}\n\n{body}")


def report(decision: Decision, known_path: Path) -> int:
    """Print the verdict and return the exit code for the CI job."""
    if decision.status == "unusable":
        body = "\n".join(f"- {reason}" for reason in decision.reasons)
        write_summary("Backend test results — could not be judged", body)
        print("\n".join(decision.reasons), file=sys.stderr)
        return decision.exit_code

    if decision.status == "ok":
        if decision.pytest_exit_code == EXIT_OK:
            body = "pytest exited 0 — every backend test passed."
        else:
            body = [
                f"pytest exited {decision.pytest_exit_code}; "
                f"all {len(decision.observed)} failure(s) are recorded in `{known_path}`.",
                "",
                "<details><summary>Known failures (recorded, not blocking)</summary>\n",
            ]
            body += [f"- `{node}`" for node in decision.known_failures]
            body += ["", "</details>", ""]
        write_summary("Backend test results", "\n".join(body))
    else:
        body = [
            f"**{len(decision.observed)} failing test(s): "
            f"{len(decision.known_failures)} known, {len(decision.new_failures)} new.**",
            "",
        ]
        if decision.known_failures:
            body.append("<details><summary>Known failures (recorded, not blocking)</summary>\n")
            body += [f"- `{node}`" for node in decision.known_failures]
            body += ["", "</details>", ""]
        body.append("<details open><summary>New failures (blocking)</summary>\n")
        body += [f"- `{node}`" for node in decision.new_failures]
        body += ["", "</details>", ""]
        write_summary("Backend test results", "\n".join(body))

    if decision.unexpectedly_passing:
        write_summary(
            "Recorded failures that now pass (the manifest can be trimmed)",
            "\n".join(f"- `{node}`" for node in decision.unexpectedly_passing),
        )
        print(
            f"\nNote: {len(decision.unexpectedly_passing)} recorded failure(s) now pass; "
            f"trim {known_path} to keep it accurate.",
            file=sys.stderr,
        )

    if decision.status == "new-failures":
        print(
            f"\n{len(decision.new_failures)} new failure(s) not in {known_path}:",
            file=sys.stderr,
        )
        for node in decision.new_failures:
            print(f"  {node}", file=sys.stderr)
        return decision.exit_code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="pytest output with the short summary")
    parser.add_argument("--junit", type=Path, default=None, help="pytest --junitxml report")
    parser.add_argument("--known", required=True, type=Path, help="known-failures manifest")
    parser.add_argument(
        "--pytest-exit-code",
        type=int,
        default=None,
        help="the exit status pytest returned (required to judge the run)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root, used to make junit paths relative (default: the repo root)",
    )
    parser.add_argument(
        "--pytest-rootdir",
        type=Path,
        default=None,
        help=(
            "pytest's rootdir, which the junit report's file attributes are "
            "relative to (default: the repository root)"
        ),
    )
    args = parser.parse_args()

    decision = evaluate(
        summary_path=args.summary,
        known_path=args.known,
        junit_path=args.junit,
        pytest_exit_code=args.pytest_exit_code,
        root=args.root,
        pytest_rootdir=args.pytest_rootdir,
    )
    return report(decision, args.known)


if __name__ == "__main__":
    sys.exit(main())
