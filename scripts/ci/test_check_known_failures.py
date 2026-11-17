"""Tests for the CI checker that judges a backend pytest run.

The checker decides whether a pull request may land, so its own failure
modes matter: a missing, empty or truncated summary must never be mistaken
for a green run, and a pytest exit status that means "the run broke" must
never be read as "no failures".

    python -m pytest scripts/ci/test_check_known_failures.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CHECKER_PATH = Path(__file__).with_name("check_known_failures.py")
_spec = importlib.util.spec_from_file_location("check_known_failures", CHECKER_PATH)
checker = importlib.util.module_from_spec(_spec)
# Registered before it runs: @dataclass resolves its own module through
# sys.modules, and a module loaded this way is not there yet.
sys.modules["check_known_failures"] = checker
_spec.loader.exec_module(checker)

KNOWN_A = "workspace/backend/tests/test_a.py::test_known"
KNOWN_B = "workspace/backend/tests/test_b.py::TestClass::test_known"


def write_summary(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summary_with_failures(path: Path, *node_ids: str) -> Path:
    lines = [
        "..........................................F............................ [ 50%]",
        "=========================== short test summary info ===========================",
    ]
    lines += [f"FAILED {node}" for node in node_ids]
    lines.append("========= 1 failed, 10 passed, 2 skipped in 1.23s =========")
    return write_summary(path, *lines)


def summary_all_passing(path: Path) -> Path:
    return write_summary(
        path,
        "......................................................... [100%]",
        "======================== 12 passed, 2 skipped in 1.23s =========================",
    )


def write_known(path: Path, *node_ids: str) -> Path:
    path.write_text(
        "# recorded failures\n" + "\n".join(node_ids) + "\n",
        encoding="utf-8",
    )
    return path


def write_junit(path: Path, cases: list[tuple[str, str, str, bool]]) -> Path:
    """cases: (classname, name, file, failed)"""
    body = ["<testsuites><testsuite>"]
    for classname, name, file_attr, failed in cases:
        body.append(
            f'<testcase classname="{classname}" name="{name}" file="{file_attr}"'
            + ("><failure message=\"boom\"/></testcase>" if failed else "/>")
        )
    body.append("</testsuite></testsuites>")
    path.write_text("".join(body), encoding="utf-8")
    return path


def judge(tmp_path: Path, summary: Path, known: Path, exit_code, junit: Path | None = None):
    return checker.evaluate(
        summary_path=summary,
        known_path=known,
        junit_path=junit,
        pytest_exit_code=exit_code,
        root=tmp_path,
    )


# ── 1. a missing summary ─────────────────────────────────────────────────────

def test_missing_summary_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    decision = judge(tmp_path, tmp_path / "does-not-exist.txt", known, 1)
    assert decision.status == "unusable"
    assert decision.exit_code == 2
    assert "no pytest summary" in decision.reasons[0]


# ── 2. an empty summary ──────────────────────────────────────────────────────

def test_empty_summary_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = write_summary(tmp_path / "summary.txt", "")
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "unusable"
    assert decision.exit_code == 2
    assert "empty" in decision.reasons[0]


def test_whitespace_only_summary_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = write_summary(tmp_path / "summary.txt", "   ", "\t")
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "unusable"
    assert decision.exit_code == 2


# ── 3. a malformed summary ───────────────────────────────────────────────────

def test_malformed_summary_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = write_summary(tmp_path / "summary.txt", "this is not a pytest run")
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "unusable"
    assert decision.exit_code == 2
    assert "does not look like a pytest run" in decision.reasons[0]


def test_truncated_summary_fails(tmp_path):
    """A summary cut off before the outcome line cannot be trusted."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = write_summary(
        tmp_path / "summary.txt",
        "..........................................F........ [ 50%]",
        "=========================== short test summary info ===========================",
        f"FAILED {KNOWN_A}",
    )
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "unusable"
    assert decision.exit_code == 2


def test_summary_without_failures_but_failing_exit_fails(tmp_path):
    """pytest exited 1 but the summary lists no failure — the two disagree."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_all_passing(tmp_path / "summary.txt")
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "unusable"
    assert decision.exit_code == 2


# ── 4. a pytest startup / import failure ─────────────────────────────────────

def test_pytest_startup_failure_fails(tmp_path):
    """A usage error (exit 4) means no tests ran — never a pass."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = write_summary(
        tmp_path / "summary.txt",
        "ERROR: file or directory not found: workspace/backend/tests/",
        "usage: pytest [options] [file_or_dir]",
    )
    decision = judge(tmp_path, summary, known, 4)
    assert decision.status == "unusable"
    assert decision.exit_code == 2
    assert "pytest exited 4" in decision.reasons[0]


def test_collection_error_fails(tmp_path):
    """An import error during collection interrupts the run (exit 2)."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = write_summary(
        tmp_path / "summary.txt",
        "ImportError while importing test module '/repo/workspace/backend/tests/test_a.py'.",
        "!!!! Interrupted: 1 error during collection !!!!",
        "1 error in 0.42s",
    )
    decision = judge(tmp_path, summary, known, 2)
    assert decision.status == "unusable"
    assert decision.exit_code == 2


def test_missing_exit_status_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A)
    decision = judge(tmp_path, summary, known, None)
    assert decision.status == "unusable"
    assert decision.exit_code == 2
    assert "exit status was not provided" in decision.reasons[0]


# ── 5. known failures only ───────────────────────────────────────────────────

def test_known_failures_only_passes(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A, KNOWN_B)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A, KNOWN_B)
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "ok"
    assert decision.exit_code == 0
    assert decision.new_failures == []
    assert sorted(decision.known_failures) == sorted([KNOWN_A, KNOWN_B])


def test_known_failures_only_is_still_reported(tmp_path):
    """A pass must not hide the failures it recorded."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A)
    decision = judge(tmp_path, summary, known, 1)
    assert KNOWN_A in decision.known_failures
    assert KNOWN_A in decision.observed


# ── 6. a new, unexpected failure ─────────────────────────────────────────────

def test_new_failure_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A, KNOWN_B)
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "new-failures"
    assert decision.exit_code == 1
    assert decision.new_failures == [KNOWN_B]
    assert decision.known_failures == [KNOWN_A]


def test_new_failure_alone_fails(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_B)
    decision = judge(tmp_path, summary, known, 1)
    assert decision.status == "new-failures"
    assert decision.exit_code == 1


# ── 7. everything passes ─────────────────────────────────────────────────────

def test_all_tests_passing_passes(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_all_passing(tmp_path / "summary.txt")
    decision = judge(tmp_path, summary, known, 0)
    assert decision.status == "ok"
    assert decision.exit_code == 0
    assert decision.observed == []


def test_success_exit_with_failures_in_summary_fails(tmp_path):
    """The summary and the exit status must agree."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A)
    decision = judge(tmp_path, summary, known, 0)
    assert decision.status == "unusable"
    assert decision.exit_code == 2


# ── node ids are compared in full ────────────────────────────────────────────

def test_unexpectedly_passing_matches_the_complete_node_id(tmp_path):
    """A recorded failure that passed this time is reported — by full id."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_all_passing(tmp_path / "summary.txt")
    junit = write_junit(
        tmp_path / "report.xml",
        [("tests.test_a", "test_known", "workspace/backend/tests/test_a.py", False)],
    )
    decision = judge(tmp_path, summary, known, 0, junit)
    assert decision.unexpectedly_passing == [KNOWN_A]


def test_same_test_name_in_another_file_is_not_a_match(tmp_path):
    """The trailing name alone must never decide the match."""
    known = write_known(tmp_path / "known.txt", "workspace/backend/tests/test_other.py::test_known")
    summary = summary_all_passing(tmp_path / "summary.txt")
    junit = write_junit(
        tmp_path / "report.xml",
        [("tests.test_a", "test_known", "workspace/backend/tests/test_a.py", False)],
    )
    decision = judge(tmp_path, summary, known, 0, junit)
    assert decision.unexpectedly_passing == []


def test_junit_class_part_is_kept_in_the_node_id(tmp_path):
    known = write_known(tmp_path / "known.txt", KNOWN_B)
    summary = summary_all_passing(tmp_path / "summary.txt")
    junit = write_junit(
        tmp_path / "report.xml",
        [("tests.test_b.TestClass", "test_known", "workspace/backend/tests/test_b.py", False)],
    )
    decision = judge(tmp_path, summary, known, 0, junit)
    assert decision.unexpectedly_passing == [KNOWN_B]


def test_junit_failure_matches_the_summary_node_id(tmp_path):
    """A failure in the junit report and in the summary is the same test."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A)
    junit = write_junit(
        tmp_path / "report.xml",
        [("tests.test_a", "test_known", "workspace/backend/tests/test_a.py", True)],
    )
    decision = judge(tmp_path, summary, known, 1, junit)
    assert decision.new_failures == []
    assert decision.known_failures == [KNOWN_A]


# ── the junit file attribute is rootdir-relative ─────────────────────────────

@pytest.fixture
def repo_test_file(tmp_path):
    """The test file, on disk, where the repository layout says it lives."""
    real = tmp_path / "workspace" / "backend" / "tests" / "test_a.py"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("# a test module", encoding="utf-8")
    return real


def test_junit_paths_relative_to_the_pytest_rootdir_are_relativized(tmp_path, repo_test_file):
    """pytest writes junit file attrs relative to its rootdir, not the CWD.

    The suite's rootdir is `workspace/backend` (it holds pytest.ini), so the
    report says `tests/test_a.py` where the manifest says
    `workspace/backend/tests/test_a.py`. `--pytest-rootdir` bridges them.
    """
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_all_passing(tmp_path / "summary.txt")
    junit = write_junit(
        tmp_path / "report.xml",
        [("tests.test_a", "test_known", "tests/test_a.py", False)],
    )
    decision = checker.evaluate(
        summary_path=summary,
        known_path=known,
        junit_path=junit,
        pytest_exit_code=0,
        root=tmp_path,
        pytest_rootdir=Path("workspace/backend"),
    )
    assert decision.unexpectedly_passing == [KNOWN_A]


def test_junit_paths_without_a_rootdir_are_taken_as_they_are(tmp_path, repo_test_file):
    """When the file attr already matches the manifest, nothing changes."""
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_all_passing(tmp_path / "summary.txt")
    junit = write_junit(
        tmp_path / "report.xml",
        [("tests.test_a", "test_known", "workspace/backend/tests/test_a.py", False)],
    )
    decision = checker.evaluate(
        summary_path=summary,
        known_path=known,
        junit_path=junit,
        pytest_exit_code=0,
        root=tmp_path,
        pytest_rootdir=Path("workspace/backend"),
    )
    # `workspace/backend` is not prepended a second time.
    assert decision.unexpectedly_passing == [KNOWN_A]


# ── the command line ──────────────────────────────────────────────────────────

def test_main_returns_the_decision_exit_code(tmp_path, monkeypatch, capsys):
    import sys

    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_B)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_known_failures.py",
            "--summary", str(summary),
            "--known", str(known),
            "--pytest-exit-code", "1",
            "--root", str(tmp_path),
        ],
    )
    assert checker.main() == 1


def test_main_returns_zero_for_known_failures_only(tmp_path, monkeypatch, capsys):
    import sys

    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_known_failures.py",
            "--summary", str(summary),
            "--known", str(known),
            "--pytest-exit-code", "1",
            "--root", str(tmp_path),
        ],
    )
    assert checker.main() == 0
    assert "Known failures" in capsys.readouterr().out


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5])
def test_unusable_exit_statuses_fail(tmp_path, exit_code):
    known = write_known(tmp_path / "known.txt", KNOWN_A)
    summary = summary_with_failures(tmp_path / "summary.txt", KNOWN_A)
    decision = judge(tmp_path, summary, known, exit_code)
    assert decision.status == "unusable"
    assert decision.exit_code == 2
