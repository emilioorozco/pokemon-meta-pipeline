"""The gold wrapper's own bookkeeping, without running dbt.

`tests/test_gold.py` builds the project for real and is marked `dbt` for it.
What is left over is the part of `pipeline.gold` that is not dbt: reading the
counts back out of `run_results.json`, and the command line that turns them into
a summary and a `run_metrics` row. Both are pure enough to check in the fast
suite, and they are the part most likely to break silently, because a
miscounted model still exits 0.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pipeline import gold
from pipeline.gold import GoldSummary, read_run_results


def write_results(dbt_dir: Path, statuses: list[str]) -> None:
    """A `run_results.json` with one node per status, as dbt writes it."""
    target = dbt_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    (target / "run_results.json").write_text(
        json.dumps({"results": [{"status": status} for status in statuses]}), encoding="utf-8"
    )


def test_run_results_counts_attempts_and_passes(tmp_path: Path) -> None:
    write_results(tmp_path, ["success", "success", "error"])
    assert read_run_results(tmp_path) == (3, 2)

    write_results(tmp_path, ["pass", "fail", "pass", "skipped"])
    assert read_run_results(tmp_path) == (4, 2)


def test_a_missing_or_broken_artifact_counts_zero_rather_than_raising(tmp_path: Path) -> None:
    """A count nobody can read is worth a zero in a metrics row, never a failed build."""
    assert read_run_results(tmp_path) == (0, 0)

    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "run_results.json").write_text("not json", encoding="utf-8")
    assert read_run_results(tmp_path) == (0, 0)


def test_the_summary_reads_as_two_lines() -> None:
    summary = GoldSummary(steps={"run": (19, 19), "test": (113, 112)})
    assert str(summary) == "models: 19/19 built\ntests:  112/113 passed"
    # Before `dbt test` has run there is no line about tests to print.
    assert str(GoldSummary(steps={"run": (19, 18)})) == "models: 18/19 built"


def test_the_command_line_records_the_counts_and_prints_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_gold(
        *,
        data_dir: Path,
        target: str,
        summary: GoldSummary | None = None,
        steps: Sequence[str] = gold.STEPS,
        select: str | None = None,
    ) -> int:
        assert summary is not None
        assert list(steps) == list(gold.STEPS)
        summary.steps["run"] = (19, 19)
        summary.steps["test"] = (113, 113)
        return 0

    monkeypatch.setattr(gold, "run_gold", fake_run_gold)

    assert gold.main(["--data-dir", str(tmp_path)]) == 0

    assert "models: 19/19 built" in capsys.readouterr().out
    (row,) = read_metric_rows(tmp_path / "lake" / "run_metrics")
    assert (row["stage"], row["status"]) == ("gold", "ok")
    assert (row["rows_in"], row["rows_out"]) == (19, 19)
    assert json.loads(row["extra_json"])["tests_passed"] == 113


def test_a_failed_dbt_run_is_a_failed_row_and_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dbt exiting non-zero is not an exception, so the status has to be set by hand."""

    def fake_run_gold(
        *,
        data_dir: Path,
        target: str,
        summary: GoldSummary | None = None,
        steps: Sequence[str] = gold.STEPS,
        select: str | None = None,
    ) -> int:
        assert summary is not None
        summary.steps["run"] = (19, 18)
        return 1

    monkeypatch.setattr(gold, "run_gold", fake_run_gold)

    assert gold.main(["--data-dir", str(tmp_path)]) == 1

    (row,) = read_metric_rows(tmp_path / "lake" / "run_metrics")
    assert row["status"] == "failed"
    assert row["error"] == "dbt exited 1"
    assert (row["rows_in"], row["rows_out"]) == (19, 18)


def read_metric_rows(directory: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return [
        row
        for path in sorted(directory.glob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]


def test_a_partial_build_records_itself_under_its_own_stage_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the DAG's three dbt tasks depend on: one command, three separable rows.

    Without `--stage-name` all three would write `<run id>-gold.parquet` and the
    last one would be the only one left, so the graph would have three nodes and
    the metrics table one.
    """
    seen: dict[str, object] = {}

    def fake_run_gold(
        *,
        data_dir: Path,
        target: str,
        summary: GoldSummary | None = None,
        steps: Sequence[str] = gold.STEPS,
        select: str | None = None,
    ) -> int:
        assert summary is not None
        seen["steps"] = list(steps)
        seen["select"] = select
        summary.steps["run"] = (2, 2)
        return 0

    monkeypatch.setattr(gold, "run_gold", fake_run_gold)

    code = gold.main(
        [
            "--data-dir",
            str(tmp_path),
            "--steps",
            "run",
            "--select",
            "tag:ml",
            "--stage-name",
            "gold_features",
        ]
    )

    assert code == 0
    assert seen == {"steps": ["run"], "select": "tag:ml"}
    # No test line, because no test step ran.
    assert "tests:" not in capsys.readouterr().out
    (row,) = read_metric_rows(tmp_path / "lake" / "run_metrics")
    assert row["stage"] == "gold_features"
    assert json.loads(row["extra_json"])["select"] == "tag:ml"


def test_an_unknown_step_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        gold.main(["--data-dir", str(tmp_path), "--steps", "compile"])
    assert raised.value.code == 2
