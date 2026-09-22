"""The gate: what it refuses, what it lets through, and what it refuses to judge.

`mart_pipeline_health` is a dbt view over the run-metrics Parquet and
`tests/test_ops.py` already builds it for real, with a Java Virtual Machine and
a dbt run behind it. What is left is the rule, which is small enough to state
against a warehouse written by hand here: any stage whose last run failed, or
whose quarantine rate is over the threshold, fails the gate. Writing the table
directly is also the only way to test a stage that failed, since making a real
stage fail on purpose would be a different test about a different thing.
"""

from pathlib import Path

import duckdb
import pytest

from pipeline import quality_gate


def warehouse_with(tmp_path: Path, rows: list[tuple[object, ...]]) -> Path:
    """A DuckDB file holding a `mart_pipeline_health` of exactly these rows."""
    path = tmp_path / "meta.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "create table mart_pipeline_health ("
            "stage varchar, last_run_id varchar, last_status varchar, last_error varchar, "
            "quarantine_rate double, quarantine_rate_over_threshold boolean)"
        )
        if rows:
            connection.executemany(
                "insert into mart_pipeline_health values (?, ?, ?, ?, ?, ?)", rows
            )
    finally:
        connection.close()
    return path


def healthy(stage: str) -> tuple[object, ...]:
    return (stage, "run-1", "ok", None, 0.0, False)


def test_a_healthy_pipeline_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = warehouse_with(tmp_path, [healthy("bronze_backfill"), healthy("silver")])

    assert quality_gate.main(["--warehouse", str(path)]) == 0

    printed = capsys.readouterr().out
    assert "gate passed: 2 stage(s) healthy." in printed


def test_a_failed_last_run_fails_the_gate_and_names_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = warehouse_with(
        tmp_path,
        [
            healthy("bronze_backfill"),
            ("silver", "run-1", "failed", "ReconciliationError: games_in != games_out", 0.0, False),
        ],
    )

    assert quality_gate.main(["--warehouse", str(path)]) == quality_gate.EXIT_FAILED

    printed = capsys.readouterr().out
    assert "gate failed on 1 stage(s):" in printed
    assert "ReconciliationError" in printed


def test_a_quarantine_rate_over_the_threshold_fails_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = warehouse_with(
        tmp_path, [("bronze_backfill", "run-1", "ok", None, 0.31, True), healthy("silver")]
    )

    assert quality_gate.main(["--warehouse", str(path)]) == quality_gate.EXIT_FAILED

    assert "quarantine rate 31.0% is over the threshold" in capsys.readouterr().out


def test_the_gate_does_not_judge_its_own_last_run(tmp_path: Path) -> None:
    """Otherwise one bad night is permanent: the gate would keep reading its own refusal."""
    path = warehouse_with(
        tmp_path,
        [healthy("silver"), ("quality_gate", "run-0", "failed", "1 stage(s) failed", 0.0, False)],
    )

    assert quality_gate.main(["--warehouse", str(path)]) == 0


def test_an_empty_mart_is_not_a_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = warehouse_with(tmp_path, [])

    assert quality_gate.main(["--warehouse", str(path)]) == 0

    assert "no stage has recorded a run" in capsys.readouterr().out


def test_a_missing_warehouse_is_a_named_failure_not_a_verdict(tmp_path: Path) -> None:
    """Exit 2, not 1: "the gate could not run" and "the pipeline is unhealthy" differ."""
    with pytest.raises(SystemExit) as raised:
        quality_gate.main(["--warehouse", str(tmp_path / "missing.duckdb")])
    assert raised.value.code == 2


def test_a_warehouse_without_the_mart_is_a_named_failure(tmp_path: Path) -> None:
    path = tmp_path / "bare.duckdb"
    duckdb.connect(str(path)).close()
    with pytest.raises(SystemExit) as raised:
        quality_gate.main(["--warehouse", str(path)])
    assert raised.value.code == 2
