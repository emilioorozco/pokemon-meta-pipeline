"""Logging and run metrics: the format, the run identifier, and the row every stage writes.

These run in the fast suite. Nothing here starts a Java Virtual Machine, opens a
warehouse or reaches the network: the whole point of the module under test is
that it is the one part of the pipeline every other part imports, so it has to
be cheap to check.

The handler is installed on the root logger by `configure_logging`, so every
test that asserts on output captures stderr rather than using `caplog`: the
question is what the formatter emitted, not what the record held.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from pipeline import observability
from pipeline.observability import (
    RUN_METRICS_SCHEMA,
    STATUS_FAILED,
    STATUS_OK,
    configure_logging,
    current_run_id,
    emit_summary,
    new_run_id,
    run_metrics_dir,
    stage_run,
)


@pytest.fixture(autouse=True)
def clean_logging() -> Any:
    """Reset the run context and drop the handler this module installs, after every test."""
    observability._run_id.set("")
    observability._stage.set("")
    yield
    root = logging.getLogger()
    for handler in [found for found in root.handlers if found.get_name() == "pra"]:
        root.removeHandler(handler)
    observability._run_id.set("")
    observability._stage.set("")


def lines(captured: str) -> list[str]:
    return [line for line in captured.strip().splitlines() if line]


def records(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """Every JSON object the handler wrote to stderr, parsed."""
    return [json.loads(line) for line in lines(capsys.readouterr().err)]


# ---- the JSON format ----


def test_a_json_line_carries_the_context_and_the_extra_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = configure_logging("silver", run_id="run-abc", json_output=True)
    logging.getLogger("pipeline.silver").info("landed", extra={"rows": 12, "table": "games"})

    assert run_id == "run-abc"
    (record,) = records(capsys)
    assert record["msg"] == "landed"
    assert record["level"] == "INFO"
    assert record["logger"] == "pipeline.silver"
    assert record["stage"] == "silver"
    assert record["run_id"] == "run-abc"
    assert record["rows"] == 12
    assert record["table"] == "games"
    # An ISO timestamp in UTC, not a float and not a local time.
    assert record["ts"].endswith("+00:00")


def test_every_line_is_one_object_on_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("gold", run_id="run-abc", json_output=True)
    log = logging.getLogger("pipeline.gold")
    log.info("first")
    log.info("second", extra={"note": "a message\nwith a newline in it"})

    written = lines(capsys.readouterr().err)
    assert len(written) == 2
    assert [json.loads(line)["msg"] for line in written] == ["first", "second"]


def test_an_exception_becomes_three_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("gold", run_id="run-abc", json_output=True)
    try:
        raise ValueError("no warehouse")
    except ValueError:
        logging.getLogger("pipeline.gold").exception("build failed")

    (record,) = records(capsys)
    assert record["exc_type"] == "ValueError"
    assert record["exc_message"] == "no warehouse"
    assert "ValueError: no warehouse" in record["stack"]


def test_a_field_that_would_overwrite_a_record_attribute_is_renamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`extra={"message": ...}` raises inside the logging module; `safe_extra` is the guard."""
    configure_logging("silver", run_id="run-abc", json_output=True)
    emit_summary(logging.getLogger("pipeline.silver"), "summary", {"message": "hello", "rows": 1})

    (record,) = records(capsys)
    assert record["msg"] == "summary"
    assert record["message_"] == "hello"
    assert record["rows"] == 1


# ---- the console renderer ----


def test_console_mode_renders_one_line_with_the_stage_and_run_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("bronze_backfill", run_id="run-xyz", json_output=False)
    logging.getLogger("pipeline.backfill").info("landed", extra={"rows": 3})

    (line,) = lines(capsys.readouterr().err)
    assert "[bronze_backfill run-xyz]" in line
    assert line.endswith("landed rows=3")
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_the_format_variable_overrides_the_terminal_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRA_LOG_FORMAT", "console")
    assert observability.json_logging() is False
    monkeypatch.setenv("PRA_LOG_FORMAT", "json")
    assert observability.json_logging() is True
    # Anything else falls back to the terminal check, and the captured stderr a
    # test runs under is never a terminal.
    monkeypatch.setenv("PRA_LOG_FORMAT", "")
    assert observability.json_logging() is True
    # An explicit argument still wins over the variable.
    monkeypatch.setenv("PRA_LOG_FORMAT", "json")
    assert observability.json_logging(False) is False


def test_a_summary_goes_to_the_log_and_its_readable_form_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two streams on purpose: stdout is the command's result, stderr is its log."""
    configure_logging("bronze_backfill", run_id="run-xyz", json_output=True)
    emit_summary(
        logging.getLogger("pipeline.backfill"),
        "backfill summary",
        {"read": 3, "landed": 3},
        text="read: 3\nlanded: 3",
    )

    captured = capsys.readouterr()
    assert captured.out == "read: 3\nlanded: 3\n"
    record = json.loads(captured.err.strip())
    assert (record["msg"], record["read"], record["landed"]) == ("backfill summary", 3, 3)


# ---- the run identifier ----


def test_a_run_without_an_identifier_gets_a_fresh_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRA_RUN_ID", raising=False)
    first = configure_logging("silver")
    observability._run_id.set("")
    second = configure_logging("silver")

    assert first != second
    assert len(first) == len(second) == 16
    assert first.isalnum()


def test_the_run_id_variable_propagates_to_every_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRA_RUN_ID", "dag-run-7")

    assert configure_logging("bronze_backfill") == "dag-run-7"
    assert configure_logging("silver") == "dag-run-7"
    assert current_run_id() == "dag-run-7"


def test_two_stages_in_one_run_share_the_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRA_RUN_ID", "dag-run-8")
    configure_logging("bronze_backfill")
    with stage_run("bronze_backfill", directory=tmp_path):
        pass
    with stage_run("silver", directory=tmp_path):
        pass

    rows = sorted(read_rows(tmp_path), key=lambda row: row["stage"])
    assert [row["run_id"] for row in rows] == ["dag-run-8", "dag-run-8"]
    assert [row["stage"] for row in rows] == ["bronze_backfill", "silver"]
    assert sorted(path.name for path in tmp_path.glob("*.parquet")) == [
        "dag-run-8-bronze_backfill.parquet",
        "dag-run-8-silver.parquet",
    ]


def test_new_run_ids_do_not_repeat() -> None:
    assert len({new_run_id() for _ in range(100)}) == 100


# ---- the run-metrics row ----


def read_rows(directory: Path) -> list[dict[str, Any]]:
    """Every row in every Parquet file under the directory."""
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(path)
        assert table.schema.equals(RUN_METRICS_SCHEMA), path.name
        rows += table.to_pylist()
    return rows


def test_a_completed_block_writes_one_ok_row(tmp_path: Path) -> None:
    configure_logging("silver", run_id="run-1", json_output=True)
    with stage_run("silver", directory=tmp_path) as metrics:
        metrics.rows_in = 10
        metrics.rows_out = 10
        metrics.rows_quarantined = 0
        metrics.extra = {"tables": {"games": 10}}

    (row,) = read_rows(tmp_path)
    assert row["run_id"] == "run-1"
    assert row["stage"] == "silver"
    assert row["status"] == STATUS_OK
    assert row["error"] is None
    assert (row["rows_in"], row["rows_out"], row["rows_quarantined"]) == (10, 10, 0)
    assert json.loads(row["extra_json"]) == {"tables": {"games": 10}}
    assert row["hostname"]
    assert 0.0 <= row["duration_s"] < 10.0
    assert row["finished_at"] >= row["started_at"]


def test_the_duration_covers_the_block(tmp_path: Path) -> None:
    configure_logging("silver", run_id="run-2", json_output=True)
    marker = 0.05
    with stage_run("silver", directory=tmp_path):
        end = observability.time.monotonic() + marker
        while observability.time.monotonic() < end:
            pass

    (row,) = read_rows(tmp_path)
    assert row["duration_s"] >= marker


def test_counts_left_unset_are_null_rather_than_zero(tmp_path: Path) -> None:
    """A stage that never counted its input is not a stage that read nothing."""
    configure_logging("promote", run_id="run-3", json_output=True)
    with stage_run("promote", directory=tmp_path):
        pass

    (row,) = read_rows(tmp_path)
    assert (row["rows_in"], row["rows_out"], row["rows_quarantined"]) == (None, None, None)


def test_a_failing_block_records_the_error_and_re_raises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("gold", run_id="run-4", json_output=True)

    with (
        pytest.raises(RuntimeError, match="dbt exploded"),
        stage_run("gold", directory=tmp_path) as metrics,
    ):
        metrics.rows_in = 17
        raise RuntimeError("dbt exploded")

    (row,) = read_rows(tmp_path)
    assert row["status"] == STATUS_FAILED
    assert row["error"] == "RuntimeError: dbt exploded"
    # What it managed to count before it fell over is kept.
    assert row["rows_in"] == 17
    complete = [record for record in records(capsys) if record["msg"] == "stage complete"]
    assert [record["level"] for record in complete] == ["ERROR"]
    assert complete[0]["status"] == STATUS_FAILED


def test_stage_complete_carries_the_same_numbers_as_the_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("bronze_backfill", run_id="run-5", json_output=True)
    with stage_run("bronze_backfill", directory=tmp_path) as metrics:
        metrics.rows_in = 12
        metrics.rows_out = 10
        metrics.rows_quarantined = 2

    (row,) = read_rows(tmp_path)
    (complete,) = [record for record in records(capsys) if record["msg"] == "stage complete"]
    assert complete["stage"] == "bronze_backfill"
    assert complete["run_id"] == "run-5"
    assert (complete["rows_in"], complete["rows_out"], complete["rows_quarantined"]) == (
        row["rows_in"],
        row["rows_out"],
        row["rows_quarantined"],
    )
    assert complete["status"] == STATUS_OK


def test_a_directory_that_cannot_be_written_does_not_fail_the_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bookkeeping must never be the reason a stage that did its work reports failure."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n")
    configure_logging("silver", run_id="run-6", json_output=True)

    with stage_run("silver", directory=blocked) as metrics:
        metrics.rows_out = 1

    warned = [record for record in records(capsys) if record["level"] == "WARNING"]
    assert [record["msg"] for record in warned] == ["run metrics were not written"]


def test_the_default_directory_follows_the_data_dir_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved per call, so an orchestrator or a test can move it after import."""
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path))
    assert run_metrics_dir() == tmp_path / "lake" / "run_metrics"

    configure_logging("drift", run_id="run-7", json_output=True)
    with stage_run("drift") as metrics:
        metrics.rows_in = 1

    (row,) = read_rows(tmp_path / "lake" / "run_metrics")
    assert row["stage"] == "drift"


def test_a_machine_without_git_still_writes_its_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container with no git is the normal case, and it must not cost the row.

    `git_commit` shells out, and a missing or unrunnable executable raises
    `OSError` rather than returning a code. Unhandled, that raised while the row
    was being built, so the whole `run_metrics` file went missing and only a
    warning said so, which is exactly the case the table exists to cover. Found
    in the Airflow image, which ships no git.
    """
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert observability.git_commit() is None

    configure_logging("silver", run_id="run-8", json_output=True)
    with stage_run("silver", directory=tmp_path / "metrics") as metrics:
        metrics.rows_out = 3

    (row,) = read_rows(tmp_path / "metrics")
    assert row["git_commit"] is None
    assert (row["stage"], row["rows_out"], row["status"]) == ("silver", 3, STATUS_OK)
