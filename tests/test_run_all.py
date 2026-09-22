"""The plain runner: the order, the skipping, the shared run identifier, the stop.

Two halves. The fast half replaces the stage commands with `true`, a script that
exits 7, and so on, so the runner's own decisions can be checked in
milliseconds: what it skips and why, where `--stop-after` stops it, and that a
failing stage ends the run with that stage's exit code rather than with 1.

The slow half is marked `spark` and runs the real thing, bronze through gold,
over the committed fixtures. It is the only test that proves the point of the
module: three separate interpreters, one run identifier, and three
`run_metrics` rows that a query can join. It needs a Java Virtual Machine for
silver and the dbt adapter for gold, both of which the marked suites already
install.
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from pipeline import run_all
from pipeline.observability import STATUS_FAILED, STATUS_OK
from pipeline.run_all import STATUS_SKIPPED, Stage
from tests.conftest import FIXTURES_DIR, TEST_HMAC_KEY

# Stage commands that need no pipeline at all. `exit 7` rather than `false`,
# because a runner that returned 1 for every failure would pass a test written
# against `false` while losing the number the caller needs.
OK_COMMAND = [sys.executable, "-c", ""]
FAILING_COMMAND = [sys.executable, "-c", "raise SystemExit(7)"]


def fake_commands(monkeypatch: pytest.MonkeyPatch, failing: str | None = None) -> list[str]:
    """Replace every stage command with a trivial one and return the list it fills in."""
    ran: list[str] = []

    def command(stage: Stage, *, source_dir: Path | None) -> list[str]:
        ran.append(stage.name)
        return FAILING_COMMAND if stage.name == failing else OK_COMMAND

    monkeypatch.setattr(run_all, "stage_command", command)
    return ran


def metric_rows(data_dir: Path) -> list[dict[str, Any]]:
    """Every `run_metrics` row under a data directory, whichever stage wrote it."""
    return [
        row
        for path in sorted((data_dir / "lake" / "run_metrics").glob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]


def run(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, argv: Sequence[str], run_id: str = "fake-run"
) -> int:
    """`main` with the environment it touches held inside the test."""
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PRA_RUN_ID", run_id)
    return run_all.main([*argv, "--data-dir", str(data_dir), "--run-id", run_id])


# ------------------------------------------------------------------ fast --


def test_the_stage_list_is_the_dependency_order() -> None:
    """The order is part of the contract, so it is asserted rather than assumed."""
    assert run_all.STAGE_NAMES == (
        "backfill",
        "silver",
        "gold",
        "train",
        "promote",
        "drift",
        "build_card_index",
        "quality_gate",
    )


def test_every_stage_runs_in_order_when_nothing_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_commands(monkeypatch)

    assert run(monkeypatch, tmp_path, []) == 0

    # The three model stages are skipped: there is no warehouse here, so
    # `features_turn` holds no rows. The card index is skipped because the
    # module does not exist yet.
    assert ran == ["backfill", "silver", "gold", "quality_gate"]


def test_skip_leaves_the_named_stages_out_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_commands(monkeypatch)

    assert run(monkeypatch, tmp_path, ["--skip", "silver,gold"]) == 0

    assert ran == ["backfill", "quality_gate"]
    (row,) = [row for row in metric_rows(tmp_path) if row["stage"] == "run_all"]
    skipped = {
        stage["stage"]: stage["reason"]
        for stage in _extra(row)["stages"]
        if stage["status"] == STATUS_SKIPPED
    }
    assert skipped["silver"] == "skipped by --skip"
    assert "not implemented yet" in skipped["build_card_index"]
    assert "features_turn" in skipped["train"]


def test_stop_after_runs_no_further(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran = fake_commands(monkeypatch)

    assert run(monkeypatch, tmp_path, ["--stop-after", "silver"]) == 0

    assert ran == ["backfill", "silver"]


def test_consumer_mode_turns_the_backfill_into_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRA_INGEST_MODE", "consumer")
    ran = fake_commands(monkeypatch)

    assert run(monkeypatch, tmp_path, ["--stop-after", "silver"]) == 0

    assert ran == ["silver"]


def test_a_failing_stage_stops_the_chain_with_its_own_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_commands(monkeypatch, failing="silver")

    assert run(monkeypatch, tmp_path, []) == 7

    assert ran == ["backfill", "silver"]
    (row,) = [row for row in metric_rows(tmp_path) if row["stage"] == "run_all"]
    assert row["status"] == STATUS_FAILED
    assert row["error"] == "silver exited 7"
    states = {stage["stage"]: stage["status"] for stage in _extra(row)["stages"]}
    assert states == {"backfill": STATUS_OK, "silver": STATUS_FAILED}


def test_an_unknown_stage_name_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_commands(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        run(monkeypatch, tmp_path, ["--skip", "nonsense"])
    assert raised.value.code == 2


def test_the_source_directory_only_reaches_the_stage_that_reads_blobs() -> None:
    stages = {stage.name: stage for stage in run_all.STAGES}
    fixtures = Path("tests/fixtures")
    assert run_all.stage_command(stages["backfill"], source_dir=fixtures)[-2:] == [
        "--source-dir",
        str(fixtures),
    ]
    assert "--source-dir" not in run_all.stage_command(stages["silver"], source_dir=fixtures)
    assert run_all.stage_command(stages["promote"], source_dir=None)[-2:] == [
        "--candidate",
        "latest",
    ]


def test_an_absent_warehouse_counts_zero_feature_rows(tmp_path: Path) -> None:
    assert run_all.feature_rows(tmp_path / "nothing.duckdb") == 0


def _extra(row: dict[str, Any]) -> dict[str, Any]:
    import json

    parsed: dict[str, Any] = json.loads(row["extra_json"])
    return parsed


# ------------------------------------------------------------------ slow --


@pytest.mark.spark
def test_bronze_through_gold_runs_end_to_end_under_one_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real commands, in real subprocesses, over the committed games.

    `--stop-after gold` because everything after it is the model loop, which has
    its own marked suite; what this proves is that three separate interpreters
    write three rows of one run.
    """
    monkeypatch.setenv("HANDLE_HMAC_KEY", TEST_HMAC_KEY.decode("utf-8"))

    code = run(
        monkeypatch,
        tmp_path,
        [
            "--source-dir",
            str(FIXTURES_DIR),
            "--skip",
            "train,promote,drift,build_card_index",
            "--stop-after",
            "gold",
        ],
        run_id="run-all-e2e",
    )

    assert code == 0
    rows = metric_rows(tmp_path)
    assert {row["run_id"] for row in rows} == {"run-all-e2e"}
    assert {row["stage"] for row in rows} == {"bronze_backfill", "silver", "gold", "run_all"}
    assert {row["status"] for row in rows} == {"ok"}
    landed = next(row for row in rows if row["stage"] == "bronze_backfill")
    assert landed["rows_out"] == len(sorted(FIXTURES_DIR.glob("*.json")))
    assert (tmp_path / "warehouse" / "meta.duckdb").is_file()
