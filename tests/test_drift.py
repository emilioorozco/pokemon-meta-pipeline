"""The drift report, on two synthetic corpora: one that did not move and one that did.

Marked `ml` and skipped by the default run, like the other model-stage suites,
though this one trains nothing: it needs the same temporary DuckDB feature
table and the same temporary MLflow store, and it belongs beside the tests of
the commands it reports on.

The feature table is `tests/test_train.py`'s, imported rather than copied, with
a recent window bolted onto the end of it. The recent window is a copy of the
training rows, which is what makes the first test an assertion rather than a
hope: a distribution compared with a copy of itself has a PSI of exactly zero,
so "near zero" here means near zero and not "small on this seed". The second
corpus takes that same copy and moves two things a real set release would move,
a prize race three prizes further along and two archetypes replaced by one new
deck, so the flag has to fire and the shifted feature has to be at the top of
the table.
"""

import json
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import duckdb
import pandas as pd
import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from pipeline import drift
from tests.test_train import synthetic_features

pytestmark = pytest.mark.ml

EXPERIMENT: Final = "test-win-probability-drift"
# The recent window: three days after the corpus ends, so `--window-days 3`
# selects exactly the rows added below and nothing that came before them.
RECENT_DAYS: Final = 3
PRIZE_SHIFT: Final = 3
# The deck that did not exist last month and is half the field this month.
NEW_ARCHETYPE: Final = "name:epsilon"
REPLACED: Final = ("name:gamma", "name:delta")


def recent_window(shift: bool) -> pd.DataFrame:
    """A copy of the training rows, re-dated onto the last three days of the corpus.

    Identical to the reference by construction unless `shift` is set, in which
    case the prize race is three prizes further along and two of the four
    archetypes have been replaced by one new deck on both sides of the table.
    """
    base = synthetic_features()
    recent = base[base["split"] == "train"].copy()
    last_day = max(base["play_date"])
    days = [last_day + timedelta(days=1 + index % RECENT_DAYS) for index in range(len(recent))]
    recent["play_date"] = days
    recent["game_id"] = ["recent-" + str(name) for name in recent["game_id"]]
    recent["feature_key"] = [
        f"{game}-{seat}-{turn}"
        for game, seat, turn in zip(
            recent["game_id"], recent["seat"], recent["turn_number"], strict=True
        )
    ]
    # Recent games are holdout rows: the reference is the training split, and a
    # window that fed itself into its own reference would compare nothing.
    recent["split"] = "holdout"
    if shift:
        recent["prize_diff"] = recent["prize_diff"] + PRIZE_SHIFT
        for column in ("archetype_key", "opponent_archetype_key"):
            recent[column] = [
                NEW_ARCHETYPE if value in REPLACED else value for value in recent[column]
            ]
    return recent


def build_warehouse(path: Path, shift: bool) -> Path:
    """A DuckDB file holding `features_turn`: the synthetic corpus plus a recent window."""
    frame = pd.concat([synthetic_features(), recent_window(shift)], ignore_index=True)
    connection = duckdb.connect(str(path))
    try:
        connection.register("frame", frame)
        connection.execute("create table features_turn as select * from frame")
    finally:
        connection.close()
    return path


@pytest.fixture(scope="module")
def stable_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A corpus whose recent window is a copy of the training window."""
    return build_warehouse(tmp_path_factory.mktemp("stable") / "meta.duckdb", shift=False)


@pytest.fixture(scope="module")
def shifted_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same corpus with the prize race and the archetype mix moved."""
    return build_warehouse(tmp_path_factory.mktemp("shifted") / "meta.duckdb", shift=True)


def run(warehouse: Path, out_dir: Path, tracking_uri: str, *extra: str) -> int:
    """`python -m pipeline.drift` over one warehouse, into a temporary store."""
    return drift.main(
        [
            "--warehouse",
            str(warehouse),
            "--out-dir",
            str(out_dir),
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            EXPERIMENT,
            "--window-days",
            str(RECENT_DAYS),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def stable(stable_warehouse: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One run over the corpus that did not move, and where it wrote."""
    out_dir = tmp_path_factory.mktemp("stable-report")
    tracking_uri = f"file:{tmp_path_factory.mktemp('stable-mlruns')}"
    assert run(stable_warehouse, out_dir, tracking_uri) == 0
    yield out_dir


@pytest.fixture(scope="module")
def shifted(shifted_warehouse: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One run over the corpus that did move, and where it wrote."""
    out_dir = tmp_path_factory.mktemp("shifted-report")
    tracking_uri = f"file:{tmp_path_factory.mktemp('shifted-mlruns')}"
    assert run(shifted_warehouse, out_dir, tracking_uri) == 0
    yield out_dir


def summary_of(out_dir: Path) -> dict[str, Any]:
    """The machine-readable half of one run's output."""
    loaded: dict[str, Any] = json.loads((out_dir / drift.SUMMARY_NAME).read_text())
    return loaded


def psi_by_feature(summary: dict[str, Any]) -> dict[str, float]:
    """Every numeric feature's PSI, keyed by feature name."""
    return {feature["feature"]: feature["psi"] for feature in summary["features"]}


def test_a_copy_of_the_reference_window_does_not_drift(stable: Path) -> None:
    summary = summary_of(stable)
    assert summary["drifted"] is False
    assert summary["max_psi"] == pytest.approx(0.0, abs=1e-9)
    assert summary["archetype_mix"]["psi"] == pytest.approx(0.0, abs=1e-9)
    assert set(psi_by_feature(summary)) == set(drift.NUMERIC_FEATURES)
    for feature, psi in psi_by_feature(summary).items():
        assert psi == pytest.approx(0.0, abs=1e-9), feature
    # The windows were built not to overlap, so the near-zero numbers above are
    # a comparison and not a distribution matched against itself.
    assert summary["windows"]["overlap_rows"] == 0
    assert summary["label_rate"]["reference"] == pytest.approx(summary["label_rate"]["current"])


def test_a_shifted_window_fires_the_flag_with_the_shifted_features_on_top(shifted: Path) -> None:
    summary = summary_of(shifted)
    assert summary["drifted"] is True
    assert summary["max_psi_feature"] == "prize_diff"
    assert summary["features"][0]["feature"] == "prize_diff"
    assert summary["features"][0]["psi"] >= summary["threshold"]
    assert summary["features"][0]["flagged"] is True
    assert summary["features"][0]["current_mean"] == pytest.approx(
        summary["features"][0]["reference_mean"] + PRIZE_SHIFT
    )
    # The other counters were not touched, so they must not have moved either.
    untouched = {name: psi for name, psi in psi_by_feature(summary).items() if name != "prize_diff"}
    assert max(untouched.values()) == pytest.approx(0.0, abs=1e-9)

    mix = summary["archetype_mix"]
    assert mix["psi"] >= summary["threshold"]
    assert mix["flagged"] is True
    assert mix["p_value"] < 0.05
    shares = {share["archetype"]: share for share in mix["shares"]}
    assert shares[NEW_ARCHETYPE]["reference"] == 0.0
    assert shares[NEW_ARCHETYPE]["current"] > 0.4
    for gone in ("name:gamma", "name:delta"):
        assert shares[gone]["current"] == 0.0
        assert shares[gone]["change"] < 0.0


def test_the_markdown_carries_the_tables_and_the_verdict(shifted: Path) -> None:
    report = (shifted / drift.REPORT_NAME).read_text()
    assert "| Feature | PSI | Reference mean |" in report
    assert "| Archetype | Reference share | Current share | Change |" in report
    assert "`prize_diff`" in report
    assert summary_of(shifted)["verdict"] in report
    assert report.startswith("# Feature drift report")
    # The reading of a PSI and the caveat about this corpus are part of the
    # report, because a number without either is a number nobody can act on.
    assert "stable" in report and "significant shift" in report
    assert "prompt to look" in report


def test_the_verdict_is_printed_with_the_path_of_the_report(
    stable_warehouse: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "report"
    assert run(stable_warehouse, out_dir, f"file:{tmp_path / 'mlruns'}") == 0
    printed = capsys.readouterr().out
    assert json.loads((out_dir / drift.SUMMARY_NAME).read_text())["verdict"] in printed
    assert str(out_dir / drift.REPORT_NAME) in printed


def test_the_run_is_in_mlflow_with_the_artifacts_and_the_metrics(
    shifted_warehouse: Path, tmp_path: Path
) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    assert run(shifted_warehouse, tmp_path / "report", tracking_uri) == 0

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert experiment is not None
    runs: list[Run] = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    logged = runs[0]

    for name in (
        "max_psi",
        "archetype_mix_psi",
        "drifted",
        "label_rate_reference",
        "label_rate_current",
    ):
        assert name in logged.data.metrics, name
    assert logged.data.metrics["drifted"] == 1.0
    assert logged.data.metrics["psi_prize_diff"] == pytest.approx(logged.data.metrics["max_psi"])
    for name in ("reference", "window_days", "psi_threshold", "current_from", "current_to"):
        assert name in logged.data.params, name
    assert logged.data.params["reference"] == "train"

    paths = {artifact.path for artifact in client.list_artifacts(logged.info.run_id)}
    assert {drift.REPORT_NAME, drift.SUMMARY_NAME} <= paths


def test_a_window_with_too_few_rows_exits_three_and_says_why(
    stable_warehouse: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "report"
    code = run(
        stable_warehouse,
        out_dir,
        f"file:{tmp_path / 'mlruns'}",
        "--as-of",
        date(2026, 5, 1).isoformat(),
    )
    assert code == drift.EXIT_TOO_SMALL
    printed = capsys.readouterr().out
    assert str(drift.MIN_CURRENT_ROWS) in printed
    assert "0 feature rows" in printed
    # Nothing is written and nothing is logged when there was nothing to compare.
    assert not out_dir.exists()


def test_an_unknown_reference_version_is_a_named_failure(
    stable_warehouse: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as raised:
        run(
            stable_warehouse, tmp_path / "report", f"file:{tmp_path / 'mlruns'}", "--reference", "9"
        )
    assert raised.value.code == 2


def test_only_the_archetypes_small_in_both_windows_are_folded() -> None:
    """A deck that was absent and is now a tenth of the field keeps its own row."""
    reference = pd.Series({"a": 500.0, "b": 480.0, "c": 5.0, "d": 0.0})
    current = pd.Series({"a": 450.0, "b": 400.0, "c": 4.0, "d": 100.0})
    folded = drift.fold_small(reference, current)
    assert set(folded["archetype"]) == {"a", "b", "d", "other"}
    assert float(folded.loc[folded["archetype"] == "other", "reference"].iloc[0]) == 5.0
