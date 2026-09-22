"""The model stage end to end, on synthetic data, with no Java and no server.

Every test here is marked `ml` and the default `uv run pytest` skips them, the
same split the Spark and dbt suites get, but for a different reason: these need
no Java Virtual Machine at all. The feature table they train on is built
straight into a temporary DuckDB file out of pandas, with the same column names
and types `features_turn` has, so `pytest -m ml` runs the whole model stage on a
laptop with nothing installed but this package. The dbt suite is what proves
the real `features_turn` matches this shape.

The data has a planted signal: `prize_diff` decides the game, with noise on top
and everything else independent of the label. That makes the assertions real
assertions rather than smoke. A model that cannot beat an archetype win-rate
baseline when the answer is written in one column is broken, so
`beats_baseline` is checked as an equality and not as a "did it run".

The tracking URI is a `file:` URI under `tmp_path`, so nothing reaches
`data/mlruns` or a server and the runs are read back with a plain MlflowClient
against the same directory.
"""

import json
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import duckdb
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.artifacts import download_artifacts
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from pipeline import train

pytestmark = pytest.mark.ml

GAMES: Final = 120
TURNS_PER_GAME: Final = 6
FIRST_DAY: Final = date(2026, 6, 1)
# The last quarter of the days, which is the shape `ml_split_cutoff` produces.
HOLDOUT_START: Final = FIRST_DAY + timedelta(days=int(GAMES * 0.75) // 2)
ARCHETYPES: Final = ("name:alpha", "name:beta", "name:gamma", "name:delta")


def synthetic_features(seed: int = 7) -> pd.DataFrame:
    """A feature table shaped like `features_turn`, with the label planted in prize_diff.

    Two seats per game and one row per turn per seat, so the grain and the row
    counts match the real model as well as the columns do. The winner is decided
    once per game from a prize lead that grows over its turns, and every other
    column is drawn independently of it, so a feature that shows up as important
    other than `prize_diff` is the test telling on itself.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for game in range(GAMES):
        play_date = FIRST_DAY + timedelta(days=game // 2)
        lead = int(rng.integers(-3, 4))
        # The label follows the lead with noise, so no feature is a perfect key.
        winner_is_seat_0 = bool(lead > 0) if lead != 0 else bool(rng.random() < 0.5)
        if rng.random() < 0.1:
            winner_is_seat_0 = not winner_is_seat_0
        decks = rng.choice(ARCHETYPES, size=2)
        for seat in (0, 1):
            own_lead = lead if seat == 0 else -lead
            for turn in range(1, TURNS_PER_GAME + 1):
                taken_self = max(0, own_lead) * turn // TURNS_PER_GAME
                taken_opp = max(0, -own_lead) * turn // TURNS_PER_GAME
                rows.append(
                    {
                        "feature_key": f"g{game}-{seat}-{turn}",
                        "game_id": f"g{game}",
                        "seat": seat,
                        "turn_number": turn,
                        "play_date": play_date,
                        "turn_count": TURNS_PER_GAME,
                        "went_first": seat == game % 2,
                        "is_uploader": seat == 0,
                        "archetype_key": str(decks[seat]),
                        "opponent_archetype_key": str(decks[1 - seat]),
                        "prizes_taken_self": taken_self,
                        "prizes_taken_opp": taken_opp,
                        "prizes_remaining_self": 6 - taken_self,
                        "prizes_remaining_opp": 6 - taken_opp,
                        "prize_diff": taken_self - taken_opp,
                        "knockouts_self": int(rng.integers(0, 3)),
                        "knockouts_opp": int(rng.integers(0, 3)),
                        "cards_drawn_self": int(rng.integers(5, 30)),
                        "energy_attached_self": int(rng.integers(0, 6)),
                        "pokemon_played_self": int(rng.integers(0, 6)),
                        "trainers_played_self": int(rng.integers(0, 20)),
                        "evolutions_self": int(rng.integers(0, 4)),
                        "attacks_self": int(rng.integers(0, 6)),
                        "turns_played_self": turn - 1,
                        "won": winner_is_seat_0 if seat == 0 else not winner_is_seat_0,
                        "split": "train" if play_date < HOLDOUT_START else "holdout",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A DuckDB file holding nothing but a synthetic `features_turn`."""
    path = tmp_path_factory.mktemp("warehouse") / "meta.duckdb"
    frame = synthetic_features()
    connection = duckdb.connect(str(path))
    try:
        connection.register("frame", frame)
        connection.execute("create table features_turn as select * from frame")
    finally:
        connection.close()
    return path


@pytest.fixture(scope="module")
def trained(warehouse: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """One `python -m pipeline.train` against that warehouse, into a temporary tracking store.

    Module scoped because it is the expensive part and every test below reads
    the same two runs. The exit code is asserted here rather than in a test of
    its own, so a failed run reports once instead of six times.
    """
    tracking_uri = f"file:{tmp_path_factory.mktemp('mlruns')}"
    code = train.main(
        [
            "--warehouse",
            str(warehouse),
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            "test-win-probability",
            "--params",
            "num_boost_round=80",
            "min_data_in_leaf=10",
        ]
    )
    assert code == 0
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


def runs_by_kind(tracking_uri: str) -> dict[str, Run]:
    """The model run and the baseline run of the experiment, keyed by their tag."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test-win-probability")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    found = {run.data.tags.get("baseline", "false"): run for run in runs}
    assert set(found) == {"true", "false"}, "expected exactly one model run and one baseline run"
    return found


def test_the_model_run_logs_its_parameters_and_data_range(trained: str) -> None:
    run = runs_by_kind(trained)["false"]
    params = run.data.params
    for name in train.DEFAULT_PARAMS:
        assert name in params, name
    for name in (
        "train_rows",
        "holdout_rows",
        "train_from",
        "train_to",
        "holdout_from",
        "holdout_to",
        "n_games",
    ):
        assert name in params, name
    assert int(params["n_games"]) == GAMES
    assert int(params["train_rows"]) + int(params["holdout_rows"]) == GAMES * 2 * TURNS_PER_GAME


def test_the_time_split_is_respected(trained: str, warehouse: Path) -> None:
    """No training row is on or after the holdout start, in the run or in the table.

    Both halves of the claim are checked: what the trainer recorded about the
    data it used, and the data itself. The first would still pass if the module
    read the wrong column, and the second would still pass if it ignored the
    column entirely, so neither is enough alone.
    """
    params = runs_by_kind(trained)["false"].data.params
    assert params["train_to"] < params["holdout_from"]
    assert params["holdout_from"] == HOLDOUT_START.isoformat()

    connection = duckdb.connect(str(warehouse), read_only=True)
    try:
        leaked = connection.sql(
            "select count(*) from features_turn "
            f"where split = 'train' and play_date >= date '{HOLDOUT_START.isoformat()}'"
        ).fetchone()
    finally:
        connection.close()
    assert leaked is not None
    assert leaked[0] == 0


def test_the_model_run_logs_metrics_on_both_halves(trained: str) -> None:
    metrics = runs_by_kind(trained)["false"].data.metrics
    for name in ("logloss", "auc", "train_logloss", "train_auc", "beats_baseline"):
        assert name in metrics, name
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["logloss"] > 0.0


def test_the_model_run_logs_every_artifact(trained: str) -> None:
    run = runs_by_kind(trained)["false"]
    listed = MlflowClient(tracking_uri=trained).list_artifacts(run.info.run_id)
    paths = {artifact.path for artifact in listed}
    assert {
        "calibration.png",
        "feature_importance.png",
        "feature_importance.csv",
        "features.json",
        "archetype_codes.json",
    } <= paths
    # MLflow 3 keeps the model beside the run rather than inside it, so the run
    # has to say where it went.
    assert run.data.tags["model_uri"].startswith("models:/")

    local = download_artifacts(run_id=run.info.run_id, artifact_path="features.json")
    described: dict[str, Any] = json.loads(Path(local).read_text())
    assert described["features"] == list(train.MODEL_FEATURES)
    assert set(described["dtypes"]) == set(train.MODEL_FEATURES)
    # The bias column and the two duplicates are in the table and not in the model.
    assert "is_uploader" not in described["features"]


def test_the_logged_model_loads_and_predicts_probabilities(trained: str) -> None:
    run = runs_by_kind(trained)["false"]
    model = mlflow.pyfunc.load_model(run.data.tags["model_uri"])

    frame = synthetic_features()
    holdout = frame[frame["split"] == "holdout"]
    codes = train.build_codes(frame[frame["split"] == "train"])
    predictions = np.asarray(model.predict(train.design_matrix(holdout, codes)), dtype=float)

    assert len(predictions) == len(holdout)
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0
    # A constant prediction would satisfy the bounds and mean nothing.
    assert predictions.std() > 0.01


def test_the_baseline_run_exists_and_loses_to_the_planted_signal(trained: str) -> None:
    runs = runs_by_kind(trained)
    baseline, model = runs["true"], runs["false"]

    assert baseline.data.params["model"] == "archetype_pair_win_rate"
    assert "logloss" in baseline.data.metrics
    assert "auc" in baseline.data.metrics
    assert baseline.data.metrics["beats_baseline"] == 1.0
    assert model.data.metrics["beats_baseline"] == 1.0
    assert model.data.metrics["logloss"] < baseline.data.metrics["logloss"]
    # The archetypes are drawn independently of the label, so the baseline has
    # nothing to go on and should land near a coin flip.
    assert baseline.data.metrics["auc"] == pytest.approx(0.5, abs=0.2)


def test_a_warehouse_without_features_is_a_named_failure(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        train.main(["--warehouse", str(tmp_path / "missing.duckdb")])
    assert raised.value.code == 2


def test_parse_params_types_what_it_can() -> None:
    assert train.parse_params(["learning_rate=0.1", "num_leaves=8", "objective=binary"]) == {
        "learning_rate": 0.1,
        "num_leaves": 8,
        "objective": "binary",
    }
    with pytest.raises(ValueError, match="key=value"):
        train.parse_params(["nonsense"])
