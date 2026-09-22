"""Model stage: a per-turn win-probability model, every run recorded in MLflow.

One command trains one model and writes two MLflow runs: the LightGBM model,
and the baseline it has to beat. Both are in the same experiment with the same
metrics on the same holdout, because a metric with nothing beside it is not a
result. The baseline is the matchup table itself, the thing the marts already
publish: predict the historical win rate of this archetype pair. If a gradient
boosted model cannot beat a group-by, the honest report is that it cannot, and
this command prints that in those words and still exits 0.

What it reads: `features_turn` out of the DuckDB warehouse gold built, through
Arrow into pandas. What it writes: nothing to the warehouse, everything to the
tracking server (a local `data/mlruns` directory unless `MLFLOW_TRACKING_URI`
points somewhere, such as the `mlflow` service in compose.yaml).

Every run also registers its model as a new version of `win-probability`,
tagged with the holdout numbers and with whether it beat the baseline. That is
all it does: a version is a candidate, and nothing serves it. Moving the
`production` alias is `python -m pipeline.promote`, a separate command with a
rule it prints, so a scheduled retrain cannot quietly ship a worse model.

Three choices worth knowing before reading the code.

The split is by date and it is made in dbt, not here. `features_turn.split` is
already `train` or `holdout`, assigned from `ml_split_cutoff`, so the boundary
is one date in one place rather than a flag two systems could set differently.
This module reads the column and asserts what it implies: the last training day
is strictly before the first holdout day, logged as parameters so a run that
was trained on its own test set is visible in the run table.

The model's feature list is narrower than the table's columns, and one of the
omissions matters more than the others. `seat` is an index into an array rather
than a property of play, and `prizes_remaining_self` and `prizes_remaining_opp`
are exactly six minus columns already in the list, so they would only split one
importance score into two. `is_uploader` is the one worth arguing about: it is
a fact about who kept the log, not about the game, and on this corpus the
uploader wins the overwhelming majority of the games that reach the feature
table. Trained on it, the model scores near one and has learned nothing except
which seat pressed the upload button, which is a number no future prediction
can use, because a prediction is asked before the game is uploaded. So the
table keeps the column and the model is not given it. `excluded_features` is
logged as a parameter and `features.json` records what the model did get.

Archetypes are integer codes, not strings, and the mapping is built from the
training split alone. An archetype that appears for the first time in the
holdout therefore arrives as -1, which LightGBM reads as a missing category.
That is the truth about it: the model has never seen that deck, and a run where
the codes had been fitted on all the data would report a score no future week
could reproduce.
"""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score

from pipeline.config import REGISTERED_MODEL_NAME, WAREHOUSE_PATH, default_tracking_uri
from pipeline.ml_features import (
    CATEGORICAL,
    LABEL,
    MODEL_FEATURES,
    ArchetypeCodes,
    design_matrix,
)
from pipeline.observability import (
    RunMetrics,
    configure_logging,
    emit_summary,
    git_commit,
    stage_run,
)

logger = logging.getLogger(__name__)
STAGE: Final = "train"

FEATURE_TABLE: Final = "features_turn"
DEFAULT_EXPERIMENT: Final = "win-probability"

# Read but not modelled: identity, the date the split is made on, and the split.
CARRIED: Final[tuple[str, ...]] = ("game_id", "seat", "play_date", "split")
# Columns of `features_turn` deliberately withheld from the model, logged as a
# parameter so the choice is part of the run rather than part of the source.
EXCLUDED_FEATURES: Final[tuple[str, ...]] = (
    "seat",
    "is_uploader",
    "prizes_remaining_self",
    "prizes_remaining_opp",
)

# Small data, so the defaults are the conservative end of every knob: shallow
# trees, a low learning rate with enough rounds to still fit, and a leaf that
# has to be backed by real rows. Every one of them is logged, and `--params`
# overrides any of them from the command line.
DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_boost_round": 250,
    "seed": 17,
    "verbosity": -1,
}

# Log loss is undefined at 0 and 1, so predictions are nudged off the ends
# before it is computed. This is arithmetic, not smoothing.
EPSILON: Final = 1e-6
# The baseline gets a wider clamp, and it is a modelling choice rather than
# arithmetic: an empirical win rate over two games is allowed to say "likely",
# not "certain", and without the clamp a single thin matchup cell decides the
# comparison. It is logged as a parameter of the baseline run.
BASELINE_CLIP: Final = 0.05


class TrainingDataError(RuntimeError):
    """The warehouse has no usable training data; the message says which half is missing."""


@dataclass(frozen=True)
class Metrics:
    """Log loss and area under the curve for one set of predictions."""

    logloss: float
    auc: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """The pair as MLflow metric names, optionally prefixed with `train_`."""
        return {f"{prefix}logloss": self.logloss, f"{prefix}auc": self.auc}


@dataclass(frozen=True)
class Baseline:
    """Historical win rates from the training split, with two fallbacks.

    The matchup rate is what the `mart_matchups` table already publishes, so
    this is the model the pipeline can serve today with no model at all. A pair
    the training split never saw falls back to how that archetype did overall,
    and an archetype it never saw falls back to the corpus win rate, which is
    almost exactly one half because every game contributes both seats.
    """

    pair: dict[tuple[str, str], float]
    own: dict[str, float]
    overall: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """One probability per row, clamped away from certainty."""
        rates = [
            self.pair.get((mine, theirs), self.own.get(mine, self.overall))
            for mine, theirs in zip(
                frame["archetype_key"], frame["opponent_archetype_key"], strict=True
            )
        ]
        clamped = np.clip(np.asarray(rates, dtype=float), BASELINE_CLIP, 1 - BASELINE_CLIP)
        return np.asarray(clamped, dtype=float)


def parse_params(pairs: Sequence[str]) -> dict[str, Any]:
    """`key=value` strings into typed parameters, numbers where they parse as numbers."""
    parsed: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator:
            raise ValueError(f"expected key=value, got {pair!r}")
        value: Any = raw
        for cast in (int, float):
            try:
                value = cast(raw)
            except ValueError:
                continue
            break
        parsed[key.strip()] = value
    return parsed


def load_features(warehouse: Path) -> pd.DataFrame:
    """Read `features_turn` out of the warehouse, DuckDB to Arrow to pandas.

    Read only, and by an explicit column list rather than a star, so a column
    renamed in dbt fails here by name instead of arriving as a missing feature.

    Ordered, and that is not cosmetic: LightGBM's row bagging samples by
    position, so the same data in a different order trains a different model.
    A rebuilt DuckDB table does not promise an order, so the query asks for one
    and two runs over the same warehouse then produce the same numbers.
    """
    if not warehouse.is_file():
        raise TrainingDataError(f"no warehouse at {warehouse}; run `python -m pipeline.gold` first")
    columns = ", ".join((*CARRIED, *MODEL_FEATURES, LABEL))
    connection = duckdb.connect(str(warehouse), read_only=True)
    try:
        table = connection.sql(
            f"select {columns} from {FEATURE_TABLE} order by game_id, seat, turn_number"
        ).to_arrow_table()
    finally:
        connection.close()
    return table.to_pandas()


def split_by_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training and holdout halves, with the date boundary between them checked."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    holdout = frame[frame["split"] == "holdout"].reset_index(drop=True)
    if train.empty or holdout.empty:
        raise TrainingDataError(
            f"{FEATURE_TABLE} has {len(train)} training rows and {len(holdout)} holdout rows; "
            "both halves are needed, so check `ml_split_cutoff` and the corpus date range"
        )
    if train["play_date"].max() >= holdout["play_date"].min():
        raise TrainingDataError(
            "the split is not a date split: the last training day is not before the first "
            "holdout day, so the model would be scored on games it had been shown"
        )
    return train, holdout


def build_codes(train: pd.DataFrame) -> ArchetypeCodes:
    """An integer code per archetype, fitted on the training split alone."""
    return {
        column: {value: index for index, value in enumerate(sorted(set(train[column])))}
        for column in CATEGORICAL
    }


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> Metrics:
    """Log loss and area under the curve, with the one-class case named rather than crashing."""
    clipped = np.clip(predictions, EPSILON, 1 - EPSILON)
    logloss = float(log_loss(labels, clipped, labels=[0, 1]))
    # A split that holds one class has no ranking to score. That is a fact
    # about the corpus, not an error, so it is reported as a missing number.
    auc = float(roc_auc_score(labels, predictions)) if len(set(labels)) > 1 else float("nan")
    return Metrics(logloss=logloss, auc=auc)


def fit_baseline(train: pd.DataFrame) -> Baseline:
    """Win rates per matchup, per archetype and overall, over training game sides.

    Over game sides and not over feature rows: a row is a turn, so counting
    rows would weight a thirty turn game three times as heavily as a ten turn
    one when deciding how a matchup usually goes.
    """
    sides = train.drop_duplicates(subset=["game_id", "seat"]).copy()
    sides["rate"] = sides[LABEL].astype(float)
    pairs = sides.groupby(["archetype_key", "opponent_archetype_key"], as_index=False)[
        "rate"
    ].mean()
    singles = sides.groupby("archetype_key", as_index=False)["rate"].mean()
    return Baseline(
        pair={
            (str(row.archetype_key), str(row.opponent_archetype_key)): float(row.rate)
            for row in pairs.itertuples(index=False)
        },
        own={str(row.archetype_key): float(row.rate) for row in singles.itertuples(index=False)},
        overall=float(sides["rate"].mean()),
    )


def calibration_figure(labels: np.ndarray, predictions: np.ndarray, title: str) -> Figure:
    """Predicted probability against observed frequency, with the diagonal to read it against.

    Quantile bins rather than uniform ones, because a small corpus leaves most
    uniform bins empty and the plot then says more about the binning than about
    the model. The figure is built through the object interface rather than
    pyplot, so no display backend is involved and nothing is left in a global
    figure registry between runs.
    """
    bins = min(10, max(2, len(set(np.round(predictions, 3)))))
    observed, predicted = calibration_curve(labels, predictions, n_bins=bins, strategy="quantile")
    figure = Figure(figsize=(5, 5), dpi=120)
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888", label="perfect")
    axes.plot(predicted, observed, marker="o", linewidth=1.5, label="model")
    axes.set_xlabel("predicted win probability")
    axes.set_ylabel("observed win rate")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_title(title)
    axes.legend(loc="upper left")
    figure.tight_layout()
    return figure


def importance_frame(booster: lgb.Booster) -> pd.DataFrame:
    """Gain and split importance per feature, heaviest first.

    Gain rather than split count, because a split count rewards a feature with
    many distinct values for being easy to split on rather than for being
    informative. Both are kept in the comma separated file.
    """
    frame = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "splits": booster.feature_importance(importance_type="split"),
        }
    )
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def importance_figure(frame: pd.DataFrame) -> Figure:
    """The importance table as a horizontal bar chart, heaviest at the top."""
    top = frame.head(15).iloc[::-1]
    figure = Figure(figsize=(7, 5), dpi=120)
    axes = figure.subplots()
    axes.barh(top["feature"], top["gain"], color="#1e8449")
    axes.set_xlabel("total gain")
    axes.set_title("feature importance")
    figure.tight_layout()
    return figure


def date_range(frame: pd.DataFrame) -> tuple[str, str]:
    """The first and last play date in a frame, as ISO strings for a parameter."""
    dates = pd.to_datetime(frame["play_date"])
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def dataset_params(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    """What data a run saw, as parameters, so two runs can be compared honestly."""
    train_from, train_to = date_range(train)
    holdout_from, holdout_to = date_range(holdout)
    both = pd.concat([train["game_id"], holdout["game_id"]])
    return {
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_games": train["game_id"].nunique(),
        "holdout_games": holdout["game_id"].nunique(),
        "n_games": both.nunique(),
        "train_from": train_from,
        "train_to": train_to,
        "holdout_from": holdout_from,
        "holdout_to": holdout_to,
    }


def log_model_run(
    *,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    params: dict[str, Any],
    data_params: dict[str, Any],
    commit: str | None,
) -> tuple[Metrics, pd.DataFrame, str]:
    """Train LightGBM inside the active MLflow run and log everything that describes it."""
    codes = build_codes(train)
    x_train = design_matrix(train, codes)
    x_holdout = design_matrix(holdout, codes)
    y_train = train[LABEL].to_numpy(dtype=int)
    y_holdout = holdout[LABEL].to_numpy(dtype=int)

    booster_params = dict(params)
    rounds = int(booster_params.pop("num_boost_round"))
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=list(CATEGORICAL),
        free_raw_data=False,
    )
    booster = lgb.train(booster_params, dataset, num_boost_round=rounds)

    train_scores = np.asarray(booster.predict(x_train), dtype=float)
    holdout_scores = np.asarray(booster.predict(x_holdout), dtype=float)
    train_metrics = evaluate(y_train, train_scores)
    holdout_metrics = evaluate(y_holdout, holdout_scores)
    importance = importance_frame(booster)

    mlflow.log_params({**params, **data_params, "excluded_features": ",".join(EXCLUDED_FEATURES)})
    mlflow.set_tags(
        {
            "stage": "model",
            "model_type": "lightgbm",
            "dataset": FEATURE_TABLE,
            "baseline": "false",
            **({"git_commit": commit} if commit else {}),
        }
    )
    mlflow.log_metrics({**holdout_metrics.as_dict(), **train_metrics.as_dict("train_")})
    mlflow.log_figure(
        calibration_figure(y_holdout, holdout_scores, "calibration, holdout"),
        "calibration.png",
    )
    mlflow.log_figure(importance_figure(importance), "feature_importance.png")
    mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")
    mlflow.log_dict(
        {
            "features": list(MODEL_FEATURES),
            "dtypes": {name: str(dtype) for name, dtype in x_train.dtypes.items()},
            "categorical": list(CATEGORICAL),
            "label": LABEL,
        },
        "features.json",
    )
    # The code map is what a serving layer needs to turn an archetype key back
    # into the integer the model was trained on, so it travels with the run.
    mlflow.log_dict(codes, "archetype_codes.json")
    # MLflow 3 stores a model as its own object rather than as a folder inside
    # the run, so the run carries the URI that finds it again. Serving reads
    # this tag; without it, finding the model means knowing MLflow's internal
    # identifier scheme, which is exactly the hand-kept note the tracker is
    # meant to replace.
    logged = mlflow.lightgbm.log_model(
        booster,
        name="model",
        signature=infer_signature(x_holdout, holdout_scores),
        input_example=x_holdout.head(5),
    )
    mlflow.set_tag("model_uri", logged.model_uri)
    return holdout_metrics, importance, str(logged.model_uri)


def register_version(
    *,
    model_uri: str,
    holdout: Metrics,
    data_params: dict[str, Any],
    beats: bool,
) -> str:
    """Register this run's model as a new version of `win-probability`, tagged with its score.

    Every training run produces a version; nothing here decides whether it is
    any good. That is `python -m pipeline.promote`, which reads exactly these
    tags, and the separation is the point: training is allowed to run on a
    schedule and produce a worse model, and the thing serving loads only moves
    when a second command says it may.

    The tags duplicate numbers that are already metrics on the source run. That
    is deliberate. A version is what the promotion step and the service read,
    and making either of them walk back to a run to find out how good the model
    is turns a comparison into a join. The run id stays on the version, so the
    full record is one hop away when the tags are not enough.

    Only the LightGBM run is registered. The baseline is a group-by kept as a
    yardstick; registering it would put something in the registry that no
    serving path can load.
    """
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = MlflowClient()
    tags = {
        "holdout_logloss": f"{holdout.logloss:.6f}",
        "holdout_auc": f"{holdout.auc:.6f}",
        "beats_baseline": "1" if beats else "0",
        # Both ends of the training window, not only the last day: the drift
        # report selects rows by them, and a window with one end is a filter
        # that quietly reaches back to the first game ever played.
        "train_from": str(data_params["train_from"]),
        "train_to": str(data_params["train_to"]),
        "holdout_to": str(data_params["holdout_to"]),
    }
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)
    return str(version.version)


def log_baseline_run(
    *,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    data_params: dict[str, Any],
    commit: str | None,
) -> Metrics:
    """Score the matchup win-rate baseline on the same holdout, in the active run."""
    baseline = fit_baseline(train)
    train_metrics = evaluate(train[LABEL].to_numpy(dtype=int), baseline.predict(train))
    holdout_metrics = evaluate(holdout[LABEL].to_numpy(dtype=int), baseline.predict(holdout))

    mlflow.log_params(
        {
            "model": "archetype_pair_win_rate",
            "fallback": "archetype_rate_then_global_rate",
            "baseline_clip": BASELINE_CLIP,
            "known_pairs": len(baseline.pair),
            "known_archetypes": len(baseline.own),
            "global_win_rate": round(baseline.overall, 6),
            **data_params,
        }
    )
    mlflow.set_tags(
        {
            "stage": "model",
            "model_type": "baseline",
            "dataset": FEATURE_TABLE,
            "baseline": "true",
            **({"git_commit": commit} if commit else {}),
        }
    )
    mlflow.log_metrics({**holdout_metrics.as_dict(), **train_metrics.as_dict("train_")})
    mlflow.log_text(
        json.dumps(
            {
                "pair_win_rate": {f"{a} vs {b}": v for (a, b), v in baseline.pair.items()},
                "archetype_win_rate": baseline.own,
                "global_win_rate": baseline.overall,
            },
            indent=2,
            sort_keys=True,
        ),
        "baseline_rates.json",
    )
    return holdout_metrics


def report(
    *,
    data_params: dict[str, Any],
    model: Metrics,
    baseline: Metrics,
    importance: pd.DataFrame,
    beats: bool,
    tracking_uri: str,
    experiment: str,
    version: str,
) -> str:
    """The numbers a reader needs to judge the run, including the bad news.

    Returned rather than printed: the command line hands it to `emit_summary`,
    which logs the run's fields as one record and writes this block to stdout.
    """
    lines = [
        f"experiment: {experiment} at {tracking_uri}",
        f"{FEATURE_TABLE}: {data_params['train_rows'] + data_params['holdout_rows']} rows, "
        f"{data_params['n_games']} games",
        f"train:   {data_params['train_rows']:>5} rows, {data_params['train_games']} games, "
        f"{data_params['train_from']} to {data_params['train_to']}",
        f"holdout: {data_params['holdout_rows']:>5} rows, {data_params['holdout_games']} games, "
        f"{data_params['holdout_from']} to {data_params['holdout_to']}",
        "",
        f"model     holdout logloss {model.logloss:.4f}  auc {model.auc:.4f}",
        f"baseline  holdout logloss {baseline.logloss:.4f}  auc {baseline.auc:.4f}",
        f"lift      logloss {model.logloss - baseline.logloss:+.4f}  "
        f"auc {model.auc - baseline.auc:+.4f}",
        "",
    ]
    if beats:
        lines.append("the model beats the archetype win-rate baseline on holdout log loss.")
    else:
        lines.append(
            "the model does NOT beat the archetype win-rate baseline on holdout log loss. "
            "The loop works; the model does not, on this much data."
        )
    lines.append("")
    lines.append("top features by gain:")
    for rank, row in enumerate(importance.head(5).itertuples(index=False), start=1):
        lines.append(f"  {rank}. {row.feature:<24} gain {row.gain:,.1f}  splits {row.splits}")
    lines.append("")
    lines.append(
        f"registered {REGISTERED_MODEL_NAME} version {version}. "
        "It serves nothing until `python -m pipeline.promote` moves the production alias."
    )
    return "\n".join(lines)


def run_training(
    *,
    warehouse: Path,
    experiment: str,
    tracking_uri: str,
    overrides: dict[str, Any],
    metrics: RunMetrics | None = None,
) -> int:
    """Both runs, end to end. Returns 0 whether or not the model wins.

    `metrics` is filled in with the run's counts and numbers when one is passed,
    so the command line records a `run_metrics` row without this function having
    to know where that row goes.
    """
    frame = load_features(warehouse)
    if frame.empty:
        # Not a failure. A warehouse that built cleanly and holds no feature
        # rows is a corpus that has not produced a modellable game yet, which
        # happens on a first run and on a small fixture set, and a scheduler
        # should carry on to the next stage rather than page somebody. A
        # warehouse that is missing entirely is still an error, and
        # `load_features` has already raised by here if it was.
        emit_summary(
            logger,
            "no training rows",
            {"dataset": FEATURE_TABLE, "warehouse": str(warehouse), "rows": 0, "trained": False},
            text=(
                f"{FEATURE_TABLE} in {warehouse} holds no rows, so there is nothing to train on "
                "and no version was registered. Land some games and run `python -m pipeline.gold` "
                "again."
            ),
            level=logging.WARNING,
        )
        if metrics is not None:
            metrics.rows_in = 0
            metrics.rows_out = 0
            metrics.rows_quarantined = 0
            metrics.extra = {"trained": False, "reason": f"{FEATURE_TABLE} is empty"}
        return 0
    train, holdout = split_by_date(frame)
    params = {**DEFAULT_PARAMS, **overrides}
    data_params = dataset_params(train, holdout)
    commit = git_commit()

    if tracking_uri.startswith("file:"):
        # MLflow 3 keeps the plain directory store behind an opt in, and a
        # directory is exactly what the default here is: a laptop run should
        # not need a server. Saying yes once, here, keeps that promise without
        # making every reader export a variable first. Point
        # `MLFLOW_TRACKING_URI` at the compose service and this never fires.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    # Two sibling runs rather than a parent and a child: they are two answers
    # to the same question on the same holdout, and a run table that puts them
    # side by side is the whole point of logging the baseline at all.
    with mlflow.start_run(run_name="lightgbm") as run:
        model_run = run.info.run_id
        model_holdout, importance, model_uri = log_model_run(
            train=train, holdout=holdout, params=params, data_params=data_params, commit=commit
        )
    with mlflow.start_run(run_name="baseline") as run:
        baseline_run = run.info.run_id
        baseline_holdout = log_baseline_run(
            train=train, holdout=holdout, data_params=data_params, commit=commit
        )
        beats = model_holdout.logloss < baseline_holdout.logloss
        # Logged on both runs, so whichever one a reader opens answers the
        # question, and as a metric rather than a tag so a run table sorts on it.
        mlflow.log_metric("beats_baseline", float(beats))
        mlflow.set_tag("compared_with", model_run)
    with mlflow.start_run(run_id=model_run):
        mlflow.log_metric("beats_baseline", float(beats))
        mlflow.log_metric("baseline_logloss", baseline_holdout.logloss)
        mlflow.log_metric("baseline_auc", baseline_holdout.auc)
        mlflow.set_tag("compared_with", baseline_run)

    # After both runs, because `beats_baseline` is one of the version's tags and
    # it is not known until the baseline has been scored on the same holdout.
    version = register_version(
        model_uri=model_uri, holdout=model_holdout, data_params=data_params, beats=beats
    )

    emit_summary(
        logger,
        "train summary",
        {
            "experiment": experiment,
            "tracking_uri": tracking_uri,
            "registered_version": version,
            "beats_baseline": beats,
            "holdout_logloss": round(model_holdout.logloss, 6),
            "holdout_auc": round(model_holdout.auc, 6),
            "baseline_logloss": round(baseline_holdout.logloss, 6),
            "baseline_auc": round(baseline_holdout.auc, 6),
            **data_params,
        },
        text=report(
            data_params=data_params,
            model=model_holdout,
            baseline=baseline_holdout,
            importance=importance,
            beats=beats,
            tracking_uri=tracking_uri,
            experiment=experiment,
            version=version,
        ),
    )
    if metrics is not None:
        rows = data_params["train_rows"] + data_params["holdout_rows"]
        metrics.rows_in = rows
        metrics.rows_out = rows
        metrics.rows_quarantined = 0
        metrics.extra = {
            "registered_version": version,
            "beats_baseline": beats,
            "holdout_logloss": model_holdout.logloss,
            "holdout_auc": model_holdout.auc,
            "baseline_logloss": baseline_holdout.logloss,
            "baseline_auc": baseline_holdout.auc,
            "experiment": experiment,
            **data_params,
        }
    return 0


def main(argv: list[str] | None = None) -> int:
    """Train the win-probability model from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.train",
        description="Train the per-turn win-probability model and record it in MLflow.",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI, else file:./data/mlruns)",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="LightGBM parameter overrides, for example learning_rate=0.1",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help="DuckDB warehouse holding features_turn",
    )
    args = parser.parse_args(argv)
    configure_logging(STAGE)
    try:
        with stage_run(STAGE) as metrics:
            return run_training(
                warehouse=args.warehouse,
                experiment=args.experiment,
                tracking_uri=args.tracking_uri or default_tracking_uri(),
                overrides=parse_params(args.params),
                metrics=metrics,
            )
    except (TrainingDataError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
