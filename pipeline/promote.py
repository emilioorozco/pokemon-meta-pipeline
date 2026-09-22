"""Promotion: the one place that decides what the serving stage is allowed to load.

`python -m pipeline.train` registers a version every time it runs. This command
is the gate between "a model exists" and "a model is served": it compares a
candidate version against whatever currently holds the `production` alias and
either moves the alias or leaves the candidate at `staging` with the reason
written onto it.

Three choices worth knowing before reading the code.

Aliases, not stages. MLflow's old `Staging` and `Production` stages are
deprecated in MLflow 3, and they were never quite the right shape anyway: a
stage is a property of the version, so two versions could not both be candidates
and a rollback meant editing history. An alias is a pointer, one name to one
version, so `models:/win-probability@production` is a stable address and a
rollback is moving it back.

The comparison reads tags first and the source run second. The trainer writes
the holdout numbers onto the version, so the usual path is one registry read.
A version registered by hand, or one whose tags were edited away, still has its
run id, so the numbers are fetched from the run's metrics instead of the command
refusing to judge it. Either way the numbers compared are holdout numbers, from
the split dbt made by date; nothing here re-scores anything, because a
promotion step that re-runs evaluation is a second implementation of evaluation.

Beating the baseline is a veto, not a term in the comparison. A model can be
better than the model before it and still worse than the archetype win-rate
group-by, and shipping that is worse than shipping nothing: it is the same
numbers with a LightGBM dependency in front of them. So `beats_baseline` has to
be 1 before the metric comparison is even reached, including for the first ever
promotion, when there is no incumbent to compare against.

The command prints one line and exits 0 whether it promoted or refused, because
a refusal is the system working. Exit 2 is for being asked about a version that
does not exist.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Final

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pipeline.config import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    STAGING_ALIAS,
    default_tracking_uri,
)

# The two metrics a version can be judged on, and which direction is better.
LOWER_IS_BETTER: Final[dict[str, bool]] = {"logloss": True, "auc": False}
# The other one, used only to break an exact tie on the primary metric.
TIE_BREAK: Final[dict[str, str]] = {"logloss": "auc", "auc": "logloss"}
# Where each metric lives: the version tag the trainer writes, and the metric
# name on the source run it falls back to.
TAG_OF: Final[dict[str, str]] = {"logloss": "holdout_logloss", "auc": "holdout_auc"}
DECISION_TAG: Final = "promotion_decision"


class PromotionError(RuntimeError):
    """The command cannot answer the question it was asked; the message says why."""


@dataclass(frozen=True)
class Candidate:
    """One registered version, reduced to the three numbers the rule needs."""

    version: str
    logloss: float
    auc: float
    beats_baseline: bool

    def metric(self, name: str) -> float:
        """The named holdout metric, `nan` when the version never recorded it."""
        return self.logloss if name == "logloss" else self.auc


def read_version(client: MlflowClient, version: str) -> Candidate:
    """A registered version's holdout numbers, from its tags or from its source run.

    Tags are strings, so a tag that is not a number is treated as a tag that is
    not there rather than crashing the comparison.
    """
    try:
        found = client.get_model_version(REGISTERED_MODEL_NAME, version)
    except MlflowException as error:
        raise PromotionError(
            f"no version {version} of {REGISTERED_MODEL_NAME} in the registry at "
            f"{client.tracking_uri}"
        ) from error
    tags = dict(found.tags)
    run_metrics: dict[str, float] = {}
    if found.run_id and not {*TAG_OF.values(), "beats_baseline"} <= set(tags):
        try:
            run_metrics = dict(client.get_run(found.run_id).data.metrics)
        except MlflowException:
            run_metrics = {}

    def number(metric: str) -> float:
        raw = tags.get(TAG_OF[metric])
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
        return float(run_metrics.get(metric, float("nan")))

    beats_raw = tags.get("beats_baseline", run_metrics.get("beats_baseline", 0))
    try:
        beats = float(beats_raw) == 1.0
    except ValueError:
        beats = False
    return Candidate(
        version=str(found.version),
        logloss=number("logloss"),
        auc=number("auc"),
        beats_baseline=beats,
    )


def latest_version(client: MlflowClient) -> str:
    """The highest version number of the registered model, as a string.

    By number rather than by creation time, because that is what "version 3" in
    a decision line means to whoever reads it later.
    """
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise PromotionError(
            f"{REGISTERED_MODEL_NAME} has no versions in the registry at "
            f"{client.tracking_uri}; run `python -m pipeline.train` first"
        )
    return str(max(int(version.version) for version in versions))


def production_version(client: MlflowClient) -> Candidate | None:
    """Whatever holds the `production` alias, or None the first time through."""
    try:
        found = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None
    return read_version(client, str(found.version))


def better(candidate: float, incumbent: float, lower_is_better: bool) -> bool:
    """Strictly better on this metric. A missing number is never better."""
    if math.isnan(candidate) or math.isnan(incumbent):
        return False
    return candidate < incumbent if lower_is_better else candidate > incumbent


def not_worse(candidate: float, incumbent: float, lower_is_better: bool) -> bool:
    """At least as good on this metric. A missing number is never good enough."""
    if math.isnan(candidate) or math.isnan(incumbent):
        return False
    return candidate <= incumbent if lower_is_better else candidate >= incumbent


def decide(candidate: Candidate, incumbent: Candidate | None, metric: str) -> tuple[bool, str]:
    """Promote or not, and the sentence that says why.

    The reason is written for someone reading it a month later in a tag, so it
    carries the numbers rather than a verdict: "0.6902 against 0.4123" survives
    being read out of context in a way that "worse" does not.
    """
    lower = LOWER_IS_BETTER[metric]
    mine = candidate.metric(metric)
    if math.isnan(mine):
        return False, f"rejected: version {candidate.version} has no holdout {metric} recorded"
    if not candidate.beats_baseline:
        return (
            False,
            f"rejected: version {candidate.version} does not beat the win-rate baseline "
            f"(beats_baseline=0, holdout {metric} {mine:.4f})",
        )
    if incumbent is None:
        return (
            True,
            f"promoted: version {candidate.version} is the first to hold {PRODUCTION_ALIAS} "
            f"(holdout {metric} {mine:.4f}, beats the baseline)",
        )
    theirs = incumbent.metric(metric)
    if better(mine, theirs, lower):
        return (
            True,
            f"promoted: version {candidate.version} improves holdout {metric} to {mine:.4f} "
            f"from {theirs:.4f} at version {incumbent.version}",
        )
    if mine == theirs:
        # An exact tie on the primary metric is not a coin flip: the two models
        # rank the holdout differently even when they score it the same, so the
        # other metric decides, and a candidate that cannot win there stays put.
        # Promoting on a true tie is deliberate, because a newer model was
        # trained on more recent games.
        tie = TIE_BREAK[metric]
        if not_worse(candidate.metric(tie), incumbent.metric(tie), LOWER_IS_BETTER[tie]):
            return (
                True,
                f"promoted: version {candidate.version} ties version {incumbent.version} on "
                f"holdout {metric} ({mine:.4f}) and is not worse on {tie}",
            )
        return (
            False,
            f"rejected: version {candidate.version} ties version {incumbent.version} on holdout "
            f"{metric} ({mine:.4f}) and is worse on {tie} "
            f"({candidate.metric(tie):.4f} against {incumbent.metric(tie):.4f})",
        )
    return (
        False,
        f"rejected: version {candidate.version} has holdout {metric} {mine:.4f} against "
        f"{theirs:.4f} at version {incumbent.version}",
    )


def apply(client: MlflowClient, candidate: Candidate, promote: bool, reason: str) -> None:
    """Move the alias, or park the candidate at `staging`, and record the reason on it.

    The reason is a tag on the version rather than a line in a log, because the
    question it answers ("why is this not the one being served?") is asked in
    front of the registry, not in front of a terminal.
    """
    client.set_model_version_tag(REGISTERED_MODEL_NAME, candidate.version, DECISION_TAG, reason)
    if promote:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, candidate.version
        )
        try:
            staged = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, STAGING_ALIAS)
        except MlflowException:
            return
        # A version that has just been promoted should not still be advertised
        # as the candidate waiting to be judged.
        if str(staged.version) == candidate.version:
            client.delete_registered_model_alias(REGISTERED_MODEL_NAME, STAGING_ALIAS)
        return
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, STAGING_ALIAS, candidate.version)


def run_promotion(*, candidate: str, metric: str, tracking_uri: str) -> int:
    """Compare, decide, move the alias, print one line. Returns 0 either way."""
    if tracking_uri.startswith("file:"):
        # Same opt in the trainer makes, for the same reason: a plain directory
        # is a supported store in MLflow 3 only when this is set, and a laptop
        # promotion should not need a server any more than a laptop train does.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    client = MlflowClient(tracking_uri=tracking_uri)
    version = latest_version(client) if candidate == "latest" else candidate
    current = read_version(client, version)
    incumbent = production_version(client)
    promote, reason = decide(current, incumbent, metric)
    apply(client, current, promote, reason)
    alias = PRODUCTION_ALIAS if promote else STAGING_ALIAS
    print(f"{reason}. {REGISTERED_MODEL_NAME} version {current.version} now holds @{alias}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Promote a registered model version from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.promote",
        description=(
            "Compare a registered model version against the one serving and move the "
            "production alias if it is at least as good."
        ),
    )
    parser.add_argument(
        "--candidate",
        default="latest",
        metavar="VERSION",
        help="registered model version to judge, or 'latest' (default)",
    )
    parser.add_argument(
        "--metric",
        default="logloss",
        choices=sorted(LOWER_IS_BETTER),
        help="the holdout metric the decision turns on (default: logloss)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI, else file:./data/mlruns)",
    )
    args = parser.parse_args(argv)
    try:
        return run_promotion(
            candidate=args.candidate,
            metric=args.metric,
            tracking_uri=args.tracking_uri or default_tracking_uri(),
        )
    except PromotionError as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
