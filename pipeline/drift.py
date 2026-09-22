"""Drift: the window a model was trained on against the window it is asked about now.

A model is a claim about a distribution, and the claim quietly expires. The
metagame is the obvious way it expires here: a set releases, two decks that did
not exist take a fifth of the tables, and a model whose archetype codes were
fitted months earlier starts answering about decks it has never seen. Nothing
breaks. The service keeps returning confident numbers, the holdout score in
MLflow keeps saying what it said the day it was trained, and the only way to
find out is to look.

This command is the looking. It compares the feature distributions of a
reference window (the training window, by default) against the most recent
window of `features_turn`, writes a report, logs it as an MLflow run, and
prints one verdict line. It never retrains and never touches the registry: the
output is a prompt to investigate, and `train` then `promote` is what acts on
it. That separation is the same one the rest of the model stage is built on.

Three choices worth knowing before reading the code.

The reference is a window, not a file. By default it is the rows dbt marked
`split = 'train'`, which is exactly what the last training run learned from. A
registered version can be named instead, and then the window comes from that
version's `train_from` and `train_to` tags, so a model still serving from three
retrains ago can be compared against today without anyone writing the dates
down. The tags are written by `pipeline.train`; a version whose tags were
edited away falls back to the source run's parameters, the same fallback
`pipeline.promote` makes and for the same reason.

The metric is the population stability index, and it is a per-feature number.
For each numeric feature, ten quantile bins are fitted on the reference, both
windows are binned with those edges, and

    PSI = sum over bins of (current_share - reference_share)
                           * ln(current_share / reference_share)

Every share is floored at `PSI_EPSILON` before the logarithm, because a bin the
current window never filled makes the ratio zero or infinite and neither is a
number. The floor is 1e-6, which is four orders of magnitude below anything a
reader acts on, so it bounds an empty bin's contribution instead of changing a
real one. The bins are quantiles of the reference rather than a uniform split
because most of these features are small integer counters whose mass sits in
two or three values, and a uniform split of `prizes_taken_self` would be eight
empty bins reporting on the binning. Duplicate quantile edges are collapsed, so
a feature with few distinct values simply gets fewer bins, and a constant
reference feature gets one bin and a PSI of zero, which is the truth about it.

The archetypes are compared as a mix rather than as two features. They are two
columns, this seat's deck and the other seat's, but a deck appearing on either
side of the table is the same event for the metagame, so the columns are pooled
and the comparison is over one distribution of archetype shares, which is also
what `mart_archetype_weekly` publishes. It gets three numbers: a chi-square
test of independence over the two windows' counts, a PSI over the shares with
each archetype as a bin, and the per-archetype share change that says which
decks moved. Archetypes under `SMALL_SHARE` in *both* windows are folded into
`other`, because a deck with four rows in each window is noise with a large
logarithm attached; a deck under the threshold in one window and over it in the
other is kept, since that is precisely the arrival this report exists to see.
The mix PSI is therefore read on a different scale from the feature ones: a
deck that held a tenth of the reference and none of the current window
contributes about 1.2 by itself, since the floor bounds an empty bin rather
than dropping it, so a short window runs to several units where a feature runs
to hundredths. The report says so, next to the share table that is the part
worth reading.

The label rate is reported and is not a PSI. A change in how often the seat
described by a row won is a hint about concept drift, the relationship between
features and label changing rather than the features moving, and PSI does not
measure that. It is labelled a hint in the report because on this corpus it is
one: the win rate is near one half by construction, since every game
contributes both seats.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from scipy.stats import chi2_contingency

from pipeline.config import (
    PIPELINE_DATA_DIR,
    REGISTERED_MODEL_NAME,
    WAREHOUSE_PATH,
    default_tracking_uri,
)
from pipeline.ml_features import CATEGORICAL, LABEL, MODEL_FEATURES

# The loader is imported rather than rewritten: it reads `features_turn` by an
# explicit column list, and a drift report built from a second copy of that
# list would be a report about a table nobody trains on.
from pipeline.observability import RunMetrics, configure_logging, emit_summary, stage_run
from pipeline.train import FEATURE_TABLE, TrainingDataError, load_features

logger = logging.getLogger(__name__)

STAGE: Final = "drift"
DEFAULT_EXPERIMENT: Final = "win-probability-drift"
# The features PSI is computed over: everything the model is given except the
# two archetype columns, which are compared as a mix instead.
NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    name for name in MODEL_FEATURES if name not in CATEGORICAL
)
PSI_BINS: Final = 10
PSI_EPSILON: Final = 1e-6
DEFAULT_THRESHOLD: Final = 0.2
DEFAULT_WINDOW_DAYS: Final = 30
# The usual reading of a population stability index, printed in the report so
# the numbers arrive with the scale they are read on.
MODERATE_PSI: Final = 0.1
SIGNIFICANT_PSI: Final = 0.2
# Archetypes below this share in both windows are one `other` bin.
SMALL_SHARE: Final = 0.02
# Under this many rows there is nothing to compare, and saying so is the
# answer. Exit 3 rather than 0, because a caller that ran this on a schedule
# needs to tell "looked, found nothing" from "could not look".
MIN_CURRENT_ROWS: Final = 20
EXIT_TOO_SMALL: Final = 3

REPORT_NAME: Final = "drift_report.md"
SUMMARY_NAME: Final = "drift_summary.json"
DEFAULT_OUT_DIR: Final = PIPELINE_DATA_DIR / "drift"


class DriftError(RuntimeError):
    """The comparison cannot be made; the message says which half is missing."""


@dataclass(frozen=True)
class Window:
    """One side of the comparison: the rows, how they were chosen, and when they were played."""

    name: str
    chosen_by: str
    frame: pd.DataFrame

    @property
    def rows(self) -> int:
        """How many feature rows the window holds."""
        return len(self.frame)

    @property
    def games(self) -> int:
        """How many distinct games those rows came from."""
        return int(self.frame["game_id"].nunique())

    @property
    def first(self) -> str:
        """The earliest play date in the window, as an ISO string."""
        return str(self.frame["play_date"].min().date())

    @property
    def last(self) -> str:
        """The latest play date in the window, as an ISO string."""
        return str(self.frame["play_date"].max().date())

    @property
    def label_rate(self) -> float:
        """The share of rows whose seat won the game."""
        return float(self.frame[LABEL].astype(float).mean())


@dataclass(frozen=True)
class FeatureDrift:
    """One numeric feature's PSI and the two windows' centres, for reading it against."""

    feature: str
    psi: float
    reference_mean: float
    reference_median: float
    current_mean: float
    current_median: float
    flagged: bool


@dataclass(frozen=True)
class ArchetypeShare:
    """One archetype's share of each window, and the change between them."""

    archetype: str
    reference: float
    current: float

    @property
    def change(self) -> float:
        """Current share minus reference share, in share points."""
        return self.current - self.reference


@dataclass(frozen=True)
class MixDrift:
    """The archetype mix compared: a PSI over shares, a chi-square test, and the movers."""

    psi: float
    chi_square: float
    p_value: float
    approximate: bool
    shares: tuple[ArchetypeShare, ...]
    flagged: bool

    def movers(self, count: int) -> tuple[ArchetypeShare, ...]:
        """The archetypes whose share moved most, largest absolute change first."""
        ordered = sorted(self.shares, key=lambda share: abs(share.change), reverse=True)
        return tuple(ordered[:count])


@dataclass(frozen=True)
class DriftReport:
    """Everything the markdown and the summary are rendered from."""

    reference: Window
    current: Window
    features: tuple[FeatureDrift, ...]
    mix: MixDrift
    threshold: float
    window_days: int
    as_of: str
    overlap_rows: int
    warehouse: Path

    @property
    def max_psi(self) -> float:
        """The largest numeric feature PSI, or 0.0 when there are no numeric features."""
        return max((feature.psi for feature in self.features), default=0.0)

    @property
    def max_psi_feature(self) -> str:
        """The feature holding `max_psi`."""
        return self.features[0].feature if self.features else "none"

    @property
    def drifted(self) -> bool:
        """True when any numeric PSI or the archetype mix PSI reaches the threshold."""
        return self.max_psi >= self.threshold or self.mix.psi >= self.threshold

    @property
    def verdict(self) -> str:
        """The one line the command prints and the report repeats."""
        verb = "drift flagged" if self.drifted else "no drift"
        return (
            f"{verb}: max feature PSI {self.max_psi:.4f} ({self.max_psi_feature}), "
            f"archetype mix PSI {self.mix.psi:.4f}, threshold {self.threshold:.2f}, "
            f"{self.current.rows} current rows against {self.reference.rows} reference rows"
        )


def with_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frame with `play_date` as a pandas timestamp, whatever DuckDB handed back."""
    dated = frame.copy()
    dated["play_date"] = pd.to_datetime(dated["play_date"])
    return dated


def bin_edges(reference: np.ndarray) -> np.ndarray:
    """The interior edges of `PSI_BINS` quantile bins fitted on the reference values.

    Only the interior edges: the outer two are dropped so that the first and
    last bins run to minus and plus infinity, and a current value outside
    anything the reference held lands in an end bin instead of nowhere.
    Duplicate edges are collapsed, which is what turns a counter with three
    distinct values into three bins rather than ten, nine of them empty.
    """
    quantiles = np.quantile(reference, np.linspace(0.0, 1.0, PSI_BINS + 1))
    return np.unique(quantiles[1:-1])


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The share of `values` falling in each bin defined by those interior edges."""
    index = np.digitize(values, edges, right=False)
    counts = np.bincount(index, minlength=len(edges) + 1).astype(float)
    return np.asarray(counts / counts.sum(), dtype=float)


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI of two share vectors, each share floored at `PSI_EPSILON` before the logarithm.

    The floor is not smoothing: it is what keeps an empty bin from making the
    sum infinite, and at 1e-6 it is far below the smallest difference anyone
    reads off this report.
    """
    safe_reference = np.clip(reference, PSI_EPSILON, None)
    safe_current = np.clip(current, PSI_EPSILON, None)
    return float(np.sum((safe_current - safe_reference) * np.log(safe_current / safe_reference)))


def feature_drift(
    feature: str, reference: pd.Series, current: pd.Series, threshold: float
) -> FeatureDrift:
    """One feature compared across the two windows: PSI, both centres, and the flag."""
    reference_values = reference.to_numpy(dtype=float)
    current_values = current.to_numpy(dtype=float)
    edges = bin_edges(reference_values)
    psi = population_stability_index(
        bin_shares(reference_values, edges), bin_shares(current_values, edges)
    )
    return FeatureDrift(
        feature=feature,
        psi=psi,
        reference_mean=float(np.mean(reference_values)),
        reference_median=float(np.median(reference_values)),
        current_mean=float(np.mean(current_values)),
        current_median=float(np.median(current_values)),
        flagged=psi >= threshold,
    )


def archetype_counts(frame: pd.DataFrame) -> pd.Series:
    """How often each archetype appears in a window, both seats pooled.

    Pooled because the two columns are the same deck seen from two sides, and
    the question is what the metagame looks like, not what the left column
    looks like. A row therefore contributes two observations, which is the same
    counting `mart_archetype_weekly` does over seats.
    """
    pooled = pd.concat([frame[column] for column in CATEGORICAL], ignore_index=True)
    return pooled.value_counts()


def fold_small(reference: pd.Series, current: pd.Series) -> pd.DataFrame:
    """Counts per archetype in both windows, with the mutually tiny ones folded into `other`.

    Tiny in *both* windows. An archetype that was absent from the reference and
    holds five per cent now is the arrival this whole command exists to notice,
    so it keeps its own row however small it was before.
    """
    keys = sorted(set(reference.index) | set(current.index))
    table = pd.DataFrame(
        {
            "archetype": keys,
            "reference": [float(reference.get(key, 0)) for key in keys],
            "current": [float(current.get(key, 0)) for key in keys],
        }
    )
    reference_total = max(table["reference"].sum(), 1.0)
    current_total = max(table["current"].sum(), 1.0)
    small = (table["reference"] / reference_total < SMALL_SHARE) & (
        table["current"] / current_total < SMALL_SHARE
    )
    if not small.any():
        return table
    folded = table[~small].copy()
    other = pd.DataFrame(
        {
            "archetype": ["other"],
            "reference": [float(table.loc[small, "reference"].sum())],
            "current": [float(table.loc[small, "current"].sum())],
        }
    )
    return pd.concat([folded, other], ignore_index=True)


def chi_square(table: pd.DataFrame) -> tuple[float, float, bool]:
    """Chi-square statistic, p-value, and whether the p-value is an approximation.

    `scipy.stats.chi2_contingency` arrives with scikit-learn, which the model
    stage already depends on, so this is not a dependency added for one test.
    The flag is here anyway because the p-value is asymptotic: on a corpus this
    small several expected cell counts are under five, where the chi-square
    approximation to the true distribution is exactly that. It is reported so
    the report can say so, not so the number can be trusted to three decimals.
    """
    counts = table[["reference", "current"]].to_numpy(dtype=float)
    if counts.shape[0] < 2 or counts.sum() == 0:
        # One archetype, or no rows: there is no table to test independence on.
        return 0.0, 1.0, True
    result = chi2_contingency(counts)
    expected = np.asarray(result.expected_freq, dtype=float)
    return float(result.statistic), float(result.pvalue), bool((expected < 5).any())


def mix_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold: float) -> MixDrift:
    """The archetype mix of both windows, compared three ways."""
    table = fold_small(archetype_counts(reference), archetype_counts(current))
    reference_shares = (table["reference"] / max(table["reference"].sum(), 1.0)).to_numpy(
        dtype=float
    )
    current_shares = (table["current"] / max(table["current"].sum(), 1.0)).to_numpy(dtype=float)
    psi = population_stability_index(reference_shares, current_shares)
    statistic, p_value, approximate = chi_square(table)
    shares = tuple(
        ArchetypeShare(archetype=str(name), reference=float(before), current=float(after))
        for name, before, after in zip(
            table["archetype"], reference_shares, current_shares, strict=True
        )
    )
    return MixDrift(
        psi=psi,
        chi_square=statistic,
        p_value=p_value,
        approximate=approximate,
        shares=shares,
        flagged=psi >= threshold,
    )


def version_training_window(client: MlflowClient, version: str) -> tuple[str, str]:
    """The `train_from` and `train_to` of a registered version, from its tags or its run.

    Tags first, because that is one registry read and it is what the trainer
    writes. A version registered by hand, or one whose tags were edited away,
    still carries its run id, and the run logged the same two dates as
    parameters, so the fallback is a fact about the same training rather than a
    guess.
    """
    try:
        found = client.get_model_version(REGISTERED_MODEL_NAME, version)
    except MlflowException as error:
        raise DriftError(
            f"no version {version} of {REGISTERED_MODEL_NAME} in the registry at "
            f"{client.tracking_uri}; pass --reference train to use the training split instead"
        ) from error
    dates = {name: found.tags.get(name) for name in ("train_from", "train_to")}
    if not all(dates.values()) and found.run_id:
        try:
            params = client.get_run(found.run_id).data.params
        except MlflowException:
            params = {}
        dates = {name: dates[name] or params.get(name) for name in dates}
    if not all(dates.values()):
        raise DriftError(
            f"version {version} of {REGISTERED_MODEL_NAME} records no train_from and train_to, "
            "on the version or on its source run, so its training window is unknown"
        )
    return str(dates["train_from"]), str(dates["train_to"])


def reference_window(frame: pd.DataFrame, reference: str, tracking_uri: str) -> Window:
    """The rows to compare against: the training split, or a version's training window."""
    if reference == "train":
        selected = frame[frame["split"] == "train"]
        chosen_by = f"`split = 'train'` in {FEATURE_TABLE}"
    else:
        client = MlflowClient(tracking_uri=tracking_uri)
        first, last = version_training_window(client, reference)
        selected = frame[
            (frame["play_date"] >= pd.Timestamp(first)) & (frame["play_date"] <= pd.Timestamp(last))
        ]
        chosen_by = (
            f"the training window of {REGISTERED_MODEL_NAME} version {reference} "
            f"({first} to {last})"
        )
    if selected.empty:
        raise DriftError(
            f"the reference window selected by {chosen_by} is empty, so there is nothing to "
            f"compare against; check the corpus date range in {FEATURE_TABLE}"
        )
    # The row index is kept rather than reset: both windows are slices of the
    # same frame, so their indexes are what `overlap` intersects.
    return Window(name="reference", chosen_by=chosen_by, frame=selected)


def current_window(frame: pd.DataFrame, as_of: date, window_days: int) -> Window:
    """The last `window_days` calendar days up to `as_of`, both ends included."""
    last = pd.Timestamp(as_of)
    first = last - pd.Timedelta(days=window_days - 1)
    selected = frame[(frame["play_date"] >= first) & (frame["play_date"] <= last)]
    chosen_by = f"the {window_days} days from {first.date().isoformat()} to {as_of.isoformat()}"
    return Window(name="current", chosen_by=chosen_by, frame=selected)


def overlap(reference: Window, current: Window) -> int:
    """How many feature rows are in both windows.

    Not a detail on a ten-day corpus: a thirty-day current window swallows the
    training window whole, every PSI is then a distribution compared with
    itself plus a few rows, and the near-zero numbers that come out are
    arithmetic rather than evidence. The count is printed and put in both
    outputs so nobody reads a reassuring report that was reassuring by
    construction.
    """
    return len(reference.frame.index.intersection(current.frame.index))


def build_report(
    *,
    frame: pd.DataFrame,
    reference: str,
    window_days: int,
    as_of: date,
    threshold: float,
    tracking_uri: str,
    warehouse: Path,
) -> DriftReport:
    """Both windows, every metric, sorted with the loudest feature first."""
    before = reference_window(frame, reference, tracking_uri)
    now = current_window(frame, as_of, window_days)
    features = tuple(
        sorted(
            (
                feature_drift(name, before.frame[name], now.frame[name], threshold)
                for name in NUMERIC_FEATURES
            ),
            key=lambda drift: drift.psi,
            reverse=True,
        )
        if now.rows
        else ()
    )
    return DriftReport(
        reference=before,
        current=now,
        features=features,
        mix=(
            mix_drift(before.frame, now.frame, threshold)
            if now.rows
            else MixDrift(0.0, 0.0, 1.0, True, (), False)
        ),
        threshold=threshold,
        window_days=window_days,
        as_of=as_of.isoformat(),
        overlap_rows=overlap(before, now) if now.rows else 0,
        warehouse=warehouse,
    )


def windows_table(report: DriftReport) -> list[str]:
    """The two windows as a markdown table: rows, games, dates, and how each was chosen."""
    lines = [
        "| Window | Rows | Games | First play date | Last play date | Chosen by |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for window in (report.reference, report.current):
        lines.append(
            f"| {window.name} | {window.rows} | {window.games} | {window.first} | "
            f"{window.last} | {window.chosen_by} |"
        )
    return lines


def features_table(report: DriftReport) -> list[str]:
    """Every numeric feature as a markdown row, largest PSI first."""
    lines = [
        "| Feature | PSI | Reference mean | Reference median | Current mean | "
        "Current median | Flag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for drift in report.features:
        flag = "**over**" if drift.flagged else "ok"
        lines.append(
            f"| `{drift.feature}` | {drift.psi:.4f} | {drift.reference_mean:.3f} | "
            f"{drift.reference_median:.3f} | {drift.current_mean:.3f} | "
            f"{drift.current_median:.3f} | {flag} |"
        )
    return lines


def mix_table(report: DriftReport, limit: int = 10) -> list[str]:
    """The archetype mix as a markdown table, the biggest movers first."""
    lines = [
        "| Archetype | Reference share | Current share | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for share in report.mix.movers(limit):
        lines.append(
            f"| `{share.archetype}` | {share.reference:.1%} | {share.current:.1%} | "
            f"{share.change:+.1%} |"
        )
    return lines


def markdown(report: DriftReport) -> str:
    """The whole report as one markdown document."""
    mix = report.mix
    approximate = (
        " Several expected cell counts are under five, so the p-value is the asymptotic "
        "approximation and should be read as a direction, not as a decision."
        if mix.approximate
        else ""
    )
    lines = [
        "# Feature drift report",
        "",
        f"`python -m pipeline.drift`, over `{FEATURE_TABLE}` in `{report.warehouse}`, "
        f"as of {report.as_of}.",
        "",
        f"**{report.verdict}**",
        "",
        "## Windows",
        "",
        *windows_table(report),
        "",
    ]
    if report.overlap_rows:
        lines += [
            f"{report.overlap_rows} of the {report.current.rows} current rows are also in the "
            "reference window. The windows overlap, so the numbers below understate any real "
            "movement: a distribution largely compared with itself has a PSI near zero whatever "
            "the world did. Shorten `--window-days`, or wait for the corpus to outgrow it.",
            "",
        ]
    lines += [
        "## Numeric features",
        "",
        *features_table(report),
        "",
        "## Archetype mix",
        "",
        f"Both archetype columns pooled, so a row contributes its own deck and its opponent's. "
        f"PSI {mix.psi:.4f} against a threshold of {report.threshold:.2f}. Chi-square "
        f"{mix.chi_square:.3f}, p {mix.p_value:.4f}.{approximate} Archetypes under "
        f"{SMALL_SHARE:.0%} of both windows are folded into `other`.",
        "",
        *mix_table(report),
        "",
        "## Label rate (a concept-drift hint, not a PSI)",
        "",
        "| Window | Rows | `won` share |",
        "| --- | ---: | ---: |",
        f"| reference | {report.reference.rows} | {report.reference.label_rate:.1%} |",
        f"| current | {report.current.rows} | {report.current.label_rate:.1%} |",
        "",
        "A moving win rate is a hint that the relationship between the features and the label "
        "changed, which is what no PSI above can see. It is a hint and not a finding: every "
        "game contributes both seats, so the corpus win rate is near one half by construction "
        "and a window narrow enough to move it is a window narrow enough to be noise.",
        "",
        "## Reading these numbers",
        "",
        f"The usual reading of a population stability index: below {MODERATE_PSI} is stable, "
        f"{MODERATE_PSI} to {SIGNIFICANT_PSI} is moderate movement worth a look, and "
        f"{SIGNIFICANT_PSI} or more is a significant shift. The flag above is that last band, "
        f"at a threshold of {report.threshold:.2f}.",
        "",
        "The archetype mix runs larger than the feature numbers and is read on its own scale. An "
        "archetype that held a share of the reference and none of the current window contributes "
        f"about {abs(np.log(PSI_EPSILON)):.0f} times that share on its own, because an empty bin "
        "is floored rather than dropped, so a short window in which two decks simply did not "
        "queue can reach a mix PSI of several units. Read the share table under it before the "
        "number: which decks moved, and by how much, is the part that means anything.",
        "",
        "On a corpus this small the numbers are noisy: a window of a few dozen games can move a "
        "PSI past the threshold with nothing behind it but which decks happened to be queued "
        "that week. The flag is a prompt to look, not an instruction to retrain. Investigate "
        "first, and if the shift is real, retrain with `python -m pipeline.train` and let "
        "`python -m pipeline.promote` decide whether the new model is any better.",
        "",
    ]
    return "\n".join(lines)


def summary(report: DriftReport) -> dict[str, Any]:
    """The same report as machine-readable JSON, for whatever reads it next."""
    return {
        "as_of": report.as_of,
        "warehouse": str(report.warehouse),
        "threshold": report.threshold,
        "drifted": report.drifted,
        "verdict": report.verdict,
        "windows": {
            "reference": {
                "rows": report.reference.rows,
                "games": report.reference.games,
                "from": report.reference.first,
                "to": report.reference.last,
                "chosen_by": report.reference.chosen_by,
            },
            "current": {
                "rows": report.current.rows,
                "games": report.current.games,
                "from": report.current.first,
                "to": report.current.last,
                "chosen_by": report.current.chosen_by,
                "window_days": report.window_days,
            },
            "overlap_rows": report.overlap_rows,
        },
        "features": [
            {
                "feature": drift.feature,
                "psi": drift.psi,
                "reference_mean": drift.reference_mean,
                "reference_median": drift.reference_median,
                "current_mean": drift.current_mean,
                "current_median": drift.current_median,
                "flagged": drift.flagged,
            }
            for drift in report.features
        ],
        "max_psi": report.max_psi,
        "max_psi_feature": report.max_psi_feature,
        "archetype_mix": {
            "psi": report.mix.psi,
            "chi_square": report.mix.chi_square,
            "p_value": report.mix.p_value,
            "p_value_is_approximate": report.mix.approximate,
            "flagged": report.mix.flagged,
            "shares": [
                {
                    "archetype": share.archetype,
                    "reference": share.reference,
                    "current": share.current,
                    "change": share.change,
                }
                for share in report.mix.shares
            ],
        },
        "label_rate": {
            "reference": report.reference.label_rate,
            "current": report.current.label_rate,
        },
        "psi_bins": PSI_BINS,
        "psi_epsilon": PSI_EPSILON,
    }


def write_outputs(report: DriftReport, out_dir: Path) -> tuple[Path, Path]:
    """Both files on disk, and where they went."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    summary_path = out_dir / SUMMARY_NAME
    report_path.write_text(markdown(report), encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, summary_path


def log_run(
    report: DriftReport, reference: str, report_path: Path, summary_path: Path, experiment: str
) -> None:
    """One MLflow run carrying the two files, the windows as parameters and the verdict as metrics.

    A run rather than a file beside the warehouse, for the same reason training
    is a run: the question asked of a drift report is almost always "and what
    did it say last month", and a directory of overwritten markdown cannot
    answer it.
    """
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="drift"):
        mlflow.log_params(
            {
                "reference": reference,
                "reference_from": report.reference.first,
                "reference_to": report.reference.last,
                "reference_rows": report.reference.rows,
                "current_from": report.current.first,
                "current_to": report.current.last,
                "current_rows": report.current.rows,
                "overlap_rows": report.overlap_rows,
                "window_days": report.window_days,
                "as_of": report.as_of,
                "psi_threshold": report.threshold,
                "psi_bins": PSI_BINS,
            }
        )
        mlflow.set_tags({"stage": "drift", "dataset": FEATURE_TABLE})
        mlflow.log_metrics(
            {
                "max_psi": report.max_psi,
                "archetype_mix_psi": report.mix.psi,
                "archetype_mix_chi_square": report.mix.chi_square,
                "archetype_mix_p_value": report.mix.p_value,
                "drifted": float(report.drifted),
                "label_rate_reference": report.reference.label_rate,
                "label_rate_current": report.current.label_rate,
            }
        )
        # One metric per feature as well, so a run table sorts on the feature
        # that moved rather than only on the loudest one.
        mlflow.log_metrics({f"psi_{drift.feature}": drift.psi for drift in report.features})
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(summary_path))


def console(report: DriftReport, report_path: Path, summary_path: Path) -> str:
    """The block the command prints: the windows, the numbers, the verdict, the paths."""
    lines = [
        f"reference: {report.reference.rows:>5} rows, {report.reference.games} games, "
        f"{report.reference.first} to {report.reference.last}  ({report.reference.chosen_by})",
        f"current:   {report.current.rows:>5} rows, {report.current.games} games, "
        f"{report.current.first} to {report.current.last}  ({report.current.chosen_by})",
    ]
    if report.overlap_rows:
        lines.append(
            f"overlap:   {report.overlap_rows} rows are in both windows, so every number below "
            "understates the movement."
        )
    lines += ["", f"{'feature':<24} {'PSI':>8}  {'ref mean':>9} {'cur mean':>9}"]
    for drift in report.features:
        flag = "  over" if drift.flagged else ""
        lines.append(
            f"{drift.feature:<24} {drift.psi:>8.4f}  {drift.reference_mean:>9.3f} "
            f"{drift.current_mean:>9.3f}{flag}"
        )
    lines += [
        "",
        f"archetype mix  PSI {report.mix.psi:.4f}  chi-square {report.mix.chi_square:.3f}  "
        f"p {report.mix.p_value:.4f}" + (" (approximate)" if report.mix.approximate else ""),
    ]
    for share in report.mix.movers(3):
        lines.append(
            f"  {share.archetype:<28} {share.reference:>6.1%} -> {share.current:>6.1%}  "
            f"{share.change:+.1%}"
        )
    lines += [
        "",
        f"label rate (hint)  reference {report.reference.label_rate:.1%}  "
        f"current {report.current.label_rate:.1%}",
        "",
        report.verdict,
        f"report:  {report_path}",
        f"summary: {summary_path}",
    ]
    return "\n".join(lines)


def run_drift(
    *,
    warehouse: Path,
    reference: str,
    window_days: int,
    as_of: date | None,
    threshold: float,
    tracking_uri: str,
    experiment: str,
    out_dir: Path,
    metrics: RunMetrics | None = None,
) -> int:
    """The whole comparison. Returns 0 whether or not it flagged, 3 with nothing to compare."""
    if tracking_uri.startswith("file:"):
        # The same opt in the other model-stage commands make: MLflow 3 keeps
        # the plain directory store behind this, and a laptop drift check
        # should not need a server any more than a laptop train does.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)

    frame = with_dates(load_features(warehouse))
    if frame.empty:
        raise DriftError(f"{FEATURE_TABLE} in {warehouse} is empty, so there is nothing to compare")
    effective_as_of = as_of or frame["play_date"].max().date()
    report = build_report(
        frame=frame,
        reference=reference,
        window_days=window_days,
        as_of=effective_as_of,
        threshold=threshold,
        tracking_uri=tracking_uri,
        warehouse=warehouse,
    )
    if report.current.rows < MIN_CURRENT_ROWS:
        emit_summary(
            logger,
            "drift window too small",
            {
                "window_days": window_days,
                "as_of": effective_as_of.isoformat(),
                "current_rows": report.current.rows,
                "min_current_rows": MIN_CURRENT_ROWS,
            },
            text=(
                f"the {window_days} days up to {effective_as_of.isoformat()} hold "
                f"{report.current.rows} feature rows, fewer than the {MIN_CURRENT_ROWS} this "
                "compares on. Every distribution over that few rows is noise, so no report was "
                "written. Widen --window-days, move --as-of, or wait for more games."
            ),
            level=logging.WARNING,
        )
        if metrics is not None:
            metrics.rows_in = report.reference.rows + report.current.rows
            metrics.rows_out = 0
            metrics.rows_quarantined = 0
            metrics.extra = {
                "reason": "window too small",
                "current_rows": report.current.rows,
                "min_current_rows": MIN_CURRENT_ROWS,
                "exit_code": EXIT_TOO_SMALL,
            }
        return EXIT_TOO_SMALL

    report_path, summary_path = write_outputs(report, out_dir)
    log_run(report, reference, report_path, summary_path, experiment)
    emit_summary(
        logger,
        "drift summary",
        {
            "reference_rows": report.reference.rows,
            "current_rows": report.current.rows,
            "max_psi": report.max_psi,
            "archetype_mix_psi": report.mix.psi,
            "drifted": report.drifted,
            "flagged_features": [drift.feature for drift in report.features if drift.flagged],
            "report_path": str(report_path),
            "summary_path": str(summary_path),
        },
        text=console(report, report_path, summary_path),
    )
    if metrics is not None:
        metrics.rows_in = report.reference.rows + report.current.rows
        metrics.rows_out = len(report.features)
        metrics.rows_quarantined = 0
        metrics.extra = {
            "reference_rows": report.reference.rows,
            "current_rows": report.current.rows,
            "max_psi": report.max_psi,
            "archetype_mix_psi": report.mix.psi,
            "drifted": report.drifted,
            "flagged_features": [drift.feature for drift in report.features if drift.flagged],
        }
    return 0


def main(argv: list[str] | None = None) -> int:
    """Compare recent feature distributions against the training window, from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.drift",
        description=(
            "Compare the feature distributions of a reference window against the most recent "
            "window of features_turn, and report what moved."
        ),
    )
    parser.add_argument(
        "--reference",
        default="train",
        metavar="train|VERSION",
        help=(
            "the window to compare against: 'train' (default) for the training split, or a "
            f"registered version of {REGISTERED_MODEL_NAME}, whose train_from and train_to "
            "tags then select the rows"
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        metavar="N",
        help=f"how many days the current window covers (default: {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        metavar="DATE",
        help="the last day of the current window (default: the latest play date in the table)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="PSI",
        help=f"the PSI at which a feature is flagged (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI, else file:./data/mlruns)",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help="DuckDB warehouse holding features_turn",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="PATH",
        help=f"where {REPORT_NAME} and {SUMMARY_NAME} are written (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    if args.window_days < 1:
        parser.exit(2, f"{parser.prog}: --window-days must be at least 1\n")
    configure_logging(STAGE)
    try:
        with stage_run(STAGE) as metrics:
            return run_drift(
                warehouse=args.warehouse,
                reference=args.reference,
                window_days=args.window_days,
                as_of=args.as_of,
                threshold=args.psi_threshold,
                tracking_uri=args.tracking_uri or default_tracking_uri(),
                experiment=args.experiment,
                out_dir=args.out_dir,
                metrics=metrics,
            )
    except (DriftError, TrainingDataError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
