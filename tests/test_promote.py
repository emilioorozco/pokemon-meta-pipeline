"""The promotion gate, against a real registry on a real file store.

Marked `ml` and skipped by the default run, for the same reason the training
tests are: this trains three LightGBM models. It trains them rather than faking
registry rows because the thing under test is whether the numbers the trainer
writes onto a version are the numbers the promoter reads back, and a fixture
that writes the tags itself would assert that the test knows its own tag names.

Three versions, in the order a week of retraining would produce them:

1. a deliberately short run, good enough to beat the win-rate baseline;
2. a single boosting round on two leaves, which is close to a constant
   prediction and worse than version 1;
3. a long run, better than both.

The assertions follow from that: version 1 takes `production` because nothing
held it, version 2 is refused and parked at `staging` with the reason written
onto it, and version 3 moves the alias. `decide` is then tested directly on
hand-built candidates for the cases three training runs cannot be made to
produce reliably, such as an exact tie.

The synthetic feature table is `tests/test_train.py`'s, imported rather than
copied, so the two suites cannot disagree about what a feature row looks like.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import duckdb
import pytest
from mlflow.tracking import MlflowClient

from pipeline import promote, train
from pipeline.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, STAGING_ALIAS
from tests.test_train import synthetic_features

pytestmark = pytest.mark.ml

# Short but real, near-constant, and long. The middle one is the worse model.
WEAK: Final[list[str]] = ["num_boost_round=20", "min_data_in_leaf=10"]
BROKEN: Final[list[str]] = ["num_boost_round=1", "num_leaves=2", "min_data_in_leaf=10"]
STRONG: Final[list[str]] = ["num_boost_round=200", "min_data_in_leaf=10"]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A DuckDB file holding nothing but a synthetic `features_turn`."""
    path = tmp_path_factory.mktemp("promote-warehouse") / "meta.duckdb"
    frame = synthetic_features()
    connection = duckdb.connect(str(path))
    try:
        connection.register("frame", frame)
        connection.execute("create table features_turn as select * from frame")
    finally:
        connection.close()
    return path


@pytest.fixture(scope="module")
def registry(warehouse: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Three training runs into one temporary tracking store, so the registry has three versions.

    Module scoped: the three runs are the expensive part and every test below
    reads the same registry. They are ordered, and the tests that follow move
    the aliases, so the tests in this module are ordered too.
    """
    tracking_uri = f"file:{tmp_path_factory.mktemp('promote-mlruns')}"
    for params in (WEAK, BROKEN, STRONG):
        code = train.main(
            [
                "--warehouse",
                str(warehouse),
                "--tracking-uri",
                tracking_uri,
                "--experiment",
                "test-promotion",
                "--params",
                *params,
            ]
        )
        assert code == 0
    yield tracking_uri


def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def alias_version(tracking_uri: str, alias: str) -> str | None:
    """Which version holds an alias, or None when nothing does."""
    try:
        found = client(tracking_uri).get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except Exception:
        return None
    return str(found.version)


def version_tags(tracking_uri: str, version: str) -> dict[str, str]:
    return dict(client(tracking_uri).get_model_version(REGISTERED_MODEL_NAME, version).tags)


def test_training_registered_a_version_per_run_with_its_holdout_numbers(registry: str) -> None:
    versions = client(registry).search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    assert {str(version.version) for version in versions} == {"1", "2", "3"}
    tags = version_tags(registry, "1")
    for name in ("holdout_logloss", "holdout_auc", "beats_baseline", "train_to", "holdout_to"):
        assert name in tags, name
    assert tags["beats_baseline"] == "1"
    assert float(tags["holdout_logloss"]) > 0.0
    # The baseline is a yardstick, not a candidate: only the LightGBM run is registered.
    assert len(versions) == 3
    # Nothing is served until the promotion step says so.
    assert alias_version(registry, PRODUCTION_ALIAS) is None


def test_the_first_version_takes_production(
    registry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert promote.main(["--candidate", "1", "--tracking-uri", registry]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("promoted:")
    assert f"@{PRODUCTION_ALIAS}" in printed
    assert alias_version(registry, PRODUCTION_ALIAS) == "1"
    assert version_tags(registry, "1")[promote.DECISION_TAG].startswith("promoted:")


def test_a_worse_version_is_refused_and_parked_at_staging(
    registry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the whole command: a retrain that got worse does not ship.

    It still exits 0. A refusal is the gate working, and a non-zero exit here
    would turn every scheduled retrain of a model that happened not to improve
    into a red build.
    """
    assert promote.main(["--candidate", "2", "--tracking-uri", registry]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("rejected:")
    assert f"@{STAGING_ALIAS}" in printed

    assert alias_version(registry, PRODUCTION_ALIAS) == "1", "production must not have moved"
    assert alias_version(registry, STAGING_ALIAS) == "2"
    reason = version_tags(registry, "2")[promote.DECISION_TAG]
    assert reason.startswith("rejected: ")
    # The reason carries the numbers, not just a verdict.
    assert "version 2" in reason


def test_a_better_version_moves_production(
    registry: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`latest` resolves to version 3, and version 3 is better than version 1."""
    assert promote.main(["--candidate", "latest", "--tracking-uri", registry]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("promoted:")
    assert alias_version(registry, PRODUCTION_ALIAS) == "3"
    # Promotion clears the candidate marker rather than leaving one version at both.
    assert alias_version(registry, STAGING_ALIAS) == "2"

    strong = float(version_tags(registry, "3")["holdout_logloss"])
    weak = float(version_tags(registry, "1")["holdout_logloss"])
    broken = float(version_tags(registry, "2")["holdout_logloss"])
    assert strong < weak < broken, (strong, weak, broken)


def test_a_missing_candidate_exits_2(registry: str) -> None:
    with pytest.raises(SystemExit) as raised:
        promote.main(["--candidate", "99", "--tracking-uri", registry])
    assert raised.value.code == 2
    # And the alias is where the last successful promotion left it.
    assert alias_version(registry, PRODUCTION_ALIAS) == "3"


def test_auc_can_be_the_primary_metric(registry: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Same rule, other direction: higher is better, and log loss becomes the tie-break."""
    assert promote.main(["--candidate", "2", "--metric", "auc", "--tracking-uri", registry]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("rejected:")
    assert "auc" in printed
    assert alias_version(registry, PRODUCTION_ALIAS) == "3"


def candidate(version: str, logloss: float, auc: float, beats: bool = True) -> promote.Candidate:
    return promote.Candidate(version=version, logloss=logloss, auc=auc, beats_baseline=beats)


def test_the_rule_in_isolation() -> None:
    """The cases three training runs cannot be made to produce on demand.

    An exact tie on the primary metric is the interesting one. Two models that
    score the same log loss are not the same model, so the other metric decides,
    and a true tie on both promotes: the newer version was trained on more
    recent games, which is the only thing left to prefer.
    """
    incumbent = candidate("1", logloss=0.40, auc=0.80)

    promoted, reason = promote.decide(candidate("2", 0.35, 0.82), incumbent, "logloss")
    assert promoted and "improves" in reason

    promoted, reason = promote.decide(candidate("2", 0.45, 0.90), incumbent, "logloss")
    assert not promoted and "0.4500" in reason and "0.4000" in reason

    promoted, _ = promote.decide(candidate("2", 0.40, 0.81), incumbent, "logloss")
    assert promoted, "a tie on log loss with a better area under the curve promotes"

    promoted, reason = promote.decide(candidate("2", 0.40, 0.79), incumbent, "logloss")
    assert not promoted and "ties" in reason

    promoted, _ = promote.decide(candidate("2", 0.40, 0.80), incumbent, "logloss")
    assert promoted, "a tie on both promotes the newer version"

    # The baseline veto comes before the comparison, and applies to the first
    # promotion too, when there is no incumbent to be better than.
    promoted, reason = promote.decide(candidate("2", 0.01, 0.99, beats=False), incumbent, "logloss")
    assert not promoted and "baseline" in reason
    promoted, reason = promote.decide(candidate("1", 0.01, 0.99, beats=False), None, "logloss")
    assert not promoted and "baseline" in reason
    promoted, _ = promote.decide(candidate("1", 0.60, 0.55), None, "logloss")
    assert promoted, "the first version to beat the baseline takes production"

    # A one-class holdout leaves area under the curve undefined, and an
    # undefined number is not a passing grade.
    promoted, reason = promote.decide(candidate("2", 0.35, float("nan")), incumbent, "auc")
    assert not promoted and "no holdout auc" in reason
