"""The serving stage end to end, with a stub model and no MLflow anywhere.

These tests are in the fast suite on purpose. The endpoints are the part that
breaks: a feature renamed in one place, a probability read out of the wrong
column, an unknown archetype turned into a 500. None of that needs a trained
model to catch, and a suite that needs a registry, a tracking server and a
LightGBM build to answer "does /predict return a number" is a suite nobody runs
before pushing.

So the loader is injected. `create_app` takes a callable returning a
`LoadedModel`, the stub below returns one built around a predictor that records
what it was asked, and the whole MLflow path is exercised once, for real, by
the smoke run in the README rather than by mocks that would only assert that
the mocks were called. `tests/test_train.py` and `tests/test_promote.py` cover
the registry side against a real file store.

The stub records its design matrix, which is what lets these tests assert the
two things a mocked model would hide: that `prize_diff` is computed when the
request leaves it out, and that an archetype the model never saw arrives as the
missing-category code rather than as a string.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Final

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipeline import serve
from pipeline.ml_features import MODEL_FEATURES, UNSEEN_CATEGORY

CODES: Final[dict[str, dict[str, int]]] = {
    "archetype_key": {"name:alpha": 0, "name:beta": 1},
    "opponent_archetype_key": {"name:alpha": 0, "name:beta": 1},
}

# A plausible mid-game board: this seat is one prize ahead on turn eight.
BODY: Final[dict[str, Any]] = {
    "turn_number": 8,
    "went_first": True,
    "archetype_key": "name:alpha",
    "opponent_archetype_key": "name:beta",
    "prizes_taken_self": 3,
    "prizes_taken_opp": 2,
    "knockouts_self": 3,
    "knockouts_opp": 2,
    "cards_drawn_self": 24,
    "energy_attached_self": 4,
    "pokemon_played_self": 5,
    "trainers_played_self": 14,
    "evolutions_self": 2,
    "attacks_self": 3,
    "turns_played_self": 4,
}


class StubPredictor:
    """A model that returns a fixed probability and remembers the matrix it was given."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, data: pd.DataFrame) -> list[list[float]]:
        self.seen = data
        return [[1.0 - self.score, self.score] for _ in range(len(data))]


class StubBooster:
    """The other shape a model can have: one probability per row from `predict`."""

    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, data: pd.DataFrame) -> list[float]:
        return [self.score] * len(data)


def loaded(predictor: Any, version: str, logloss: float = 0.42) -> serve.LoadedModel:
    """A `LoadedModel` around a stub, as the real loader would return one."""
    return serve.LoadedModel(
        predictor=predictor,
        codes=CODES,
        name="win-probability",
        version=version,
        alias="production",
        metrics={"holdout_logloss": logloss, "holdout_auc": 0.71, "beats_baseline": 1.0},
        loaded_at=datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
    )


class Registry:
    """A swappable loader, so `/reload` has something new to pick up."""

    def __init__(self, model: serve.LoadedModel) -> None:
        self.model = model
        self.loads = 0

    def __call__(self) -> serve.LoadedModel:
        self.loads += 1
        return self.model


@pytest.fixture
def predictor() -> StubPredictor:
    return StubPredictor(0.73)


@pytest.fixture
def registry(predictor: StubPredictor) -> Registry:
    return Registry(loaded(predictor, "1"))


@pytest.fixture
def client(registry: Registry) -> Iterator[TestClient]:
    with TestClient(serve.create_app(registry)) as started:
        yield started


def test_the_request_model_mirrors_the_feature_list() -> None:
    """The one assertion that keeps the HTTP body and the design matrix in step.

    Not a tautology: the request model is written out field by field so each one
    carries a description into the OpenAPI schema, so it is exactly the kind of
    list that drifts when a feature is added to the trainer alone.
    """
    assert set(serve.PredictRequest.model_fields) == set(MODEL_FEATURES)
    for name, field in serve.PredictRequest.model_fields.items():
        assert field.description, f"{name} has no description for the generated docs"


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_is_still_ok_with_nothing_promoted() -> None:
    """An empty registry is a state the service reports, not a reason to exit.

    This is the first-deploy case: the image is fine and no version holds the
    alias yet, and a process that refused to start would look like a broken
    build to whatever is restarting it.
    """

    def empty() -> serve.LoadedModel:
        raise RuntimeError("Registered model alias production not found.")

    with TestClient(serve.create_app(empty)) as started:
        assert started.get("/health").json() == {"status": "ok", "model_loaded": False}
        refused = started.post("/predict", json=BODY)
        assert refused.status_code == 503
        assert "production" in refused.json()["detail"]
        assert started.get("/model").status_code == 503


def test_model_reports_the_version_and_its_holdout_numbers(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["name"] == "win-probability"
    assert body["version"] == "1"
    assert body["alias"] == "production"
    assert body["loaded_at"].startswith("2026-09-22T12:00:00")
    assert body["metrics"]["holdout_logloss"] == pytest.approx(0.42)
    assert body["metrics"]["beats_baseline"] == 1.0


def test_predict_returns_a_probability_with_its_provenance(
    client: TestClient, predictor: StubPredictor
) -> None:
    body = client.post("/predict", json=BODY).json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["win_probability"] == pytest.approx(0.73)
    assert (body["model_name"], body["model_version"], body["model_alias"]) == (
        "win-probability",
        "1",
        "production",
    )
    assert body["features_used"] == list(MODEL_FEATURES)
    assert body["unknown_archetypes"] == []

    seen = predictor.seen
    assert seen is not None
    assert list(seen.columns) == list(MODEL_FEATURES)
    # The body left `prize_diff` out, so the service computed it.
    assert seen["prize_diff"].iloc[0] == 1
    # And both archetypes were encoded rather than passed through as strings.
    assert seen["archetype_key"].iloc[0] == 0
    assert seen["opponent_archetype_key"].iloc[0] == 1


def test_predict_takes_a_prize_diff_that_is_sent(
    client: TestClient, predictor: StubPredictor
) -> None:
    client.post("/predict", json={**BODY, "prize_diff": -2})
    seen = predictor.seen
    assert seen is not None
    assert seen["prize_diff"].iloc[0] == -2


def test_predict_answers_for_an_unknown_archetype_and_names_it(
    client: TestClient, predictor: StubPredictor
) -> None:
    """A deck from a set released after training is a note, not a 422.

    The metagame moves every set release, so refusing the request would make
    every caller handle the ordinary case as an error. The model gets the same
    missing-category code it was trained to expect, and the caller is told which
    half of the matchup the answer is guessing at.
    """
    body = client.post("/predict", json={**BODY, "opponent_archetype_key": "name:omega"}).json()
    assert body["unknown_archetypes"] == ["name:omega"]
    assert 0.0 <= body["win_probability"] <= 1.0
    seen = predictor.seen
    assert seen is not None
    assert seen["opponent_archetype_key"].iloc[0] == UNSEEN_CATEGORY


def test_a_malformed_body_is_a_422(client: TestClient) -> None:
    assert client.post("/predict", json={"turn_number": 3}).status_code == 422
    assert client.post("/predict", json={**BODY, "turn_number": 0}).status_code == 422
    assert client.post("/predict", json={**BODY, "prizes_taken_self": "three"}).status_code == 422
    assert client.post("/predict", json={**BODY, "archetype_key": ""}).status_code == 422


def test_reload_picks_up_the_next_promotion(client: TestClient, registry: Registry) -> None:
    """What a promotion looks like from the service: same URL, different version."""
    assert client.get("/model").json()["version"] == "1"
    registry.model = loaded(StubPredictor(0.19), "2", logloss=0.31)

    reloaded = client.post("/reload")
    assert reloaded.status_code == 200
    assert reloaded.json()["version"] == "2"

    after = client.get("/model").json()
    assert after["version"] == "2"
    assert after["metrics"]["holdout_logloss"] == pytest.approx(0.31)
    assert client.post("/predict", json=BODY).json()["win_probability"] == pytest.approx(0.19)


def test_reload_that_fails_is_a_503_and_leaves_nothing_loaded() -> None:
    def broken() -> serve.LoadedModel:
        raise RuntimeError("registry is down")

    with TestClient(serve.create_app(broken)) as started:
        failed = started.post("/reload")
        assert failed.status_code == 503
        assert "registry is down" in failed.json()["detail"]
        assert started.get("/health").json()["model_loaded"] is False


def test_the_documented_contract_is_still_the_four_model_endpoints(client: TestClient) -> None:
    """`/metrics` is mounted but stays out of the schema, and nothing else moved.

    The telemetry endpoint is about the process, not about win probabilities, so
    a caller reading `/openapi.json` to generate a client should not find it.
    The four that are the contract have to still be there, which is the half of
    this that would catch instrumentation replacing a route by accident.
    """
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/model", "/reload", "/predict"}
    assert client.get("/metrics").status_code == 200


def test_the_request_log_still_carries_the_method_path_and_model_version(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The log line survived the metrics moving into the same middleware."""
    with caplog.at_level("INFO", logger="pipeline.serve"):
        client.post("/predict", json=BODY)
    record = next(entry for entry in caplog.records if entry.message == "request")
    assert (record.method, record.path, record.status) == ("POST", "/predict", 200)  # type: ignore[attr-defined]
    assert record.model_version == "1"  # type: ignore[attr-defined]
    assert record.duration_ms >= 0  # type: ignore[attr-defined]


def test_the_demo_stub_answers_and_never_pretends_to_be_a_model() -> None:
    """`PRA_SERVE_STUB_MODEL` is for demonstrations, and has to look like one.

    What is asserted is not the arithmetic, which is a logistic on the prize
    lead and means nothing. It is that the version is `stub` everywhere a
    version is reported, so a screenshot of a dashboard or a trace taken during
    a demonstration cannot be mistaken for one of a promoted model, and that
    every archetype comes back as unseen, because nothing trained it.
    """
    with TestClient(serve.create_app(serve.stub_loader())) as started:
        body = started.post("/predict", json=BODY).json()
        assert body["model_version"] == "stub"
        assert body["model_alias"] == "stub"
        assert sorted(body["unknown_archetypes"]) == ["name:alpha", "name:beta"]
        # Behind by two prizes is under a half, ahead by two is over it.
        behind = started.post("/predict", json={**BODY, "prize_diff": -2}).json()
        ahead = started.post("/predict", json={**BODY, "prize_diff": 2}).json()
        assert behind["win_probability"] < 0.5 < ahead["win_probability"]


def test_the_stub_is_off_unless_it_is_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(serve.STUB_MODEL_VAR, raising=False)
    assert serve.stub_requested() is False
    monkeypatch.setenv(serve.STUB_MODEL_VAR, "0")
    assert serve.stub_requested() is False
    monkeypatch.setenv(serve.STUB_MODEL_VAR, "1")
    assert serve.stub_requested() is True


def test_a_booster_shaped_model_is_read_the_other_way() -> None:
    """LightGBM returns one probability per row; scikit-learn returns a column per class."""
    with TestClient(serve.create_app(Registry(loaded(StubBooster(0.61), "5")))) as started:
        assert started.post("/predict", json=BODY).json()["win_probability"] == pytest.approx(0.61)
