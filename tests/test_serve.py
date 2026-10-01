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

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Final

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipeline import serve
from pipeline.ml_features import MODEL_FEATURES, UNSEEN_CATEGORY
from pipeline.sql_gate import GATE_JEV, GATE_OFF, GATE_VAR

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


@pytest.fixture(autouse=True)
def no_agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither provider key, and no gate setting, unless a test asks for one.

    `/health` and `/ask` now read the environment to decide whether the agent
    could work at all, so a developer running the suite under `op run` would
    otherwise get different answers from the same tests than continuous
    integration does. Everything about the agent's environment is set inside
    the tests that are about it.
    """
    for name in (*serve.AGENT_KEY_VARS, GATE_VAR, "PRA_SQL_GATE_ON_ERROR"):
        monkeypatch.delenv(name, raising=False)


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
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_health_is_still_ok_with_nothing_promoted() -> None:
    """An empty registry is a state the service reports, not a reason to exit.

    This is the first-deploy case: the image is fine and no version holds the
    alias yet, and a process that refused to start would look like a broken
    build to whatever is restarting it.
    """

    def empty() -> serve.LoadedModel:
        raise RuntimeError("Registered model alias production not found.")

    with TestClient(serve.create_app(empty)) as started:
        assert started.get("/health").json()["model_loaded"] is False
        refused = started.post("/predict", json=BODY)
        assert refused.status_code == 503
        assert "production" in refused.json()["detail"]
        assert started.get("/model").status_code == 503


# ----------------------------------------------------- where the model loads --
#
# `eager_model` is the difference between the command line, which wants a bad
# registry to be a startup failure, and Lambda, where the build is the cold
# path of every route and loading the model means pulling the MLflow store out
# of the lake. `Registry.loads` counts, so these are assertions about when the
# work happens rather than about how long it takes.


def test_the_model_is_loaded_while_the_app_is_built_by_default(registry: Registry) -> None:
    """What `python -m pipeline.serve` and `compose.yaml` have always done."""
    serve.create_app(registry)
    assert registry.loads == 1


def test_health_does_not_load_the_model_on_a_lazy_host(registry: Registry) -> None:
    """The whole of the cold start fix: `/health` costs nothing a model costs.

    On the deployed function the load was 28.7 of the 29 seconds the first
    `/health` took, and `/health` does not read the model.
    """
    with TestClient(serve.create_app(registry, eager_model=False)) as started:
        assert registry.loads == 0
        body = started.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert registry.loads == 0


def test_a_lazy_model_is_loaded_by_the_first_route_that_reads_it(registry: Registry) -> None:
    """`/model`, `/predict` and `/reload` pay for it; once, between them."""
    with TestClient(serve.create_app(registry, eager_model=False)) as started:
        assert started.get("/model").status_code == 200
        assert registry.loads == 1
        assert started.post("/predict", json=BODY).status_code == 200
        assert started.get("/health").json()["model_loaded"] is True
        assert registry.loads == 1


def test_a_lazy_load_that_fails_is_a_503_and_not_a_retry_on_every_request() -> None:
    """An empty registry answers no, quickly, and `/reload` is the way to ask again."""
    attempts = 0

    def empty() -> serve.LoadedModel:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Registered model alias production not found.")

    with TestClient(serve.create_app(empty, eager_model=False)) as started:
        assert started.get("/health").json()["model_loaded"] is False
        assert started.get("/model").status_code == 503
        assert started.post("/predict", json=BODY).status_code == 503
        assert attempts == 1
        assert started.post("/reload").status_code == 503
        assert attempts == 2


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


def test_the_documented_contract_is_still_the_five_endpoints(client: TestClient) -> None:
    """`/metrics` is mounted but stays out of the schema, and nothing else moved.

    The telemetry endpoint is about the process, not about win probabilities, so
    a caller reading `/openapi.json` to generate a client should not find it.
    The five that are the contract have to still be there, which is the half of
    this that would catch instrumentation replacing a route by accident.
    """
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/model", "/reload", "/predict", "/ask"}
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


# ------------------------------------------------------------------ /ask --
#
# The route, its body and its failure mode, with a stub agent rather than a
# real one: `tests/test_agent.py` drives the real loop with a scripted chat
# model against a real warehouse, and the question here is only whether the
# endpoint is wired to it. That keeps this module free of LangChain, which is
# the same reason it is free of MLflow.


class StubAgent:
    """An agent that answers from a script and records what it was asked."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.asked: list[str] = []

    def ask(self, question: str) -> "StubAgent":
        self.asked.append(question)
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.payload


ANSWER: Final[dict[str, Any]] = {
    "answer": "Alpha wins 58% of 12 games against Beta.",
    "tool_calls": [
        {"tool": "query_marts", "input_summary": "select ... from mart_matchups", "rows": 1}
    ],
    "model": "scripted-fake",
    "usage": {"input_tokens": 120, "output_tokens": 40},
}


def test_ask_returns_the_answer_and_what_the_agent_read(registry: Registry) -> None:
    agent = StubAgent(ANSWER)
    app = serve.create_app(registry, agent_factory=lambda: agent)
    with TestClient(app) as started:
        body = started.post("/ask", json={"question": "how does Alpha do against Beta"}).json()

    assert body == ANSWER
    assert agent.asked == ["how does Alpha do against Beta"]


def test_ask_refuses_an_empty_question(registry: Registry) -> None:
    app = serve.create_app(registry, agent_factory=lambda: StubAgent(ANSWER))
    with TestClient(app) as started:
        assert started.post("/ask", json={"question": ""}).status_code == 422
        assert started.post("/ask", json={}).status_code == 422


def test_an_agent_that_cannot_be_built_is_a_503_that_says_why(registry: Registry) -> None:
    """A missing provider key must not stop `/predict` from working.

    The agent is built on the first question rather than at startup, so a
    service with no key serves predictions and refuses only the route that
    needs one.
    """

    def broken() -> serve.AskAgent:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    with TestClient(serve.create_app(registry, agent_factory=broken)) as started:
        assert started.post("/predict", json=BODY).status_code == 200
        refused = started.post("/ask", json={"question": "anything"})
        assert refused.status_code == 503
        assert "ANTHROPIC_API_KEY" in refused.json()["detail"]


def test_the_agent_is_built_once_and_reused(registry: Registry) -> None:
    builds = 0

    def factory() -> serve.AskAgent:
        nonlocal builds
        builds += 1
        return StubAgent(ANSWER)

    with TestClient(serve.create_app(registry, agent_factory=factory)) as started:
        started.post("/ask", json={"question": "one"})
        started.post("/ask", json={"question": "two"})
    assert builds == 1


def test_a_failed_agent_build_is_tried_again_on_the_next_question(
    registry: Registry, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure must not be cached, because on Lambda the cache is the container.

    The first deployed container answered every `/ask` for hours with the same
    `GateConfigError` in 17 ms, and the fix was one environment variable. A
    build is attempted again on the next question, so a corrected variable or
    a key that has arrived is live without a redeployment.
    """
    attempts = 0

    def flaky() -> serve.AskAgent:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("the warehouse was not there yet")
        return StubAgent(ANSWER)

    with TestClient(serve.create_app(registry, agent_factory=flaky)) as started:
        with caplog.at_level(logging.INFO):
            refused = started.post("/ask", json={"question": "one"})
            assert refused.status_code == 503
            assert started.get("/health").json()["agent_ready"] is False
            answered = started.post("/ask", json={"question": "two"})
        assert answered.status_code == 200
        assert answered.json() == ANSWER
        assert started.get("/health").json()["agent_ready"] is True
    assert attempts == 2
    assert "building the agent again" in caplog.text


# -------------------------------------------------- the degraded states --
#
# What the first deployed container actually did, as assertions. Both of these
# are a dependency that is not ready rather than a process that is unhealthy,
# so `/health` is 200 with the reason in a field and `/ask` is the route that
# refuses. The agent factory is the real one here, which is the point: these
# are about the environment the real one would read, and neither test gets far
# enough to import LangChain.


def test_no_provider_key_is_a_200_health_and_a_503_ask(registry: Registry) -> None:
    """Before the secret was filled, every route 502ed. This is what it does now."""
    with TestClient(serve.create_app(registry, eager_model=False)) as started:
        body = started.get("/health").json()
        assert started.get("/health").status_code == 200
        assert body["status"] == "ok"
        assert body["keys_loaded"] is False
        assert body["missing_keys"] == list(serve.AGENT_KEY_VARS)
        assert body["agent_ready"] is False
        assert body["agent_reason"] == serve.NO_PROVIDER_KEY

        refused = started.post("/ask", json={"question": "anything"})
        assert refused.status_code == 503
        assert refused.json()["detail"] == serve.NO_PROVIDER_KEY
        # And the half of the service that needs no key still works.
        assert started.post("/predict", json=BODY).status_code == 200


def test_one_key_of_the_two_is_named_on_health(
    registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the gate off the judge key is not needed, and `/health` still says it is unset."""
    monkeypatch.setenv(serve.PROVIDER_KEY_VAR, "not-a-real-key")
    monkeypatch.setenv(GATE_VAR, GATE_OFF)
    with TestClient(serve.create_app(registry, eager_model=False)) as started:
        body = started.get("/health").json()
    assert body["keys_loaded"] is False
    assert body["missing_keys"] == [serve.JUDGE_KEY_VAR]
    assert body["agent_ready"] is True
    assert body["agent_reason"] is None


def test_a_gate_value_the_gate_refuses_does_not_take_the_service_down(
    registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`PRA_SQL_GATE=1` was set on the function, and the accepted values are `off` and `jev`."""
    monkeypatch.setenv(serve.PROVIDER_KEY_VAR, "not-a-real-key")
    monkeypatch.setenv(serve.JUDGE_KEY_VAR, "not-a-real-key-either")
    monkeypatch.setenv(GATE_VAR, "1")
    with TestClient(serve.create_app(registry, eager_model=False)) as started:
        health = started.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["keys_loaded"] is True
        assert body["missing_keys"] == []
        assert body["agent_ready"] is False
        for expected in (GATE_VAR, GATE_OFF, GATE_JEV):
            assert expected in body["agent_reason"]

        refused = started.post("/ask", json={"question": "anything"})
        assert refused.status_code == 503
        assert refused.json()["detail"] == body["agent_reason"]
        assert started.post("/predict", json=BODY).status_code == 200


def test_a_good_gate_and_both_keys_report_ready(
    registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deployed configuration, once it is right: nothing to report."""
    monkeypatch.setenv(serve.PROVIDER_KEY_VAR, "not-a-real-key")
    monkeypatch.setenv(serve.JUDGE_KEY_VAR, "not-a-real-key-either")
    monkeypatch.setenv(GATE_VAR, GATE_JEV)
    with TestClient(serve.create_app(registry, eager_model=False)) as started:
        body = started.get("/health").json()
    assert (body["keys_loaded"], body["agent_ready"], body["agent_reason"]) == (True, True, None)


def test_the_judge_key_names_the_same_variable_the_gate_reads() -> None:
    """Two modules spell these out as strings; a rename has to reach both."""
    from pipeline import lambda_serve
    from pipeline.sql_gate import API_KEY_VAR as GATE_API_KEY_VAR

    assert serve.JUDGE_KEY_VAR == GATE_API_KEY_VAR
    assert sorted(serve.AGENT_KEY_VARS) == sorted(lambda_serve.SECRET_KEYS)
