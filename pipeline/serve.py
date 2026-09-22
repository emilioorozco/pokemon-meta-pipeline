"""Serving: the promoted model behind `POST /predict`, and nothing else.

One FastAPI application with four endpoints. `/predict` answers the question
the feature table was built to ask, "this side, this board, this far in, who
wins?", `/health` says whether a model is loaded, `/model` says which one, and
`/reload` picks up a promotion without a restart. A fifth, `/metrics`, is the
Prometheus exposition and belongs to the process rather than to the model; it
is mounted by `pipeline.telemetry`, which also traces the requests.

Four choices worth knowing before reading the code.

It loads by alias, never by version. The address is
`models:/win-probability@production`, so promoting a new version and restarting
nothing is the deployment: `python -m pipeline.promote` moves the alias and
`POST /reload` picks it up. A service pinned to version 7 means every promotion
is also a code change, which is how a registry ends up with a `production` alias
that nothing actually serves.

A missing model is a state, not a crash. If no version holds the alias yet, the
application still starts and `/health` still answers, with `model_loaded: false`
and a 503 from `/predict`. A container that exits because the registry is empty
tells an orchestrator that the image is broken, which is the wrong problem: the
image is fine and the registry is empty, and the difference is visible only if
the process stays up long enough to say so.

The features come from `pipeline.ml_features`, which is also where the trainer
gets them. The request model is written out field by field, because these
descriptions are the OpenAPI documentation and a loop over a tuple would
produce a schema with sixteen identical ones, but a test asserts the two lists
match, so they cannot drift apart silently.

An archetype the model never trained on is answered, not refused. It encodes to
the same -1 the training code uses for an unseen category and comes back named
in `unknown_archetypes`, because the useful reply to "Charizard against
something I have never heard of" is a probability that leans on the other
fifteen features plus a note that half the matchup is guesswork. A 422 would
make the caller handle a metagame that moves every set release as an error.

A fifth endpoint, `POST /ask`, is the agent. It is here rather than in a second
service because it answers over the same warehouse, with the same traces and
the same Prometheus registry, and a second process would double the deployment
for one route. Its agent is injected exactly as the model loader is, so the
tests drive the real agent loop with a scripted chat model and no API key, and
it is built on the first question rather than at startup: a service whose
`/predict` works should not fail to start because a provider key is missing.
"""

import argparse
import contextlib
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from opentelemetry.sdk.trace.export import SpanExporter
from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import (
    CARD_INDEX_DIR,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    WAREHOUSE_PATH,
    default_tracking_uri,
)
from pipeline.ml_features import CATEGORICAL, MODEL_FEATURES, ArchetypeCodes, design_matrix
from pipeline.observability import configure_logging
from pipeline.telemetry import INFERENCE_SPAN, route_label, setup_metrics, setup_tracing

logger = logging.getLogger(__name__)

STAGE: Final = "serve"
CODES_ARTIFACT: Final = "archetype_codes.json"
# Demonstrations only: set it and the service answers from the hand-written
# logistic in `StubPredictor` instead of the registry. See `stub_loader`.
STUB_MODEL_VAR: Final = "PRA_SERVE_STUB_MODEL"
STUB_VERSION: Final = "stub"
# Version tags that are numbers worth reporting on `/model`. The rest of the
# tags are dates and decisions, which belong in the registry rather than in a
# health check.
METRIC_TAGS: Final[tuple[str, ...]] = ("holdout_logloss", "holdout_auc", "beats_baseline")


class Predictor(Protocol):
    """What serving needs from a model: one number per row.

    A LightGBM `Booster` satisfies this with `predict`; anything scikit-learn
    shaped offers `predict_proba` instead, and `probability` below takes either.
    Typing it as a protocol rather than as `lgb.Booster` is what lets the tests
    inject a stub and lets a future model be something else.
    """

    def predict(self, data: pd.DataFrame) -> Any:
        """Scores for each row of the design matrix."""


@dataclass(frozen=True)
class LoadedModel:
    """A model and everything a caller needs to know about which model it is."""

    predictor: Predictor
    codes: ArchetypeCodes
    name: str
    version: str
    alias: str
    metrics: dict[str, float]
    loaded_at: datetime


Loader = Callable[[], LoadedModel]


class PredictRequest(BaseModel):
    """The board at the start of a turn, from one seat's point of view.

    Every field is a column of `features_turn`, cumulative over that seat's
    turns strictly before this one, which is the same cut-off the training rows
    were built with. Sending totals that include the current turn is not an
    error the service can detect, and it is the one way to get a confidently
    wrong answer out of it.
    """

    turn_number: int = Field(
        ge=1, description="Turn this prediction is for; the state is the start of it"
    )
    went_first: bool = Field(description="This seat took the first turn of the game")
    archetype_key: str = Field(
        min_length=1,
        description="This seat's deck, as the shared archetype id or `name:<canonical name>`",
    )
    opponent_archetype_key: str = Field(
        min_length=1, description="The other seat's deck, keyed the same way"
    )
    prizes_taken_self: int = Field(ge=0, le=6, description="Prizes this seat has taken so far")
    prizes_taken_opp: int = Field(ge=0, le=6, description="Prizes the other seat has taken so far")
    knockouts_self: int = Field(ge=0, description="Knockouts this seat has scored so far")
    knockouts_opp: int = Field(ge=0, description="Knockouts the other seat has scored so far")
    cards_drawn_self: int = Field(
        ge=0, description="Cards this seat has drawn, opening hand and mulligans included"
    )
    energy_attached_self: int = Field(ge=0, description="Energy this seat has attached so far")
    pokemon_played_self: int = Field(
        ge=0, description="Pokemon this seat has put into play so far; not a bench count"
    )
    trainers_played_self: int = Field(
        ge=0, description="Trainers and stadiums this seat has played so far"
    )
    evolutions_self: int = Field(ge=0, description="Evolutions this seat has made so far")
    attacks_self: int = Field(ge=0, description="Attacks this seat has made so far")
    turns_played_self: int = Field(ge=0, description="This seat's own turns before this one")
    prize_diff: int | None = Field(
        default=None,
        description="Prize lead; computed as taken_self minus taken_opp when it is left out",
    )

    def features(self) -> dict[str, Any]:
        """The payload as the model's feature columns, with `prize_diff` filled in."""
        values = self.model_dump()
        if values["prize_diff"] is None:
            values["prize_diff"] = values["prizes_taken_self"] - values["prizes_taken_opp"]
        return {name: values[name] for name in MODEL_FEATURES}


class PredictResponse(BaseModel):
    """One probability, and enough provenance to reproduce it."""

    # `model_` is a Pydantic protected prefix, and these three fields are the
    # answer to "which model said this", which is the name a caller expects.
    model_config = ConfigDict(protected_namespaces=())

    win_probability: float = Field(
        description="Probability this seat wins, from the model holding the loaded alias"
    )
    model_name: str = Field(description="Registered model the prediction came from")
    model_version: str = Field(description="Registered version, so a prediction can be traced back")
    model_alias: str = Field(description="Alias that resolved to that version, normally production")
    features_used: list[str] = Field(description="Feature columns in the order the model saw them")
    unknown_archetypes: list[str] = Field(
        description="Archetypes the model never trained on; each encoded as a missing category"
    )


class AgentResult(Protocol):
    """What the agent hands back: enough to build the response body from.

    A protocol over `as_dict` rather than the dataclass itself, so importing
    this module never imports LangChain. The serving container installs the
    `ml` extra and nothing else, and `/predict` has to keep working in it.
    """

    def as_dict(self) -> dict[str, Any]:
        """The answer, its tool calls, the model and the token usage."""


class AskAgent(Protocol):
    """A built agent: one question in, one answer out, no state between them."""

    def ask(self, question: str) -> AgentResult:
        """Answer one question."""


AgentFactory = Callable[[], AskAgent]


class AskRequest(BaseModel):
    """One question in natural language."""

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="What to ask about the metagame, in plain English",
    )


class ToolCallResponse(BaseModel):
    """One tool the agent called while answering, and what came back."""

    tool: str = Field(description="Tool name, `query_marts` or `lookup_cards`")
    input_summary: str = Field(description="The query it was given, shortened to one line")
    rows: int = Field(description="Rows the tool returned; zero for a refusal or an empty result")


class AskResponse(BaseModel):
    """The agent's answer, and everything it did to get there."""

    model_config = ConfigDict(protected_namespaces=())

    answer: str = Field(description="The answer, in prose, citing the sample size it read")
    tool_calls: list[ToolCallResponse] = Field(
        description="Every tool call of this run, in order. An empty list means the model "
        "answered without reading the warehouse, which its prompt tells it not to do"
    )
    model: str = Field(description="Provider model that answered")
    usage: dict[str, int] = Field(
        description="Token counts the provider reported; empty when it reported none"
    )


class HealthResponse(BaseModel):
    """Liveness, and whether the alias resolved to anything."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="Always `ok` while the process is answering")
    model_loaded: bool = Field(description="False when no version holds the alias yet")


class ModelResponse(BaseModel):
    """Which model is loaded, and how it scored on the holdout it was promoted on."""

    name: str = Field(description="Registered model name")
    version: str = Field(description="Registered version currently loaded")
    alias: str = Field(description="Alias this version was loaded through")
    loaded_at: datetime = Field(description="When this process loaded it")
    metrics: dict[str, float] = Field(
        description="Holdout numbers from the version's tags: log loss, area under the curve, "
        "and whether it beat the win-rate baseline"
    )


def probability(predictor: Predictor, matrix: pd.DataFrame) -> float:
    """One win probability out of whatever shape the model returns.

    LightGBM's binary booster returns the positive-class probability directly;
    a scikit-learn style estimator returns a column per class. Both are read
    here rather than at the loader, so a stub in a test and the real booster
    take the same path through the endpoint.
    """
    proba = getattr(predictor, "predict_proba", None)
    if callable(proba):
        scores = proba(matrix)
        return float(scores[0][1])
    return float(predictor.predict(matrix)[0])


def unknown_archetypes(payload: dict[str, Any], codes: ArchetypeCodes) -> list[str]:
    """The archetype keys in this request that the model was never trained on."""
    unseen: list[str] = []
    for column in CATEGORICAL:
        value = str(payload[column])
        if value not in codes.get(column, {}) and value not in unseen:
            unseen.append(value)
    return unseen


class ModelHolder:
    """The loaded model, and the loader that can replace it.

    A small mutable box rather than a module global, so two applications in one
    test process do not share a model, and so `/reload` is an assignment rather
    than a re-import.
    """

    def __init__(self, loader: Loader) -> None:
        self.loader = loader
        self.current: LoadedModel | None = None
        self.error: str | None = None

    def load(self) -> LoadedModel:
        """Load through the loader, recording the failure rather than raising it."""
        try:
            self.current = self.loader()
            self.error = None
        except Exception as failure:
            self.current = None
            self.error = f"{type(failure).__name__}: {failure}"
            raise
        return self.current

    def try_load(self) -> None:
        """Load if possible. Used at startup, where an empty registry is not an error."""
        # `self.error` already holds the reason, and `/health` is how it is read.
        with contextlib.suppress(Exception):
            self.load()

    def required(self) -> LoadedModel:
        """The loaded model, or a 503 that says what went wrong instead."""
        if self.current is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"no model loaded: {self.error or 'nothing has been loaded yet'}. "
                    f"Register a version with `python -m pipeline.train` and give it the "
                    f"{PRODUCTION_ALIAS} alias with `python -m pipeline.promote`."
                ),
            )
        return self.current


class AgentHolder:
    """The agent, built on the first question and kept afterwards.

    Lazily, because building it constructs a provider client that wants a key,
    and a service that refused to start without one would make the agent a
    dependency of `/predict`. The failure is recorded and returned as a 503
    with its reason, and the next request tries again, so a key arriving later
    fixes the endpoint without a restart.
    """

    def __init__(self, factory: AgentFactory) -> None:
        self.factory = factory
        self.current: AskAgent | None = None

    def required(self) -> AskAgent:
        """The agent, or a 503 naming what went wrong building it."""
        if self.current is not None:
            return self.current
        try:
            self.current = self.factory()
        except Exception as failure:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"the agent is unavailable: {type(failure).__name__}: {failure}. "
                    f"It needs a provider key in the environment and a warehouse built "
                    f"by `python -m pipeline.gold`."
                ),
            ) from failure
        return self.current


def default_agent_factory(*, tracer: Any, metrics: Any) -> AgentFactory:
    """An agent over the real warehouse, sharing this process's tracer and registry.

    `pipeline.agent` is imported inside the closure for the same reason MLflow
    is imported inside `mlflow_loader`: the application has to be importable,
    and every other endpoint testable, without LangChain installed.
    """

    def build() -> AskAgent:
        from pipeline.agent import build_agent, default_card_index

        return build_agent(
            warehouse=WAREHOUSE_PATH,
            card_index=default_card_index(CARD_INDEX_DIR),
            tracer=tracer,
            metrics=metrics,
        )

    return build


def mlflow_loader(*, tracking_uri: str | None = None, alias: str = PRODUCTION_ALIAS) -> Loader:
    """A loader that reads whichever version holds the alias, with its archetype codes.

    MLflow is imported inside the closure rather than at module scope, which is
    what lets `create_app` be imported, and the endpoints tested, with a stub
    model and no MLflow installed or running.
    """

    def load() -> LoadedModel:
        import mlflow
        import mlflow.lightgbm
        from mlflow.artifacts import download_artifacts
        from mlflow.tracking import MlflowClient

        uri = tracking_uri or default_tracking_uri()
        if uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        model = mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
        # The codes travel with the run rather than with the model, because they
        # are how a request is turned into the integers the model was fitted on.
        # A model loaded without them would still predict, on the wrong numbers.
        local = download_artifacts(run_id=version.run_id, artifact_path=CODES_ARTIFACT)
        with open(local) as handle:
            codes: ArchetypeCodes = json.load(handle)
        metrics: dict[str, float] = {}
        for tag in METRIC_TAGS:
            raw = version.tags.get(tag)
            if raw is None:
                continue
            try:
                metrics[tag] = float(raw)
            except ValueError:
                continue
        return LoadedModel(
            predictor=model,
            codes=codes,
            name=REGISTERED_MODEL_NAME,
            version=str(version.version),
            alias=alias,
            metrics=metrics,
            loaded_at=datetime.now(UTC),
        )

    return load


class StubPredictor:
    """A hand-written logistic on the prize lead. Demonstrations only, never a model.

    It exists because the observability stack has to be demonstrable on a laptop
    whose registry holds nothing servable, and on this corpus that is the normal
    state: `promote` refuses a candidate that does not beat the archetype
    win-rate baseline, and on 66 feature rows nothing does. Without this, showing
    a trace with an inference span in it would mean either a registry fixture
    nobody maintains or bypassing the promotion gate by hand, and the second one
    is much worse than an obviously fake predictor.

    The shape is the one thing about it that is real: it is a `Predictor`, it
    takes the design matrix the endpoint built and returns one number per row, so
    the code path under the span, the histogram and the counter is the same code
    path the LightGBM booster takes.
    """

    def predict(self, data: pd.DataFrame) -> list[float]:
        """A logistic on the prize lead, so the answers vary with the board."""
        return [1.0 / (1.0 + math.exp(-0.7 * float(lead))) for lead in data["prize_diff"]]


def stub_loader() -> Loader:
    """A loader that returns the stub above, for a demonstration with no registry.

    Reached only through `PRA_SERVE_STUB_MODEL`, and it says so in the version,
    the alias and a warning on startup: a `/predict` answered by this must be
    impossible to mistake for one answered by a model, in the response body, in
    the log and on the metric labels.

    The archetype code map is empty on purpose. Nothing trained it, so every
    archetype is one the model has never seen, and every reply names both of
    them in `unknown_archetypes` rather than implying knowledge it does not have.
    """

    def load() -> LoadedModel:
        logger.warning(
            "serving a stub predictor, not a model",
            extra={"reason": f"{STUB_MODEL_VAR} is set", "model_version": STUB_VERSION},
        )
        return LoadedModel(
            predictor=StubPredictor(),
            codes={column: {} for column in CATEGORICAL},
            name=REGISTERED_MODEL_NAME,
            version=STUB_VERSION,
            alias=STUB_VERSION,
            metrics={},
            loaded_at=datetime.now(UTC),
        )

    return load


def stub_requested() -> bool:
    """Whether `PRA_SERVE_STUB_MODEL` asks for the stub rather than the registry."""
    return os.environ.get(STUB_MODEL_VAR, "").strip().lower() in {"1", "true", "yes"}


def describe(model: LoadedModel) -> ModelResponse:
    """The loaded model as the `/model` body."""
    return ModelResponse(
        name=model.name,
        version=model.version,
        alias=model.alias,
        loaded_at=model.loaded_at,
        metrics=model.metrics,
    )


def create_app(
    loader: Loader | None = None,
    *,
    span_exporter: SpanExporter | None = None,
    agent_factory: AgentFactory | None = None,
) -> FastAPI:
    """The application, with its model loader and its agent injected.

    The loader is a plain callable returning a `LoadedModel`, so a test passes a
    stub and the command line passes `mlflow_loader()`. Nothing below this line
    knows that MLflow exists.

    `agent_factory` is the same shape one level up: a callable returning
    something that answers `ask`, so a test passes an agent built around a
    scripted chat model and the command line passes the real one. Nothing below
    this line knows that LangChain exists either.

    `span_exporter` is the same idea for traces: the telemetry tests pass an
    in-memory exporter, and everything else leaves it out and lets
    `OTEL_EXPORTER_OTLP_ENDPOINT` decide whether there is a collector at all.
    """
    holder = ModelHolder(loader or mlflow_loader())
    app = FastAPI(
        title="Pokemon win-probability service",
        version="1.0.0",
        description=(
            "Per-turn win probability from the LightGBM model holding the "
            f"`{PRODUCTION_ALIAS}` alias of the `{REGISTERED_MODEL_NAME}` registered model."
        ),
    )
    tracer = setup_tracing(app, exporter=span_exporter)
    metrics = setup_metrics(app)
    agent = AgentHolder(agent_factory or default_agent_factory(tracer=tracer, metrics=metrics))

    def record_load() -> None:
        """Put the loaded version on the `model_info` gauge, or clear it if none."""
        if holder.current is None:
            metrics.model_info.clear()
            return
        metrics.set_model_info(holder.current.name, holder.current.version, holder.current.alias)

    # At import rather than on first request: a service that loads lazily reports
    # healthy until someone asks it a question, which is the wrong time to find
    # out the registry is empty.
    holder.try_load()
    record_load()

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """One record per call: what was asked, what came back, how long, and by which model.

        A long-running service writes no `run_metrics` row, because there is no
        run to close: the unit here is the request, and the request log is what
        a latency or an error-rate panel is built from. `model_version` is on
        every line so a shifted prediction distribution can be attributed to a
        promotion rather than guessed at.

        The same timing feeds the Prometheus counter and histogram, from here
        rather than from a second middleware: two middlewares would time two
        slightly different things and disagree about latency by however much
        code sits between them, and the one that is wrong would be whichever a
        reader was not looking at.
        """
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            metrics.observe_request(request.method, route_label(request), 500, elapsed)
            logger.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": 500,
                    "duration_ms": round(elapsed * 1000, 3),
                    "model_version": holder.current.version if holder.current else None,
                },
            )
            raise
        elapsed = time.perf_counter() - started
        metrics.observe_request(request.method, route_label(request), response.status_code, elapsed)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000, 3),
                "model_version": holder.current.version if holder.current else None,
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse, summary="Liveness and model state")
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=holder.current is not None)

    @app.get("/model", response_model=ModelResponse, summary="Which model is loaded")
    def model() -> ModelResponse:
        return describe(holder.required())

    @app.post("/reload", response_model=ModelResponse, summary="Pick up a new promotion")
    def reload() -> ModelResponse:
        try:
            holder.load()
        except Exception as failure:
            record_load()
            raise HTTPException(
                status_code=503, detail=f"reload failed: {type(failure).__name__}: {failure}"
            ) from failure
        record_load()
        return describe(holder.required())

    @app.post("/predict", response_model=PredictResponse, summary="Win probability for one turn")
    def predict(request: PredictRequest) -> PredictResponse:
        loaded = holder.required()
        payload = request.features()
        unseen = unknown_archetypes(payload, loaded.codes)
        matrix = design_matrix(pd.DataFrame([payload]), loaded.codes)
        # The span is around the model call and nothing else. A span covering
        # the whole handler would answer "is /predict slow", which the HTTP span
        # the instrumentation already emits answers; the question this one is
        # for is whether the time is in the model or around it.
        with tracer.start_as_current_span(INFERENCE_SPAN) as span:
            span.set_attribute("model.version", loaded.version)
            span.set_attribute("model.alias", loaded.alias)
            span.set_attribute("features.unknown_archetypes", len(unseen))
            with metrics.time_inference(loaded.version):
                score = probability(loaded.predictor, matrix)
        metrics.count_prediction(loaded.version, unknown=bool(unseen))
        return PredictResponse(
            win_probability=score,
            model_name=loaded.name,
            model_version=loaded.version,
            model_alias=loaded.alias,
            features_used=list(MODEL_FEATURES),
            unknown_archetypes=unseen,
        )

    @app.post("/ask", response_model=AskResponse, summary="Ask the agent about the metagame")
    def ask(request: AskRequest) -> AskResponse:
        """One question through the agent loop, and what it read to answer it.

        No span is opened here: the agent opens `agent.answer` around its own
        run and a span per tool call inside it, and the HTTP span from the
        instrumentation is already the parent of all of them.
        """
        result = agent.required().ask(request.question)
        return AskResponse.model_validate(result.as_dict())

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the service with uvicorn from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.serve",
        description="Serve the promoted win-probability model over HTTP.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: loopback)")
    parser.add_argument("--port", type=int, default=8000, help="port to bind (default: 8000)")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI, else file:./data/mlruns)",
    )
    parser.add_argument(
        "--alias",
        default=PRODUCTION_ALIAS,
        help=f"registered model alias to load (default: {PRODUCTION_ALIAS})",
    )
    args = parser.parse_args(argv)
    configure_logging(STAGE)

    import uvicorn

    loader = (
        stub_loader()
        if stub_requested()
        else mlflow_loader(tracking_uri=args.tracking_uri, alias=args.alias)
    )
    app = create_app(loader)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
