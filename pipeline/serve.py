"""Serving: the promoted model behind `POST /predict`, and nothing else.

One FastAPI application. `/predict` answers the question the feature table was
built to ask, "this side, this board, this far in, who wins?", `/health` says
which of the service's dependencies are there, `/warm` makes the expensive
ones resident before a reader waits on them, `/model` says which model is
loaded, and `/reload` picks up a promotion without a restart. `POST /ask` is
the agent, and the last two paragraphs below are about those two. `/metrics`
is the Prometheus exposition and belongs to the process rather than to the
model; it is mounted by `pipeline.telemetry`, which also traces the requests.

Five choices worth knowing before reading the code.

It loads by alias, never by version. The address is
`models:/win-probability@production`, so promoting a new version and restarting
nothing is the deployment: `python -m pipeline.promote` moves the alias and
`POST /reload` picks it up. A service pinned to version 7 means every promotion
is also a code change, which is how a registry ends up with a `production` alias
that nothing actually serves.

A missing dependency is a state, not a crash. If no version holds the alias
yet, the application still starts and `/health` still answers, with
`model_loaded: false` and a 503 from `/predict`. A container that exits because
the registry is empty tells an orchestrator that the image is broken, which is
the wrong problem: the image is fine and the registry is empty, and the
difference is visible only if the process stays up long enough to say so. The
same holds for the agent's two provider keys and for the SQL gate's setting:
`/health` answers 200 with `keys_loaded` and `agent_ready` false and the reason
spelled out, and `/ask` is the route that refuses. A health check that went red
because a key had not been pasted in would make a running function look like a
broken image.

Where the model is loaded is a parameter, because the two hosts want opposite
answers. `python -m pipeline.serve` and `compose.yaml` load it while the app is
being built, so a registry that cannot be read is a startup failure and not a
surprise on the first prediction. `pipeline.lambda_serve` passes
`eager_model=False`, because on Lambda the build is the cold path of every
route and loading the model means pulling the MLflow file store out of the lake
one object at a time; on the deployed function that was 28.7 of the 29 seconds
the first `/health` took, for a model `/health` never looks at. Lazily, the
three routes that read the model are the three that wait for it.

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

A sixth, `GET /warm`, is the other side of that laziness. Everything the first
question pays for is paid once per container, so something has to ask for it
when no reader is waiting: `/warm` builds the agent and runs one embedding
through the card tool's model, calls no provider, and answers 200 whatever
happens. It is what the keepalive ping calls, because a ping to `/health` kept
a container alive with nothing in it and the next real question still paid for
all of it.
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
from pipeline.sql_gate import GateConfigError, gate_from_env
from pipeline.storage import tracking_store
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
# The two keys `/ask` needs in the environment: the agent's provider and the
# SQL gate's judge. Written out as strings rather than imported from
# `pipeline.agent`, which would import LangChain into a module that has to stay
# importable without it; `pipeline.lambda_serve` names the same two for the
# same reason, and the tests assert that all three agree.
PROVIDER_KEY_VAR: Final = "ANTHROPIC_API_KEY"
JUDGE_KEY_VAR: Final = "JEV_API_KEY"
AGENT_KEY_VARS: Final[tuple[str, ...]] = (PROVIDER_KEY_VAR, JUDGE_KEY_VAR)
NO_PROVIDER_KEY: Final = f"the agent has no provider key configured; ${PROVIDER_KEY_VAR} is not set"


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


class WarmableAgent(Protocol):
    """An agent that can make its expensive parts resident without being asked anything.

    Optional, and read off a built agent with `getattr` rather than required
    of every `AskAgent`: the tests inject agents that answer from a script and
    have nothing to warm, and `/warm` has to work with those too.
    """

    def warm(self) -> bool:
        """Load what a first question would otherwise load. True if anything ran."""


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
    """Liveness, and one field per dependency that is allowed to be missing.

    `status` says the process is answering, which is the only thing a 200 from
    this route has ever meant. The other four say which halves of it work:
    `/predict` needs a model, `/ask` needs a provider key and a gate setting
    the gate accepts, and each can be absent on a service that is otherwise
    fine. They are fields rather than status codes so that a function that is
    up with a dependency that is not reads as exactly that.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="Always `ok` while the process is answering")
    model_loaded: bool = Field(
        description="False when no version holds the alias, and when nothing has "
        "asked for the model yet on a host that loads it lazily"
    )
    keys_loaded: bool = Field(
        default=True, description="False when either of the agent's provider keys is unset"
    )
    missing_keys: list[str] = Field(
        default_factory=list,
        description="The provider-key variables that are unset; empty when both are set",
    )
    agent_ready: bool = Field(
        default=True,
        description="False when `/ask` would refuse without even building the agent",
    )
    agent_reason: str | None = Field(
        default=None,
        description="Why the agent is not ready, naming the variable to fix; null when it is",
    )


class WarmResponse(BaseModel):
    """What a keepalive ping warmed, and what each part of it cost.

    The same dependency fields `/health` carries, so a caller that is pinging
    rather than checking does not have to call both, plus what this call
    actually did. It is a 200 whatever happened: a ping that failed the health
    check it replaced would be a worse monitor than the one it replaced.
    """

    status: str = Field(description="Always `ok` while the process is answering")
    keys_loaded: bool = Field(description="False when either of the agent's provider keys is unset")
    missing_keys: list[str] = Field(
        default_factory=list, description="The provider-key variables that are unset"
    )
    agent_ready: bool = Field(description="False when nothing could be warmed, and why below")
    agent_reason: str | None = Field(
        default=None, description="Why nothing was warmed; null when there was nothing to say"
    )
    agent_built: bool = Field(
        description="Whether the agent is built and resident in this process now"
    )
    embedder_loaded: bool = Field(
        description="Whether one embedding ran through the card tool's model. False when "
        "there is no card index, which is an agent with only its SQL half"
    )
    seconds: dict[str, float] = Field(
        default_factory=dict,
        description="What this call spent per step, keyed `agent_built` and `embedder_loaded`. "
        "Near zero on a container that was already warm, which is the point of pinging",
    )


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
        self.attempted = False

    def load(self) -> LoadedModel:
        """Load through the loader, recording the failure rather than raising it."""
        self.attempted = True
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

    def ensure(self) -> bool:
        """Load on first need, and say whether this call is the one that did it.

        The lazy half of `create_app`'s `eager_model`. Only the three routes
        that read the model call it, so a host that loads lazily never pays
        for a model on `/health` or `/ask`; the return value is there because
        the caller has a Prometheus gauge to set afterwards and only when
        something actually happened.

        A load that failed counts as having been attempted: the next `/model`
        does not retry it, because retrying a download of the registry on
        every request to a service whose registry is empty is a service that
        answers slowly and still says no. `/reload` is the retry, and it is
        one call away.
        """
        if self.current is not None or self.attempted:
            return False
        self.try_load()
        return True

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


@dataclass(frozen=True)
class AgentReadiness:
    """Whether `/ask` can be expected to work, and the reason it cannot."""

    ready: bool
    reason: str | None = None


Readiness = Callable[[], AgentReadiness]


def missing_keys() -> list[str]:
    """Which of the agent's two provider keys are not in the environment.

    Read on every call rather than once at import, because on Lambda the keys
    arrive after the import: `pipeline.lambda_serve` pulls them out of Secrets
    Manager and writes them into `os.environ`, and a secret that still held a
    placeholder when the container started can be filled while it is running.
    """
    return [name for name in AGENT_KEY_VARS if not os.environ.get(name, "").strip()]


def agent_readiness() -> AgentReadiness:
    """What would stop `/ask` working, read from the environment and nothing else.

    Two things stopped it on the first deployed container, and neither was a
    reason to take the service down: no provider key, because the secret was
    still a placeholder, and `PRA_SQL_GATE=1`, which the gate does not accept.
    Both used to surface as whatever the agent factory raised on the way past,
    which is a 503 with a class name in it, after the container had already
    paid for the LangChain and torch imports to get there.

    Asked here instead, so `/health` can answer it for nothing and `/ask` can
    refuse with the sentence an operator has to act on. The gate's own
    `gate_from_env` is the judge of the gate's configuration, including the
    judge key it needs when it is on; this only catches what it says.
    """
    if PROVIDER_KEY_VAR in missing_keys():
        return AgentReadiness(False, NO_PROVIDER_KEY)
    try:
        gate_from_env()
    except GateConfigError as failure:
        return AgentReadiness(False, str(failure))
    return AgentReadiness(True)


def run_warmer(built: AskAgent) -> tuple[bool, float | None, str | None]:
    """Ask a built agent to warm itself: did it, how long, and what went wrong.

    `warm` is optional on an agent, because the agents the tests inject answer
    from a script and have nothing behind them to make resident. An agent
    without one reports `(False, None, None)`: nothing ran, nothing took any
    time, and nothing is wrong.

    A warmer that raises is caught here. The embedding model not loading is
    worth saying out loud and is not worth failing a keepalive ping over; the
    next question will try again and fail properly if it has to.
    """
    warmer = getattr(built, "warm", None)
    if not callable(warmer):
        return False, None, None
    started = time.perf_counter()
    try:
        ran = bool(warmer())
    except Exception as failure:
        elapsed = round(time.perf_counter() - started, 3)
        reason = f"the embedding model could not be warmed: {type(failure).__name__}: {failure}"
        logger.warning("warming the embedder failed", extra={"error": reason})
        return False, elapsed, reason
    return ran, round(time.perf_counter() - started, 3), None


class AgentHolder:
    """The agent, built on the first question and kept until it stops working.

    Lazily, because building it constructs a provider client that wants a key,
    imports LangChain and torch and loads the card index, and a service that
    refused to start without all of that would make the agent a dependency of
    `/predict`. On Lambda the laziness is also the warming: the build happens
    inside the first invocation and everything it imported stays resident on
    `app.state` for the life of the execution environment, which is the only
    kind of warming a frozen container allows. A background thread would be
    frozen with it.

    A failure is recorded and never cached. The next question builds again,
    so a key pasted in or an environment variable corrected is live without a
    redeployment. That matters more here than it did under uvicorn: the first
    deployed container answered every `/ask` for the rest of its life in 17 ms
    with the same `GateConfigError`, which reads like a broken image and was a
    one-character environment variable.
    """

    def __init__(self, factory: AgentFactory, *, readiness: Readiness | None = None) -> None:
        self.factory = factory
        self.readiness = readiness
        self.current: AskAgent | None = None
        self.error: str | None = None

    def state(self) -> AgentReadiness:
        """Whether `/ask` is expected to work, for `/health` to report without building.

        `readiness` is `None` when the agent was injected, which is a test's
        or a demonstration's and owes the environment nothing: refusing on
        behalf of something that was never going to read `ANTHROPIC_API_KEY`
        would be reporting someone else's problem.

        A build that already succeeded is ready whatever the environment says
        now, and a build that failed is reported with its reason until one
        succeeds.
        """
        if self.current is not None:
            return AgentReadiness(True)
        if self.readiness is not None:
            state = self.readiness()
            if not state.ready:
                return state
        if self.error is not None:
            return AgentReadiness(
                False, f"the last attempt to build the agent failed: {self.error}"
            )
        return AgentReadiness(True)

    def required(self) -> AskAgent:
        """The agent, or a 503 naming what has to change for there to be one."""
        if self.current is not None:
            return self.current
        if self.readiness is not None:
            state = self.readiness()
            if not state.ready:
                # Before the factory, so the answer costs no imports and says
                # what to fix rather than what raised.
                raise HTTPException(status_code=503, detail=state.reason)
        if self.error is not None:
            logger.info(
                "building the agent again after an earlier failure",
                extra={"previous_error": self.error},
            )
        try:
            self.current = self.factory()
        except Exception as failure:
            self.error = f"{type(failure).__name__}: {failure}"
            logger.warning(
                "the agent could not be built; the next question tries again",
                extra={"error": self.error},
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"the agent is unavailable: {self.error}. "
                    f"It needs a provider key in the environment and a warehouse built "
                    f"by `python -m pipeline.gold`."
                ),
            ) from failure
        self.error = None
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

        # A store in the lake is pulled down for the length of the load and not
        # written back: the service reads the registry and never changes it.
        with tracking_store(tracking_uri or default_tracking_uri(), write_back=False) as uri:
            if uri.startswith("file:"):
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(uri)
            client = MlflowClient(tracking_uri=uri)
            version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
            model = mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
            # The codes travel with the run rather than with the model, because
            # they are how a request is turned into the integers the model was
            # fitted on. A model loaded without them would still predict, on the
            # wrong numbers. Read inside the block, because a synced store is
            # gone once it closes.
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
    eager_model: bool = True,
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

    `eager_model` is the one argument that is not an injection: it says where
    the model is loaded. True, the default, loads it here, which is what
    `python -m pipeline.serve` and `compose.yaml` have always done and what
    makes an unreadable registry a startup failure rather than a surprise on
    the first prediction. `pipeline.lambda_serve` passes False, because on
    Lambda this function runs inside the first invocation and the load pulls
    the MLflow file store out of the lake: on the deployed function that was
    28.7 of the 29 seconds the first `/health` took, for a model `/health`
    does not read. False moves it to the first `/model`, `/predict` or
    `/reload`, and `/health` says `model_loaded: false` until then without
    setting it off.
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
    # The environment check belongs to the real agent only. An injected
    # factory builds something that was never going to read a provider key, so
    # refusing `/ask` because the environment has none would be refusing on
    # behalf of a dependency that does not exist.
    agent = AgentHolder(
        agent_factory or default_agent_factory(tracer=tracer, metrics=metrics),
        readiness=agent_readiness if agent_factory is None else None,
    )
    # Both holders on the application object, for a host that owns the process
    # and has to reach inside it. The only one is `pipeline.lambda_serve`, which
    # drops the agent when the warehouse under it has been replaced by a
    # nightly, because the agent holds an open connection to the old file.
    # Nothing on the command-line path reads either attribute, and nothing
    # below this line does.
    app.state.model = holder
    app.state.agent = agent

    def record_load() -> None:
        """Put the loaded version on the `model_info` gauge, or clear it if none."""
        if holder.current is None:
            metrics.model_info.clear()
            return
        metrics.set_model_info(holder.current.name, holder.current.version, holder.current.alias)

    def ensure_model() -> None:
        """Load the model on the first route that needs it, when it was not loaded here.

        A no-op under `eager_model`, and under lazy loading it runs once:
        `ModelHolder.ensure` remembers that it tried.
        """
        if holder.ensure():
            record_load()

    # Eagerly, where the host wants a bad registry to be a startup failure: a
    # service that loads lazily reports healthy until someone asks it a
    # question, which is the wrong time to find out the registry is empty. The
    # gauge is cleared either way, so an unloaded model is an absent series
    # rather than a stale one.
    if eager_model:
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

    @app.get("/health", response_model=HealthResponse, summary="Liveness and dependency state")
    def health() -> HealthResponse:
        """Liveness, and the truth about each dependency, always 200.

        200 because the function is up. A provider key that has not been
        filled in and a mistyped `PRA_SQL_GATE` are dependencies that are not
        ready, not a process that is unhealthy, and a check that went red for
        them would have whatever watches it replacing a container that is
        working. The fields carry the bad news and `/ask` is where it is
        enforced.

        Nothing in here loads the model or builds the agent, and that is the
        point: `/health` is the one route a cold container has to be able to
        answer in milliseconds.
        """
        absent = missing_keys()
        state = agent.state()
        return HealthResponse(
            status="ok",
            model_loaded=holder.current is not None,
            keys_loaded=not absent,
            missing_keys=absent,
            agent_ready=state.ready,
            agent_reason=state.reason,
        )

    @app.get("/warm", response_model=WarmResponse, summary="Pay a first question's costs early")
    def warm() -> WarmResponse:
        """Build the agent and make its embedding model resident. Never raises.

        What a keepalive ping should call instead of `/health`. `/health`
        touches nothing on purpose, so pinging it kept a container alive with
        none of the expensive things in it and the next real question still
        paid for all of them. On the deployed function that was a question
        about a card hitting the 60 s timeout, because the first
        `lookup_cards` has to import torch, load the embedding model and
        embed, over image layers Lambda fetches the first time they are
        touched.

        The provider is never called. The agent's graph is built, which is the
        LangChain import and the tool construction, and one short fixed string
        goes through the card tool's embedder; the language model is a network
        call per question and warming it would be spending money on a ping.

        Nothing in here raises. Every step is reported, a step that failed is
        logged and leaves its reason in `agent_reason`, and the answer is 200
        either way: whatever is pinging this wants a container kept warm, and
        a 500 would make a dependency's bad afternoon look like a dead
        function.
        """
        absent = missing_keys()
        state = agent.state()
        built = agent.current is not None
        embedded = False
        spent: dict[str, float] = {}
        if state.ready:
            started = time.perf_counter()
            try:
                current = agent.required()
            except HTTPException as refused:
                spent["agent_built"] = round(time.perf_counter() - started, 3)
                state = AgentReadiness(False, str(refused.detail))
            except Exception as failure:
                spent["agent_built"] = round(time.perf_counter() - started, 3)
                state = AgentReadiness(False, f"{type(failure).__name__}: {failure}")
                logger.warning("warming could not build the agent", extra={"error": state.reason})
            else:
                spent["agent_built"] = round(time.perf_counter() - started, 3)
                built = True
                embedded, elapsed, failed = run_warmer(current)
                if elapsed is not None:
                    spent["embedder_loaded"] = elapsed
                if failed is not None:
                    state = AgentReadiness(False, failed)
        return WarmResponse(
            status="ok",
            keys_loaded=not absent,
            missing_keys=absent,
            agent_ready=state.ready,
            agent_reason=state.reason,
            agent_built=built,
            embedder_loaded=embedded,
            seconds=spent,
        )

    @app.get("/model", response_model=ModelResponse, summary="Which model is loaded")
    def model() -> ModelResponse:
        ensure_model()
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
        ensure_model()
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
    # Explicitly, although it is the default: a command line that is given a
    # `--tracking-uri` it cannot read should fail now rather than on the first
    # prediction, and that is a decision this line is making.
    app = create_app(loader, eager_model=True)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
