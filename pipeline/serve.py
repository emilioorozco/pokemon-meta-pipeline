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
Its body carries the evidence as well as the answer, because the application
shows a member what was looked up: every statement in full with its first
rows, the cards that matched, the gate's worst verdict over the run, how long
the call took and which run's warehouse answered (docs/agent-service.md). It
takes four optional fields beside the question: a `context` string saying where
the member is standing in the application, which goes to the model as data in
its own element and is never written to a log; a `context_game` summary of the
game on their screen and a `context_first_line` sentence describing it, which
between them decide whether that summary joins the context (one typed Choice
call over the question and the sentence, never over the summary); and a `job`
label saying which kind of question the application routed this as, which is
logged and does nothing else yet.

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
from enum import StrEnum
from typing import Any, Final, Literal, Protocol

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
from pipeline.storage import AnyLocation, duckdb_connect, local_file, location, tracking_store
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
# What `card_tool_reason` says before anything has built an agent to ask. Not
# a fault: `/health` builds nothing on purpose, so until a question or a ping
# has been through, whether there is a card tool is unknown rather than false.
NO_AGENT_YET: Final = "the agent has not been built yet; a question or `GET /warm` builds it"
# The ceiling on the application's page context, in characters. A sentence or
# two today and a redacted summary of the member's own game later, which is
# what sets the number: a few hundred characters of prose with room for the
# summary, and far enough under the prompt's own budget that a context cannot
# become the largest thing in the call. Over it is a 422 and not a truncation:
# a summary cut in half is a summary that says something else, and the
# application is better placed to shorten its own text than this service is.
MAX_CONTEXT_CHARS: Final = 4_000
# The ceiling on the one sentence that describes the game on screen. Three
# hundred characters is a long sentence and a short paragraph, which is the
# shape the field is for: "Your Dragapult ex game against Gardevoir ex, you
# went second, lost in 9 turns" is 74. It is short on purpose rather than by
# accident, because this is the only part of the game that goes to the
# relevance judge and the whole saving of asking about one sentence is lost
# if the sentence is the summary again.
MAX_FIRST_LINE_CHARS: Final = 300
# The three verdicts `POST /ask` can report about the game on screen. The same
# closed set `pipeline.sql_gate` reaches and `pipeline.telemetry` labels by,
# written out here because this module stays importable without LangChain and
# therefore without importing the agent.
ContextRelevanceVerdict = Literal["relevant", "irrelevant", "skipped"]
# Which run built the warehouse an answer came from. Every stage of a nightly
# shares one run id (`pipeline.run_all` sets it for all of them), so the gold
# stage's row in `mart_pipeline_health` carries the same id the publish stage
# wrote beside the rows the application reads, which is the point: a member can
# line an answer up with the night it came from.
RUN_ID_STAGE: Final = "gold"
RUN_ID_QUERY: Final = (
    "select last_run_id from mart_pipeline_health "
    f"where stage = '{RUN_ID_STAGE}' and last_run_id is not null limit 1"
)


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
        """The answer, its tool calls, its evidence, the model and the token usage."""


class AskAgent(Protocol):
    """A built agent: one question in, one answer out, no state between them."""

    def ask(
        self,
        question: str,
        context: str | None = None,
        job: str | None = None,
        context_game: str | None = None,
        context_first_line: str | None = None,
    ) -> AgentResult:
        """Answer one question, told where the member is and what kind of question it is."""


class CardAwareAgent(Protocol):
    """An agent that says whether it was built with its card lookup tool.

    Optional in the same way `WarmableAgent` is, and read off a built agent
    with `getattr`: the agents the tests and the demonstrations inject have no
    card half at all, and reporting a missing tool on behalf of something that
    never had one would be reporting a problem that does not exist.
    """

    card_tool_reason: str | None


class WarmableAgent(Protocol):
    """An agent that can make its expensive parts resident without being asked anything.

    Optional, and read off a built agent with `getattr` rather than required
    of every `AskAgent`: the tests inject agents that answer from a script and
    have nothing to warm, and `/warm` has to work with those too.
    """

    def warm(self) -> bool:
        """Load what a first question would otherwise load. True if anything ran."""


AgentFactory = Callable[[], AskAgent]


class AskJob(StrEnum):
    """The application's own label for what kind of question this is.

    The application routes a question deterministically before it sends it,
    and the label travels with the request so that a log line and a span can
    be read by job without this service guessing at one. An enum rather than
    a free string: a typo in the application would otherwise become a new
    category in a dashboard, which is a silent way to lose half a chart.

    Nothing about the answer depends on it yet. The job playbooks, which are
    the point of having the label, are a later ticket; this one carries it.
    """

    META = "meta"
    MY_GAME = "my_game"
    MY_MISTAKE = "my_mistake"
    MY_RECORD = "my_record"
    CARD_RULES = "card_rules"
    OUT_OF_SCOPE = "out_of_scope"


class AskRequest(BaseModel):
    """One question in natural language, and what the application knows around it."""

    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="What to ask about the metagame, in plain English",
    )
    context: str | None = Field(
        default=None,
        max_length=MAX_CONTEXT_CHARS,
        description=(
            "Where the member is in the application and what is on their screen, as plain "
            f"text and at most {MAX_CONTEXT_CHARS:,} characters; a longer one is a 422 "
            "rather than a truncation, because a summary cut in half is a summary that "
            "says something else. It is read as information and never as an instruction "
            "(docs/agent-safety.md), it is never logged, and no JSON is expected here"
        ),
    )
    context_game: str | None = Field(
        default=None,
        max_length=MAX_CONTEXT_CHARS,
        description=(
            "A redacted plain-text summary of the game the member is looking at, built by "
            "the application from that member's own log; this service never fetches a "
            f"game. A few hundred characters to about 1,500, and at most "
            f"{MAX_CONTEXT_CHARS:,}; a longer one is a 422. It is placed after the "
            "`context` sentence and a blank line, when the relevance decision says the "
            "game bears on the question. Read as information and never as an instruction, "
            "and never logged"
        ),
    )
    context_first_line: str | None = Field(
        default=None,
        max_length=MAX_FIRST_LINE_CHARS,
        description=(
            "One sentence describing the same game, such as `Your Dragapult ex game "
            "against Gardevoir ex, you went second, lost in 9 turns`. At most "
            f"{MAX_FIRST_LINE_CHARS} characters. It is the only part of the game the "
            "relevance decision is shown, which is what keeps that decision one short "
            "call; without it the decision is `skipped` and the game is attached anyway. "
            "Never logged"
        ),
    )
    job: AskJob | None = Field(
        default=None,
        description=(
            "The application's router label for this question. Logged and put on the "
            "span; it changes nothing about the answer today"
        ),
    )


class ToolCallResponse(BaseModel):
    """One tool the agent called while answering, and what came back."""

    tool: str = Field(description="Tool name, `query_marts` or `lookup_cards`")
    input_summary: str = Field(description="The query it was given, shortened to one line")
    rows: int = Field(description="Rows the tool returned; zero for a refusal or an empty result")


class QueryEvidenceResponse(BaseModel):
    """One statement the agent put to the warehouse, and the rows it got back."""

    sql: str = Field(description="The statement in full, as the model wrote it, not shortened")
    description: str = Field(
        default="",
        description="What the lookup was for, in one plain-language line of at most 160 "
        "characters, derived from the statement by the service and never written by the "
        "model. It names no table and no column, so an application can show it to a "
        "member in place of the SQL",
    )
    row_count: int = Field(description="Rows the statement returned; zero for a refusal")
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The first 10 rows, as JSON values: numbers, strings, booleans and null, "
        "with dates as ISO strings and every string cut to 500 characters",
    )
    gate: str = Field(
        description="The SQL gate's verdict on this statement: `off`, `jev:allowed`, "
        "`jev:allowed_low`, `jev:refused` or `jev:error`"
    )
    refused_reason: str | None = Field(
        default=None,
        description="Why there are no rows: the validator's refusal, the gate's refusal, or "
        "the warehouse's own error. Null when the statement ran",
    )
    refused_code: str | None = Field(
        default=None,
        description="The same reason as one word, for an application that has to switch on "
        "it: `table_not_found` (a table name the model invented, which is a detour and "
        "not a block), `table_not_allowed` (a real table off the allowlist), "
        "`statement_not_allowed`, `judge_low_confidence`, `judge_refused` or `error`. "
        "Null when the statement ran",
    )


class CardEvidenceResponse(BaseModel):
    """One printed card the agent looked up, as a reader cites it."""

    name: str = Field(description="The card's printed name")
    set_code: str = Field(description="The set it is cited from, as the card corpus spells it")
    number: str = Field(description="Its number in that set")
    text: str = Field(
        description="The card's printed text without the name and set over it, cut to "
        "500 characters"
    )


class EvidenceResponse(BaseModel):
    """What the answer rests on: the queries that ran and the cards that matched."""

    queries: list[QueryEvidenceResponse] = Field(
        default_factory=list, description="Every statement of this run, in call order"
    )
    cards: list[CardEvidenceResponse] = Field(
        default_factory=list,
        description="The cards the run matched, each once, in the order it first saw them, "
        "at most 10",
    )


class AskResponse(BaseModel):
    """The agent's answer, and everything it did to get there.

    `tool_calls` is the tally this route has always carried and `evidence` is
    the same run written out so it can be read: the whole statement rather
    than a hundred characters of it, the rows rather than their number, and
    the cards rather than the fact that a lookup happened. Both are here
    because the evaluation scripts read the first and a member reading the
    answer reads the second.
    """

    model_config = ConfigDict(protected_namespaces=())

    answer: str = Field(description="The answer, in prose, citing the sample size it read")
    tool_calls: list[ToolCallResponse] = Field(
        description="Every tool call of this run, in order. An empty list means the model "
        "answered without reading the warehouse, which its prompt tells it not to do"
    )
    model: str = Field(description="Provider model that answered")
    usage: dict[str, int] = Field(
        description="Token counts the provider reported; empty when it reported none. "
        "`input_tokens`, `output_tokens` and `total_tokens`, plus "
        "`cache_read_input_tokens` and `cache_creation_input_tokens`, which are the "
        "share of the input side served from and written to the cached prompt prefix "
        "and are zero when the prefix was too short to cache"
    )
    evidence: EvidenceResponse = Field(
        default_factory=EvidenceResponse,
        description="What the answer was built from, for a panel that shows the member "
        "what was looked up",
    )
    gate_summary: str = Field(
        default="off",
        description="What happened to the answer, not to the worst attempt behind it: "
        "`refused` only when every query was refused or none ran, `allowed_low` if the "
        "gate let one through unsurely, `allowed` if they ran under the gate, `off` if "
        "there were none or the gate is not on. A refused attempt followed by a query "
        "that ran is not `refused`; the attempt is still in `evidence.queries`",
    )
    latency_ms: int = Field(
        default=0, description="Wall time of the whole call inside the service, in milliseconds"
    )
    run_id: str | None = Field(
        default=None,
        description="Which run's data answered: the pipeline run that built the warehouse, "
        "or that warehouse's last-modified time when the run metadata cannot be read. "
        "Null when there is no warehouse to ask",
    )
    context_used: bool = Field(
        default=False,
        description="Whether a non-empty context was really placed in front of the "
        "question, from either `context` or `context_game`. False for no context and "
        "for one that was empty once our own delimiters were taken out of it, so an "
        "`about this page` chip built on this is honest. The context text itself is "
        "not echoed here",
    )
    context_game_used: bool = Field(
        default=False,
        description="Whether the `context_game` summary in particular was placed. "
        "False when none was sent and when the relevance decision said `irrelevant`, "
        "so a chip that names the game can be exact rather than inferred from "
        "`context_used`",
    )
    context_relevance: ContextRelevanceVerdict | None = Field(
        default=None,
        description="What the relevance decision said about the game on screen: "
        "`relevant` (the summary was placed), `irrelevant` (it was dropped), or "
        "`skipped` (no judge was configured, the call failed, or no "
        "`context_first_line` came with the game, and the summary was placed anyway). "
        "Null when no `context_game` was sent and there was nothing to decide",
    )


class HealthResponse(BaseModel):
    """Liveness, and one field per dependency that is allowed to be missing.

    `status` says the process is answering, which is the only thing a 200 from
    this route has ever meant. The rest say which halves of it work:
    `/predict` needs a model, `/ask` needs a provider key and a gate setting
    the gate accepts, and the card half of `/ask` needs a readable index of
    the right format. Each can be absent on a service that is otherwise fine.
    They are fields rather than status codes so that a function that is up
    with a dependency that is not reads as exactly that.

    `agent_ready` and `card_tool` are deliberately two fields. The first is
    "`/ask` would refuse", the second is "`/ask` would answer, without card
    text", and a deployment has sat in the second state for hours looking
    like the first was fine, which it was.
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
    card_tool: bool = Field(
        default=True,
        description="Whether the agent in hand answers card questions. False when no agent "
        "has been built yet, and when the one that was came up without `lookup_cards`",
    )
    card_tool_reason: str | None = Field(
        default=None,
        description="Why there is no card tool, naming what to rebuild; null when there is",
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
    card_tool: bool = Field(
        default=True,
        description="Whether the agent that is built answers card questions. A ping that "
        "finds this false builds the agent again, in case the index has been rebuilt since",
    )
    card_tool_reason: str | None = Field(
        default=None,
        description="Why there is no card tool, naming what to rebuild; null when there is",
    )
    embedder_loaded: bool = Field(
        description="Whether one embedding ran through the card tool's model. False whenever "
        "`card_tool` is false, which is an agent with only its SQL half, and "
        "`card_tool_reason` says which index is missing or of the wrong format"
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


def card_tool_state(built: AskAgent) -> tuple[bool, str | None]:
    """Whether a built agent got its card lookup tool, and the reason it did not.

    Read with `getattr` for the reason `run_warmer` reads `warm` with one: an
    injected agent is a script with no card half, and it owes this nothing. An
    agent that does not report is therefore taken to be whole, so a test's
    stub is not permanently half-built in `/health`.
    """
    if not hasattr(built, "card_tool_reason"):
        return True, None
    reason: str | None = getattr(built, "card_tool_reason", None)
    return reason is None, reason


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

    A build that came up without the card tool is not a failure and is not a
    finished build either, and this used to have no third answer for it. A
    container built while the lake still held an index of the previous format
    kept that half-agent and said "I do not have access to card text" long
    after the nightly had rebuilt the index. So the reason is recorded, it is
    reported on `/health` and `/warm`, and the build is thrown away and tried
    again when a ping asks or when the host notices the index has changed.
    The agent answers SQL questions the whole time, which is why it is kept
    rather than refused.
    """

    def __init__(self, factory: AgentFactory, *, readiness: Readiness | None = None) -> None:
        self.factory = factory
        self.readiness = readiness
        self.current: AskAgent | None = None
        self.error: str | None = None
        # Why the agent in hand has no card tool; None when it has one. Read
        # off the built agent rather than guessed at, and meaningless while
        # `current` is None, which `card_state` is what accounts for.
        self.card_reason: str | None = None

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

    def card_state(self) -> tuple[bool, str | None]:
        """Whether the agent in hand answers card questions, and why it does not.

        Builds nothing. Before the first build there is no agent to ask, and
        the answer is False with `NO_AGENT_YET`: `/health` is the one route a
        cold container has to answer in milliseconds, and guessing at the
        state of an index nobody has read would be worse than saying so.
        """
        if self.current is None:
            return False, NO_AGENT_YET
        return self.card_reason is None, self.card_reason

    def incomplete(self) -> bool:
        """An agent that answers SQL questions but was built without the card tool."""
        return self.current is not None and self.card_reason is not None

    def retry_incomplete(self) -> bool:
        """Throw away a half-built agent so the next build reads the index again.

        `GET /warm` is the caller, not `/ask`: a question that rebuilt the
        graph every time the index was unreadable would pay the LangChain
        construction per question to keep failing the same way. A ping has
        nobody waiting on it, and it is the one that runs every few minutes,
        so an index rebuilt at any point in the night is picked up within one
        ping of landing.
        """
        if not self.incomplete():
            return False
        logger.info(
            "dropping the agent that was built without its card tool, to read the index again",
            extra={"reason": self.card_reason},
        )
        self.current = None
        return True

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
        _, self.card_reason = card_tool_state(self.current)
        if self.card_reason is not None:
            logger.warning(
                "the agent was built without its card lookup tool; it answers SQL questions "
                "and the next ping tries the index again",
                extra={"reason": self.card_reason},
            )
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


# The run id of each warehouse copy this process has asked about, keyed by the
# local file and the modification time of it. The answer changes only when the
# file does, and on a long-lived container the file is replaced by the nightly
# and by nothing else (`pipeline.lambda_serve` swaps the copy, which lands here
# as a new path).
_run_ids: dict[tuple[str, int], str | None] = {}


def warehouse_run_id(warehouse: AnyLocation = WAREHOUSE_PATH) -> str | None:
    """Which run's data answered, for the member reading the answer.

    `mart_pipeline_health` carries one row per stage with that stage's last
    run id on it, and the gold stage's row is the run that built the marts in
    this file. Every stage of a nightly shares one id, so it is also the id
    the publish stage wrote on the rows the application already shows, and the
    two can be lined up.

    That mart is a view over the run-metrics Parquet in the lake, so reading
    it needs the lake as well as the warehouse file, and a serving container
    may have the file and not the lake. None of that is a failure here: when
    the query cannot run, or no gold run is recorded, the answer is the
    warehouse's own last-modified time as an ISO string under the same key.
    It is a weaker identifier and it still answers "which night is this",
    which is what the field is for. Null is the third answer, for a service
    with no warehouse at all.

    Cached per copy of the file. The query is cheap and it is not free, and
    the answer cannot change while the bytes do not.

    Nothing in here raises. This is provenance on an answer the agent has
    already produced, and a lake that cannot be reached must not turn a
    question that was answered into a 500.
    """
    try:
        target = location(warehouse)
        if not target.is_file():
            return None
        local = local_file(target)
        modified = local.stat().st_mtime_ns
    except Exception as unreachable:  # noqa: BLE001 - provenance must not fail a question
        logger.info(
            "the warehouse could not be reached for its run id",
            extra={"error": f"{type(unreachable).__name__}: {unreachable}"},
        )
        return None
    key = (str(local), modified)
    if key in _run_ids:
        return _run_ids[key]
    resolved = _published_run_id(target) or _modified_at(modified)
    _run_ids[key] = resolved
    return resolved


def _published_run_id(warehouse: AnyLocation) -> str | None:
    """The gold stage's last run id out of the warehouse, or None if it cannot be read."""
    import duckdb

    try:
        connection = duckdb_connect(warehouse)
    except (duckdb.Error, OSError) as unreadable:
        logger.info(
            "the warehouse could not be opened for its run id",
            extra={"error": f"{type(unreadable).__name__}: {unreadable}"},
        )
        return None
    try:
        row = connection.sql(RUN_ID_QUERY).fetchone()
    except duckdb.Error as unreadable:
        # An older warehouse without the ops views, and a container that cannot
        # reach the lake those views read, both land here. The fallback says
        # which copy of the data answered, which is most of the question.
        logger.info(
            "the warehouse has no readable run metadata; falling back to its timestamp",
            extra={"error": f"{type(unreadable).__name__}: {unreadable}"},
        )
        return None
    finally:
        connection.close()
    return str(row[0]) if row and row[0] else None


def _modified_at(modified_ns: int) -> str:
    """A file's modification time as an ISO string, which is the weaker run id."""
    return datetime.fromtimestamp(modified_ns / 1_000_000_000, UTC).isoformat()


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
    warehouse: AnyLocation | None = None,
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

    `warehouse` is only ever read for the `run_id` an answer carries: the
    agent opens its own, through the factory above. It is a parameter, and
    resolved here rather than in the signature, so that a test can point it at
    a warehouse it built rather than at whatever is in the data directory of
    the machine running the suite.
    """
    holder = ModelHolder(loader or mlflow_loader())
    marts = WAREHOUSE_PATH if warehouse is None else warehouse
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
    # drops the agent when either of the two things it was built over has been
    # replaced by a nightly: the warehouse, to which it holds an open
    # connection, and the card index, which it loaded into memory.
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
        card_ok, card_reason = agent.card_state()
        return HealthResponse(
            status="ok",
            model_loaded=holder.current is not None,
            keys_loaded=not absent,
            missing_keys=absent,
            agent_ready=state.ready,
            agent_reason=state.reason,
            card_tool=card_ok,
            card_tool_reason=card_reason,
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

        It is also the retry for a half-built agent. One that came up without
        `lookup_cards` is reported as `card_tool: false` with the reason, and
        thrown away here so that this build reads the index again: the nightly
        rebuilds that index while nobody is asking anything, and a container
        built before it landed used to keep the half-agent until it died.

        Nothing in here raises. Every step is reported, a step that failed is
        logged and leaves its reason in `agent_reason`, and the answer is 200
        either way: whatever is pinging this wants a container kept warm, and
        a 500 would make a dependency's bad afternoon look like a dead
        function.
        """
        absent = missing_keys()
        state = agent.state()
        # An agent that came up without its card tool is not a finished build,
        # and a ping is the right moment to try again: the nightly replaces the
        # index while nobody is asking anything, and a container built before
        # it landed would otherwise answer card questions with an apology for
        # the rest of its life. The SQL half is briefly given up to get the
        # whole thing back; a build that fails leaves the reason in
        # `agent_reason` and the next question builds again.
        agent.retry_incomplete()
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
        card_ok, card_reason = agent.card_state()
        return WarmResponse(
            status="ok",
            keys_loaded=not absent,
            missing_keys=absent,
            agent_ready=state.ready,
            agent_reason=state.reason,
            agent_built=built,
            card_tool=card_ok,
            card_tool_reason=card_reason,
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

        Three fields are the service's rather than the agent's. `latency_ms`
        is measured from the first line of the handler, so a question that
        had to build the agent reports what the caller actually waited for;
        `run_id` says which warehouse answered; and `evidence` is the agent's
        own and passes through untouched.

        The four context fields and `job` go straight through to the agent and
        are not read here. `context_used`, `context_game_used` and
        `context_relevance` come back off the agent rather than being computed
        from the request, because what the application wants to know is what
        was really put in front of the question: a context that was nothing
        but delimiters was not placed, and a game the relevance decision
        dropped was not either.
        """
        started = time.perf_counter()
        result = agent.required().ask(
            request.question,
            context=request.context,
            job=request.job.value if request.job is not None else None,
            context_game=request.context_game,
            context_first_line=request.context_first_line,
        )
        payload = dict(result.as_dict())
        payload["run_id"] = warehouse_run_id(marts)
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000)
        return AskResponse.model_validate(payload)

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
