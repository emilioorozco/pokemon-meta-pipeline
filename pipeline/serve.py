"""Serving: the promoted model behind `POST /predict`, and nothing else.

One FastAPI application with four endpoints. `/predict` answers the question
the feature table was built to ask, "this side, this board, this far in, who
wins?", `/health` says whether a model is loaded, `/model` says which one, and
`/reload` picks up a promotion without a restart.

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
"""

import argparse
import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, default_tracking_uri
from pipeline.ml_features import CATEGORICAL, MODEL_FEATURES, ArchetypeCodes, design_matrix

CODES_ARTIFACT: Final = "archetype_codes.json"
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


def describe(model: LoadedModel) -> ModelResponse:
    """The loaded model as the `/model` body."""
    return ModelResponse(
        name=model.name,
        version=model.version,
        alias=model.alias,
        loaded_at=model.loaded_at,
        metrics=model.metrics,
    )


def create_app(loader: Loader | None = None) -> FastAPI:
    """The application, with its model loader injected.

    The loader is a plain callable returning a `LoadedModel`, so a test passes a
    stub and the command line passes `mlflow_loader()`. Nothing below this line
    knows that MLflow exists.
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
    # At import rather than on first request: a service that loads lazily reports
    # healthy until someone asks it a question, which is the wrong time to find
    # out the registry is empty.
    holder.try_load()

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
            raise HTTPException(
                status_code=503, detail=f"reload failed: {type(failure).__name__}: {failure}"
            ) from failure
        return describe(holder.required())

    @app.post("/predict", response_model=PredictResponse, summary="Win probability for one turn")
    def predict(request: PredictRequest) -> PredictResponse:
        loaded = holder.required()
        payload = request.features()
        matrix = design_matrix(pd.DataFrame([payload]), loaded.codes)
        return PredictResponse(
            win_probability=probability(loaded.predictor, matrix),
            model_name=loaded.name,
            model_version=loaded.version,
            model_alias=loaded.alias,
            features_used=list(MODEL_FEATURES),
            unknown_archetypes=unknown_archetypes(payload, loaded.codes),
        )

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

    import uvicorn

    app = create_app(mlflow_loader(tracking_uri=args.tracking_uri, alias=args.alias))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
