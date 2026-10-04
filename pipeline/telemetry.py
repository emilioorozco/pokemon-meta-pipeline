"""Traces and metrics for the serving process: the two pillars a log line is not.

`pipeline/observability.py` covers a batch stage, where the unit of work is a
run: one JSON line per event and one `run_metrics` row per stage. A service has
no run to close, so the same questions need different instruments, and this
module adds the two that a request-scoped process wants.

**Metrics** answer "how is it doing", over all requests at once: rate, error
ratio, and the latency distribution that a mean hides. They are counters and
histograms in a Prometheus registry, scraped off `GET /metrics`, and their cost
does not grow with traffic, which is exactly why they cannot answer anything
about one particular request.

**Traces** answer "where did this one go", request by request: a span per HTTP
call from the FastAPI instrumentation, and a child span around the inference
itself, so "the p95 moved" can be followed to "the model call is slow" rather
than "something in the process is slow". They are exported over OTLP to a
collector, which is the piece that is missing on a laptop, so it is optional.

Four choices worth knowing.

The tracer provider is not global. `setup_tracing` builds one and hangs it off
`app.state`, rather than calling `trace.set_tracer_provider`, because the global
can be set exactly once per process: two applications in one test process would
mean the second one silently keeps the first one's exporter. The endpoint reads
its tracer back off the application, which is the same shape as the injected
model loader in `serve.py`, and for the same reason.

No collector is not an error. Without `OTEL_EXPORTER_OTLP_ENDPOINT`, or with one
naming a host that does not resolve, the provider is a genuine no-op and the
service starts exactly as before; an exporter that retried a connection to
nothing on every request would make observability the reason the service is
down, which is the failure mode it exists to prevent.

The Prometheus registry is private, not the process-global default. Two
applications in one test process would otherwise collide on the first metric
name registered, and `generate_latest` on the default registry would also serve
whatever any other library happened to register.

Labels are bounded on purpose. The route label is the template, `/predict`, not
the request path, so a service that is scanned for `/wp-admin.php` does not get
one time series per probe; `unknown_archetype` is a yes or no rather than the
archetype name, which is unbounded by definition, because the name is already on
the trace and in the request log where an unbounded value is free.
"""

import logging
import os
import socket
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from pipeline.observability import git_commit

logger = logging.getLogger(__name__)

SERVICE_NAME: Final = "pra-predict"
OTLP_ENDPOINT_VAR: Final = "OTEL_EXPORTER_OTLP_ENDPOINT"
INFERENCE_SPAN: Final = "predict.inference"
# Matched against the URL by the FastAPI instrumentation, as a comma-separated
# list of regular expressions. A liveness probe and a scrape every fifteen
# seconds are the two highest-volume requests a quiet service gets, and neither
# is a trace anyone will ever read.
EXCLUDED_URLS: Final = "health,metrics"
# The route label for a request that matched no route. Without it every 404 path
# is a new time series, which is how a metrics store is filled up by a scanner.
UNMATCHED_ROUTE: Final = "unmatched"

# The label values of `agent_prompt_tokens_total`, and the usage key each one
# counts. A closed set of three, because the provider splits the input side
# exactly this way and a fourth value would mean the split stopped adding up.
PROMPT_TOKEN_KINDS: Final[dict[str, str]] = {
    "cache_read": "cache_read_input_tokens",
    "cache_creation": "cache_creation_input_tokens",
    "uncached": "input_tokens",
}

# The label values of `agent_context_relevance_total`, which is the closed set
# of three verdicts `pipeline.sql_gate` can reach about the game on a member's
# screen. `skipped` is not a model's answer: it is no judge configured, a call
# that failed, or a game sent with no sentence to judge it by, and all three
# attach the game anyway.
RELEVANCE_VERDICTS: Final[tuple[str, ...]] = ("relevant", "irrelevant", "skipped")

# Tighter than the Prometheus defaults, which start at 5 milliseconds: one
# LightGBM call on one row is faster than that, so the default buckets would put
# every inference in the first one and report a p95 of "under 5ms" forever.
_INFERENCE_BUCKETS: Final[tuple[float, ...]] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

# One short Choice call over a network, which the vendor answers in tens of
# milliseconds and which this project gives the same five second budget as
# everything else it asks Jev. So the buckets start where a fast call lands
# and end at the timeout, rather than at the Prometheus default of ten
# seconds nothing here can reach.
_RELEVANCE_BUCKETS: Final[tuple[float, ...]] = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)


def service_version() -> str:
    """The commit this process is running, or `dev` outside a checkout.

    The same identifier `run_metrics` records per run, so a trace and a batch
    row can be tied to the same code. A container built from a clean image has
    no git at all, which is a `dev` rather than a failed startup.
    """
    return git_commit() or "dev"


def resolves(endpoint: str) -> bool:
    """Whether the endpoint's host exists on this network at all.

    A name lookup, deliberately, and not a connection: the two states this has
    to tell apart are "there is no collector here" and "the collector is up but
    still binding its port". Under compose the second one resolves, because the
    name exists as soon as the container does, so a service started a moment
    before its collector still exports; the first one does not resolve, which is
    what `docker compose up predict` without the observability profile looks
    like, and what this function is for.

    The alternative was to let the exporter find out: it logs a warning per
    retry, per batch, forever, so a service run without the profile fills its
    log with the telemetry failing to leave. Losing the spans is the correct
    outcome; announcing it every two seconds is not.
    """
    host = urlparse(endpoint).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
    except OSError:
        return False
    return True


def _exporter() -> SpanExporter | None:
    """An OTLP exporter when an endpoint is configured and reachable, else nothing.

    Imported inside the function because the exporter package pulls in protobuf
    and the HTTP client, and a process that is not exporting should not pay for
    either at import time.
    """
    endpoint = os.environ.get(OTLP_ENDPOINT_VAR, "").strip()
    if not endpoint:
        return None
    if not resolves(endpoint):
        logger.warning(
            "tracing is off: the configured collector does not resolve",
            extra={"endpoint": endpoint},
        )
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def build_tracer_provider(
    service_name: str = SERVICE_NAME,
    *,
    exporter: SpanExporter | None = None,
    span_processor: SpanProcessor | None = None,
) -> trace.TracerProvider:
    """A provider that exports if it can and is a genuine no-op if it cannot.

    Separate from `setup_tracing` because the agent's command line has spans to
    emit and no FastAPI application to hang them off: `python -m pipeline.agent`
    wants the same provider and none of the HTTP instrumentation. Both callers
    go through here so there is one answer to "is there a collector".
    """
    processor = span_processor
    if processor is None and exporter is not None:
        # An injected exporter is a caller that wants to read the spans back,
        # which a batching processor would hand over some time later or never.
        processor = SimpleSpanProcessor(exporter)
    if processor is None:
        configured = _exporter()
        # Batched for the real one: a span per request sent as its own HTTP
        # request would put the collector on the serving path.
        processor = BatchSpanProcessor(configured) if configured is not None else None

    if processor is None:
        # Nothing to export to: a real no-op, not a provider quietly recording
        # and dropping. `with tracer.start_as_current_span(...)` still works, on
        # a non-recording span that costs nothing.
        return trace.NoOpTracerProvider()
    sdk_provider = TracerProvider(
        resource=Resource.create(
            {"service.name": service_name, "service.version": service_version()}
        )
    )
    sdk_provider.add_span_processor(processor)
    return sdk_provider


def setup_tracing(
    app: FastAPI,
    service_name: str = SERVICE_NAME,
    *,
    exporter: SpanExporter | None = None,
    span_processor: SpanProcessor | None = None,
) -> trace.Tracer:
    """Instrument the application and return the tracer its endpoints should use.

    `exporter` and `span_processor` are for the tests, which assert against an
    in-memory exporter; the command line passes neither and the environment
    decides. The tracer is also stored on `app.state.tracer`, because the
    endpoint functions are closures inside `create_app` and reading it back off
    the request's application is what keeps two applications in one process from
    sharing a provider.
    """
    provider = build_tracer_provider(service_name, exporter=exporter, span_processor=span_processor)

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls=EXCLUDED_URLS,
        # The ASGI instrumentation emits a child span per `receive` and `send`
        # message by default, so one `/predict` arrives in Jaeger as six spans
        # of which four are the protocol talking to itself. They matter when the
        # question is about streaming or a slow client, and this service does
        # neither: it reads one small body and writes one small body.
        exclude_spans=["receive", "send"],
    )
    tracer = provider.get_tracer(__name__)
    app.state.tracer_provider = provider
    app.state.tracer = tracer
    return tracer


@dataclass(frozen=True)
class ServiceMetrics:
    """The service's Prometheus instruments, and the registry they live in.

    A frozen dataclass rather than module-level collectors, so a test can build
    an application, read its numbers and throw both away without the next test
    seeing the counts.
    """

    registry: CollectorRegistry
    requests: Counter
    request_duration: Histogram
    inference_duration: Histogram
    predictions: Counter
    agent_tool_calls: Counter
    agent_prompt_tokens: Counter
    agent_context_relevance: Counter
    agent_relevance_duration: Histogram
    agent_unverified_numbers: Counter
    model_info: Gauge

    def observe_request(self, method: str, route: str, status: int, duration_s: float) -> None:
        """One HTTP call: the counter and the latency histogram, from one place."""
        self.requests.labels(method=method, route=route, status=str(status)).inc()
        self.request_duration.labels(method=method, route=route).observe(duration_s)

    @contextmanager
    def time_inference(self, model_version: str) -> Iterator[None]:
        """Time the model call itself, separately from the request around it."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.inference_duration.labels(model_version=model_version).observe(
                time.perf_counter() - started
            )

    def count_prediction(self, model_version: str, *, unknown: bool) -> None:
        """One answered `/predict`, split by whether the model had seen the decks."""
        self.predictions.labels(
            model_version=model_version, unknown_archetype=str(unknown).lower()
        ).inc()

    def count_tool_call(self, tool: str, gate: str = "off") -> None:
        """One agent tool call, by tool and by what the optional SQL gate said.

        `gate` is a closed set of five: `off` when no gate ran, which is every
        `lookup_cards` call and every call made with `PRA_SQL_GATE` unset, and
        `jev:allowed`, `jev:allowed_low` (an allow under the confidence
        threshold, let through and marked), `jev:refused` or `jev:error` when
        one did. It defaults so
        that a caller with no gate to report does not have to know the gate
        exists, and it is bounded for the usual reason: a label whose values a
        provider chooses is one time series per provider mood.
        """
        self.agent_tool_calls.labels(tool=tool, gate=gate).inc()

    def count_prompt_tokens(self, usage: Mapping[str, int]) -> None:
        """One answered question's input tokens, split three ways by how they were paid for.

        The three kinds are the provider's own split of the input side and they
        do not overlap: `cache_read` was served from a written prefix at a
        tenth of the price, `cache_creation` wrote one at a quarter over, and
        `uncached` is `input_tokens`, which the provider reports as the
        remainder after the other two. Summed, they are what the question cost
        on input; divided, `cache_read` over the sum is the hit ratio.

        Nothing is recorded when the run reported no usage at all, because a
        counter that ticked by zero on every fake model would make an empty
        dashboard look like a working one.
        """
        if not usage:
            return
        for kind, name in PROMPT_TOKEN_KINDS.items():
            self.agent_prompt_tokens.labels(kind=kind).inc(usage.get(name, 0))

    def observe_context_relevance(self, verdict: str, duration_s: float) -> None:
        """One decision about the game on a member's screen: the verdict and its cost in time.

        Recorded only when the application really sent a game, so the counter
        is the number of questions asked from a game page and not the number
        of questions. Two series out of one call because they answer two
        questions a panel asks together: how often the game is worth
        attaching, which is the split by verdict, and what deciding that adds
        to a member's wait, which is the histogram. A `skipped` share that
        climbs is the judge failing, and it is visible here before it is
        visible anywhere else.

        `verdict` is a closed set of three (`RELEVANCE_VERDICTS`), for the
        same reason the gate label is closed: a label a provider chooses is
        one time series per provider mood.
        """
        self.agent_context_relevance.labels(verdict=verdict).inc()
        self.agent_relevance_duration.observe(duration_s)

    def count_unverified_numbers(self, count: int) -> None:
        """Numbers one answer stated that nothing the run read can account for.

        A count of numbers rather than of answers, and incremented by zero on
        a healthy question so the series exists from the first scrape: a
        counter that appears only when something is wrong is a counter nobody
        has a baseline for. No label, because the only labels available would
        be the numbers themselves, and a number a model wrote is one time
        series per hallucination.

        It does not sit at zero on a healthy service, because the check
        knows values and not arithmetic and a correctly summed total counts
        here too. What it is good for is the slope: a rate climbing against
        the questions answered is the prompt's rule 10 losing, and it is
        visible here before it is visible in anybody's reading of an answer.
        """
        self.agent_unverified_numbers.inc(count)

    def set_model_info(self, name: str, version: str, alias: str) -> None:
        """Record which model is loaded, as the usual info-gauge-set-to-one.

        Cleared first: after a `/reload` the previous version's series would
        otherwise stay at 1 forever, and a dashboard reading `model_info` would
        show two production models.
        """
        self.model_info.clear()
        self.model_info.labels(name=name, version=version, alias=alias).set(1)


def route_label(request: Request) -> str:
    """The matched route's template, or `unmatched`.

    Starlette puts the route it chose on the scope during routing, so this is
    read after the response, not before. `request.url.path` would be the raw
    path, which for a parameterised route is one time series per parameter value
    and for a scanner is one per probe.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else UNMATCHED_ROUTE


def build_metrics() -> ServiceMetrics:
    """The instruments and their private registry, with nothing mounted.

    The agent's command line increments the tool-call counter with no HTTP
    server to scrape it, and the agent tests read the counter back without
    building an application, so the construction is separate from the mounting.
    """
    registry = CollectorRegistry()
    metrics = ServiceMetrics(
        registry=registry,
        requests=Counter(
            "http_requests_total",
            "HTTP requests answered, by method, matched route and status code.",
            ["method", "route", "status"],
            registry=registry,
        ),
        request_duration=Histogram(
            "http_request_duration_seconds",
            "Wall time from the first middleware to the response, by method and route.",
            ["method", "route"],
            registry=registry,
        ),
        inference_duration=Histogram(
            "model_inference_duration_seconds",
            "Wall time of the model call alone, by the registered version that answered.",
            ["model_version"],
            buckets=_INFERENCE_BUCKETS,
            registry=registry,
        ),
        predictions=Counter(
            "model_predictions_total",
            "Predictions returned, by model version and whether an archetype was unseen.",
            ["model_version", "unknown_archetype"],
            registry=registry,
        ),
        agent_tool_calls=Counter(
            "agent_tool_calls_total",
            "Agent tool invocations, by tool name and by the SQL gate's verdict.",
            ["tool", "gate"],
            registry=registry,
        ),
        agent_prompt_tokens=Counter(
            "agent_prompt_tokens_total",
            "Input tokens the provider billed the agent for, by how they were paid for.",
            ["kind"],
            registry=registry,
        ),
        agent_context_relevance=Counter(
            "agent_context_relevance_total",
            "Decisions about the game on a member's screen, by verdict.",
            ["verdict"],
            registry=registry,
        ),
        agent_relevance_duration=Histogram(
            "agent_context_relevance_duration_seconds",
            "Wall time of the one Choice call that decides whether the game on screen "
            "bears on the question.",
            buckets=_RELEVANCE_BUCKETS,
            registry=registry,
        ),
        agent_unverified_numbers=Counter(
            "agent_unverified_numbers_total",
            "Numbers in an agent answer that no row, card or fact of its run can account for.",
            registry=registry,
        ),
        model_info=Gauge(
            "model_info",
            "Always 1; the loaded model's name, version and alias are the labels.",
            ["name", "version", "alias"],
            registry=registry,
        ),
    )
    return metrics


def setup_metrics(app: FastAPI) -> ServiceMetrics:
    """Build the instruments, mount `GET /metrics`, and return them.

    Mounting the endpoint here rather than in `serve.py` keeps the exposition
    format, the registry and the content type in one module: the service's
    endpoints are about win probabilities, and this one is about the process.
    """
    metrics = build_metrics()
    registry = metrics.registry

    @app.get(
        "/metrics",
        summary="Prometheus exposition",
        # Out of the OpenAPI schema on purpose: it is not part of the service's
        # contract with a caller, it is what the scrape reads.
        include_in_schema=False,
    )
    def prometheus() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    app.state.metrics = metrics
    return metrics
