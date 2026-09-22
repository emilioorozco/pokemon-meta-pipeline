"""The traces and the metrics the serving process emits, with no collector anywhere.

These belong in the fast suite for the same reason the serving tests do. What
breaks in instrumentation is not the exporter, it is the label that turns out to
be the raw path, the histogram that is never observed because the timer sits on
the wrong side of a return, and the span whose attributes were renamed in the
dashboard but not in the code. None of that needs a collector to catch, and a
suite that needs one is a suite that is skipped.

So both exports are replaced rather than mocked: spans go to the SDK's in-memory
exporter and the metrics go to a registry the application owns, which is why
`setup_metrics` builds its own registry instead of using the process-global one.
The assertions are then made against the real Prometheus text format and the
real finished spans, not against a recording of which functions were called.
"""

import re
from collections.abc import Iterator
from typing import Final

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pipeline import serve, telemetry
from tests.test_serve import BODY, Registry, StubPredictor, loaded

# Every metric family the dashboard and the alerting rules are allowed to
# assume exists. A rename is a broken panel, which is the kind of breakage that
# is noticed weeks later by someone looking at a flat line.
FAMILIES: Final[tuple[str, ...]] = (
    "http_requests_total",
    "http_request_duration_seconds",
    "model_inference_duration_seconds",
    "model_predictions_total",
    "agent_tool_calls_total",
    "model_info",
)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def client(exporter: InMemorySpanExporter) -> Iterator[TestClient]:
    app = serve.create_app(Registry(loaded(StubPredictor(0.73), "7")), span_exporter=exporter)
    with TestClient(app) as started:
        yield started


def sample(text: str, metric: str, **labels: str) -> float:
    """One sample out of the Prometheus exposition, by metric name and labels.

    Parsing the text rather than reading the collectors directly is the point:
    the exposition is what Prometheus scrapes, so a metric that is recorded but
    not exposed, or exposed under a label the dashboard does not use, fails here
    the way it would fail in Grafana.
    """
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        match = re.match(rf"^{re.escape(metric)}(?:\{{(.*)\}})?\s+(\S+)$", line)
        if match is None:
            continue
        found: dict[str, str] = {}
        for pair in re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', match.group(1) or ""):
            found[pair[0]] = pair[1]
        if all(found.get(key) == value for key, value in labels.items()):
            return float(match.group(2))
    raise AssertionError(f"no sample {metric}{labels} in:\n{text}")


def test_metrics_exposes_every_family_the_dashboard_reads(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for family in FAMILIES:
        assert f"# TYPE {family}" in response.text, family


def test_the_loaded_model_is_on_the_info_gauge(client: TestClient) -> None:
    """`model_info` is how a panel says which version produced the line it is drawing."""
    text = client.get("/metrics").text
    assert sample(text, "model_info", name="win-probability", version="7", alias="production") == 1


def test_two_predicts_are_two_requests_and_two_inferences(client: TestClient) -> None:
    """The counter, the request histogram and the inference histogram all see both calls."""
    for _ in range(2):
        assert client.post("/predict", json=BODY).status_code == 200

    text = client.get("/metrics").text
    assert sample(text, "http_requests_total", method="POST", route="/predict", status="200") == 2
    assert sample(text, "http_request_duration_seconds_count", method="POST", route="/predict") == 2
    assert sample(text, "model_inference_duration_seconds_count", model_version="7") == 2
    # And the model call is inside the request, not longer than it.
    assert sample(text, "model_inference_duration_seconds_sum", model_version="7") <= sample(
        text, "http_request_duration_seconds_sum", method="POST", route="/predict"
    )
    assert (
        sample(text, "model_predictions_total", model_version="7", unknown_archetype="false") == 2
    )


def test_an_unseen_archetype_is_counted_apart(client: TestClient) -> None:
    client.post("/predict", json={**BODY, "opponent_archetype_key": "name:omega"})
    text = client.get("/metrics").text
    assert sample(text, "model_predictions_total", model_version="7", unknown_archetype="true") == 1


def test_the_route_label_is_the_template_not_the_path(client: TestClient) -> None:
    """A scanner must not be able to add a time series per URL it tries."""
    client.get("/nothing-here")
    client.get("/also/not/here")
    text = client.get("/metrics").text
    assert sample(text, "http_requests_total", method="GET", route="unmatched", status="404") == 2
    assert "/nothing-here" not in text


def test_a_failed_reload_clears_the_info_gauge() -> None:
    """Nothing loaded is no `model_info` series, rather than a stale one at 1."""

    def broken() -> serve.LoadedModel:
        raise RuntimeError("registry is down")

    with TestClient(serve.create_app(broken)) as started:
        assert started.post("/reload").status_code == 503
        assert "model_info{" not in started.get("/metrics").text


def test_reload_moves_the_info_gauge_to_the_new_version(
    exporter: InMemorySpanExporter,
) -> None:
    """One series, not two: a promotion replaces the label set rather than adding one."""
    registry = Registry(loaded(StubPredictor(0.73), "7"))
    with TestClient(serve.create_app(registry, span_exporter=exporter)) as started:
        registry.model = loaded(StubPredictor(0.19), "8")
        assert started.post("/reload").status_code == 200

        text = started.get("/metrics").text
        assert sample(text, "model_info", version="8") == 1
        assert text.count("model_info{") == 1


def test_predict_emits_an_inference_span_with_its_attributes(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    client.post("/predict", json={**BODY, "opponent_archetype_key": "name:omega"})
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert telemetry.INFERENCE_SPAN in spans, sorted(spans)

    inference = spans[telemetry.INFERENCE_SPAN]
    assert inference.attributes is not None
    assert inference.attributes["model.version"] == "7"
    assert inference.attributes["model.alias"] == "production"
    assert inference.attributes["features.unknown_archetypes"] == 1
    # And it is a child of the HTTP span, not a root of its own: a span that is
    # not in the request's trace cannot be found from the slow request.
    http = spans["POST /predict"]
    assert inference.parent is not None
    assert inference.parent.span_id == http.context.span_id


def test_health_and_metrics_are_not_traced(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """The two highest-volume requests a quiet service gets are excluded.

    A liveness probe every second and a scrape every fifteen would otherwise be
    almost the whole trace store, and neither has ever been worth reading.
    """
    client.get("/health")
    client.get("/metrics")
    assert exporter.get_finished_spans() == ()

    client.post("/predict", json=BODY)
    assert [span.name for span in exporter.get_finished_spans()] != []


def test_startup_without_a_collector_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """The laptop case: no `OTEL_EXPORTER_OTLP_ENDPOINT`, no collector, no error.

    An exporter that retried a connection to nothing on every request would make
    the telemetry the reason the service is unhealthy, which is the one thing it
    must never be.
    """
    monkeypatch.delenv(telemetry.OTLP_ENDPOINT_VAR, raising=False)
    with TestClient(serve.create_app(Registry(loaded(StubPredictor(0.5), "1")))) as started:
        assert started.get("/health").json()["model_loaded"] is True
        assert started.post("/predict", json=BODY).status_code == 200
        assert started.get("/metrics").status_code == 200


def test_an_endpoint_that_resolves_is_an_exporting_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a collector at a host that exists, the provider is the real SDK.

    `localhost` rather than a compose service name, because nothing needs to be
    listening for this: the question is whether the endpoint names a host on
    this network, and the answer decides whether spans are recorded at all.
    """
    monkeypatch.setenv(telemetry.OTLP_ENDPOINT_VAR, "http://localhost:4318")
    app = serve.create_app(Registry(loaded(StubPredictor(0.5), "1")))
    assert isinstance(app.state.tracer_provider, telemetry.TracerProvider)
    resource = app.state.tracer_provider.resource.attributes
    assert resource["service.name"] == telemetry.SERVICE_NAME
    assert resource["service.version"]
    app.state.tracer_provider.shutdown()


def test_an_endpoint_that_does_not_resolve_is_the_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `docker compose up predict` case: the profile is off and there is no collector.

    Left to the exporter this is a warning per retry per batch for as long as
    the service runs, which is a log full of the telemetry failing to leave.
    """
    monkeypatch.setenv(telemetry.OTLP_ENDPOINT_VAR, "http://otel-collector.invalid.example:4318")
    assert telemetry.resolves("http://otel-collector.invalid.example:4318") is False
    with TestClient(serve.create_app(Registry(loaded(StubPredictor(0.5), "1")))) as started:
        assert started.post("/predict", json=BODY).status_code == 200


def test_the_agent_counter_exists_before_the_agent_does(
    exporter: InMemorySpanExporter,
) -> None:
    """Stage 6 will call this; the name is fixed now so the panel is not invented later."""
    app = serve.create_app(Registry(loaded(StubPredictor(0.73), "7")), span_exporter=exporter)
    app.state.metrics.count_tool_call("sql")
    with TestClient(app) as started:
        assert sample(started.get("/metrics").text, "agent_tool_calls_total", tool="sql") == 1
