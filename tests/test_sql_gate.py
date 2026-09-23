"""The optional SQL gate: the request it sends, the answers it reads, and the failures.

Every test here fakes the HTTP layer and nothing else. `JevGate` takes its
`post` function as an argument, so the tests pass one that asserts the request
and returns a recorded reply, and the request building, the parsing, the
threshold, the retry, the pricing and the fail-closed default are all covered
with no key, no network and no bill. That matters more than usual here: this
project has no Jev key yet, the live check is one command in
`docs/sql-gate.md`, and everything that can be pinned without it is pinned
here.

The replies below are the shapes the vendors publish, not invented ones: an
`answers` map keyed by the caller's question id, a Choice answer carrying
`choice`, `probabilities` and `confidence`, and a `usage` object of
`input_tokens`, `output_tokens` and, on OpenRouter, `cost`. The one thing no
fixture can prove is that the live service really answers in that shape, which
is what the live command is for.
"""

import json
from typing import Any, Final

import pytest

from pipeline import sql_gate
from pipeline.sql_gate import (
    GateCallError,
    GateConfigError,
    HttpReply,
    OffGate,
    OpenRouterJevGate,
    TypeSafeJevGate,
)

SCHEMA: Final = "mart_matchups:\n  games - how many games"
QUESTION: Final = "how does Dragapult control do against Gholdengo"
SQL: Final = "select archetype_name, games from mart_matchups limit 5"


def answered(choice: str, confidence: float, **usage: Any) -> dict[str, Any]:
    """A Decisions response with one Choice answer, in the published shape."""
    return {
        "id": "gen-dec-test",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            sql_gate.QUESTION_ID: {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {"allow": confidence, "refuse": 1 - confidence},
            }
        },
        "usage": {"input_tokens": 476, "output_tokens": 0, **usage},
    }


class Wire:
    """A fake `post_json`: records what was sent, replies with what it was given."""

    def __init__(self, *replies: HttpReply | OSError) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpReply:
        self.sent.append((url, headers, payload, timeout))
        reply = self.replies.pop(0) if self.replies else HttpReply(200, {})
        if isinstance(reply, OSError):
            raise reply
        return reply


def gate(*replies: HttpReply | OSError, **options: Any) -> tuple[OpenRouterJevGate, Wire]:
    wire = Wire(*replies)
    return OpenRouterJevGate(api_key="not-a-real-key", post=wire, **options), wire


# ---------------------------------------------------------------- the off --


def test_the_off_gate_allows_everything_and_reports_no_gate() -> None:
    """The shape of a run with the flag unset, which is every run today."""
    decision = OffGate().judge(QUESTION, "select 1 from mart_matchups", SCHEMA)
    assert decision.allowed
    assert decision.gate == sql_gate.GATE_OFF
    assert decision.label == "off"
    assert decision.cost_usd == 0.0
    assert decision.ran is False


# ------------------------------------------------------------ the request --


def test_the_request_is_one_choice_question_at_the_decisions_endpoint() -> None:
    """The contract with the provider, asserted field by field.

    OpenRouter serves this model at `/api/alpha/decisions` and not at
    `/chat/completions`, the body is `{model, state, questions}`, and a Choice
    question's options live under `criteria` rather than `options`. Each one of
    those is a thing this project got wrong at least once while reading the
    documentation, so each one is a line here.
    """
    judge, wire = gate(HttpReply(200, answered("allow", 0.91)))
    judge.judge(QUESTION, SQL, SCHEMA)

    url, headers, payload, timeout = wire.sent[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"].startswith("Bearer ")
    assert timeout == sql_gate.DEFAULT_TIMEOUT_S
    assert payload["model"] == "typesafe/jev-1.13"
    question = payload["questions"][sql_gate.QUESTION_ID]
    assert question["type"] == "choice"
    assert question["instructions"] == sql_gate.CHOICE_INSTRUCTIONS
    assert set(question["criteria"]) == {"allow", "refuse"}
    # The state is the three things the gate judges, and nothing else.
    assert QUESTION in payload["state"]
    assert SQL in payload["state"]
    assert "mart_matchups" in payload["state"]
    # It is JSON, which the wire never got to find out for itself.
    assert json.loads(json.dumps(payload)) == payload


def test_the_direct_provider_differs_in_the_url_and_the_model_and_nothing_else() -> None:
    """One client parameterised by provider: the same body goes to both."""
    openrouter, wire_one = gate(HttpReply(200, answered("allow", 0.91)))
    wire_two = Wire(HttpReply(200, answered("allow", 0.91)))
    typesafe = TypeSafeJevGate(api_key="not-a-real-key", post=wire_two)

    openrouter.judge(QUESTION, SQL, SCHEMA)
    typesafe.judge(QUESTION, SQL, SCHEMA)

    assert wire_two.sent[0][0] == "https://api.typesafe.ai/v1/systemone"
    assert wire_two.sent[0][2]["model"] == "jev-latest"
    assert wire_one.sent[0][2]["state"] == wire_two.sent[0][2]["state"]
    assert wire_one.sent[0][2]["questions"] == wire_two.sent[0][2]["questions"]


def test_the_base_url_and_the_model_can_be_pointed_somewhere_else() -> None:
    """`JEV_BASE_URL` is what makes an alpha path survivable, and a proxy possible."""
    wire = Wire(HttpReply(200, answered("allow", 0.91)))
    judge = OpenRouterJevGate(
        api_key="not-a-real-key",
        base_url="https://gateway.example.com/jev/",
        model="typesafe/jev-1.13-20260917",
        post=wire,
    )
    judge.judge(QUESTION, SQL, SCHEMA)
    assert wire.sent[0][0] == "https://gateway.example.com/jev/alpha/decisions"
    assert wire.sent[0][2]["model"] == "typesafe/jev-1.13-20260917"


def test_the_key_is_not_in_the_repr() -> None:
    """A gate in a traceback or a debugger must not be a leaked key."""
    judge, _ = gate()
    assert "not-a-real-key" not in repr(judge)
    assert "typesafe/jev-1.13" in repr(judge)


# ------------------------------------------------------------- the answer --


def test_an_allow_above_the_threshold_is_an_allow_and_carries_its_cost() -> None:
    judge, _ = gate(HttpReply(200, answered("allow", 0.91, cost=0.000_019_992)))
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert decision.allowed
    assert decision.confidence == 0.91
    assert decision.input_tokens == 476
    # OpenRouter prices the call itself, so that number is used rather than a
    # second calculation of our own.
    assert decision.cost_usd == 0.000_019_992
    assert decision.label == "jev:allowed"
    assert decision.errored is False


def test_a_refuse_is_a_refusal_whatever_the_confidence_is() -> None:
    judge, _ = gate(HttpReply(200, answered("refuse", 0.99)))
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert not decision.allowed
    assert decision.label == "jev:refused"
    assert "did not read this" in decision.reason


def test_an_allow_under_the_threshold_is_refused_and_says_so() -> None:
    """The threshold is the whole of what this gate decides with."""
    judge, _ = gate(HttpReply(200, answered("allow", 0.55)), threshold=0.7)
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert not decision.allowed
    assert decision.label == "jev:refused"
    assert "0.55" in decision.reason and "0.70" in decision.reason
    # The same answer passes a gate that was told to be less careful.
    lenient, _ = gate(HttpReply(200, answered("allow", 0.55)), threshold=0.5)
    assert lenient.judge(QUESTION, SQL, SCHEMA).allowed


def test_the_cost_falls_back_to_the_published_rate_when_nothing_prices_the_call() -> None:
    """TypeSafe's own `usage` has no `cost`, so the input tokens are priced here."""
    wire = Wire(HttpReply(200, answered("allow", 0.91)))
    decision = TypeSafeJevGate(api_key="not-a-real-key", post=wire).judge(QUESTION, SQL, SCHEMA)
    assert decision.cost_usd == pytest.approx(476 * sql_gate.PRICE_PER_INPUT_TOKEN)
    # A few hundred tokens at $0.042 per million is a fraction of a cent, which
    # is the arithmetic the whole feature rests on.
    assert decision.cost_usd < 0.0001


# ------------------------------------------------------------- the failures --


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({}, "no answer"),
        ({"answers": {}}, "no answer"),
        ({"answers": {sql_gate.QUESTION_ID: "allow"}}, "not an object"),
        (
            {"answers": {sql_gate.QUESTION_ID: {"type": "choice", "choice": "maybe"}}},
            "neither option",
        ),
        (
            {"answers": {sql_gate.QUESTION_ID: {"type": "choice", "choice": "allow"}}},
            "no confidence",
        ),
        (
            {
                "answers": {
                    sql_gate.QUESTION_ID: {
                        "type": "choice",
                        "choice": "allow",
                        "confidence": "high",
                    }
                }
            },
            "no confidence",
        ),
    ],
)
def test_an_answer_that_cannot_be_read_is_an_error_and_therefore_a_refusal(
    body: dict[str, Any], expected: str
) -> None:
    """Including a missing `confidence`, which the published schema allows.

    There is no safe default for it. Taking an absent confidence as certainty
    would let a response shape change turn the gate off without anyone
    noticing, which is the failure a safety check is least allowed to have.
    """
    judge, _ = gate(HttpReply(200, body))
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert not decision.allowed
    assert decision.errored
    assert decision.label == "jev:error"
    assert expected in decision.reason


def test_a_bad_status_is_an_error_rather_than_an_exception() -> None:
    judge, _ = gate(HttpReply(401, {"error": {"code": 401, "message": "Missing Authentication"}}))
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert decision.errored
    assert "HTTP 401" in decision.reason


def test_a_server_error_is_retried_once_and_then_given_up_on() -> None:
    judge, wire = gate(HttpReply(503, {}), HttpReply(200, answered("allow", 0.95)))
    assert judge.judge(QUESTION, SQL, SCHEMA).allowed
    assert len(wire.sent) == 2

    judge, wire = gate(HttpReply(500, {}), HttpReply(500, {}))
    assert judge.judge(QUESTION, SQL, SCHEMA).errored
    assert len(wire.sent) == 2


def test_a_rate_limit_is_retried_and_a_bad_request_is_not() -> None:
    """A 429 is worth asking again; a 400 is the same answer a second time."""
    judge, wire = gate(HttpReply(429, {}), HttpReply(200, answered("allow", 0.95)))
    assert judge.judge(QUESTION, SQL, SCHEMA).allowed
    assert len(wire.sent) == 2

    judge, wire = gate(HttpReply(400, {}), HttpReply(200, answered("allow", 0.95)))
    assert judge.judge(QUESTION, SQL, SCHEMA).errored
    assert len(wire.sent) == 1


def test_a_network_failure_is_retried_and_then_refuses_by_default() -> None:
    judge, wire = gate(TimeoutError("timed out"), TimeoutError("timed out"))
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert len(wire.sent) == 2
    assert not decision.allowed
    assert decision.errored
    assert "could not be reached" in decision.reason


def test_on_error_allow_lets_the_query_through_and_still_says_it_errored() -> None:
    """The escape hatch, and the reason it is not silent: the label is still an error."""
    judge, _ = gate(TimeoutError("timed out"), TimeoutError("timed out"), on_error="allow")
    decision = judge.judge(QUESTION, SQL, SCHEMA)
    assert decision.allowed
    assert decision.errored
    assert decision.label == "jev:error"
    assert "set to allow" in decision.reason


def test_a_body_that_is_not_json_at_all_reads_as_an_empty_one() -> None:
    assert sql_gate._decode(b"<html>502 Bad Gateway</html>") == {}
    assert sql_gate._decode(b"[1, 2, 3]") == {}
    assert sql_gate._decode(b'{"answers": {}}') == {"answers": {}}


def test_reading_an_answer_raises_the_error_the_gate_catches() -> None:
    """The parser on its own, so its message is asserted where it is produced."""
    with pytest.raises(GateCallError, match="no answer"):
        sql_gate._read_answer({})


# ------------------------------------------------------------ the factory --


def test_the_factory_is_off_unless_something_turns_it_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("", "off", "OFF", " off "):
        monkeypatch.setenv(sql_gate.GATE_VAR, value)
        assert isinstance(sql_gate.gate_from_env(), OffGate)
    monkeypatch.delenv(sql_gate.GATE_VAR, raising=False)
    assert isinstance(sql_gate.gate_from_env(), OffGate)


def test_the_factory_builds_openrouter_by_default_and_typesafe_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(sql_gate.GATE_VAR, "jev")
    monkeypatch.setenv(sql_gate.API_KEY_VAR, "not-a-real-key")
    monkeypatch.delenv(sql_gate.PROVIDER_VAR, raising=False)
    monkeypatch.delenv(sql_gate.MODEL_VAR, raising=False)
    monkeypatch.delenv(sql_gate.BASE_URL_VAR, raising=False)
    monkeypatch.delenv(sql_gate.THRESHOLD_VAR, raising=False)
    monkeypatch.delenv(sql_gate.ON_ERROR_VAR, raising=False)

    built = sql_gate.gate_from_env()
    assert isinstance(built, OpenRouterJevGate)
    assert built.name == "jev"
    assert built.model == "typesafe/jev-1.13"
    assert built.threshold == sql_gate.DEFAULT_THRESHOLD
    assert built.on_error == "refuse"

    monkeypatch.setenv(sql_gate.PROVIDER_VAR, "typesafe")
    monkeypatch.setenv(sql_gate.THRESHOLD_VAR, "0.85")
    monkeypatch.setenv(sql_gate.ON_ERROR_VAR, "allow")
    monkeypatch.setenv(sql_gate.BASE_URL_VAR, "https://proxy.example.com")
    direct = sql_gate.gate_from_env()
    assert isinstance(direct, TypeSafeJevGate)
    assert direct.base_url == "https://proxy.example.com"
    assert direct.threshold == 0.85
    assert direct.on_error == "allow"


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({sql_gate.GATE_VAR: "maybe"}, "has to be off or jev"),
        ({sql_gate.GATE_VAR: "jev"}, "needs \\$JEV_API_KEY"),
        (
            {sql_gate.GATE_VAR: "jev", sql_gate.API_KEY_VAR: "k", sql_gate.PROVIDER_VAR: "azure"},
            "has to be one of",
        ),
        (
            {sql_gate.GATE_VAR: "jev", sql_gate.API_KEY_VAR: "k", sql_gate.THRESHOLD_VAR: "high"},
            "not a number",
        ),
        (
            {sql_gate.GATE_VAR: "jev", sql_gate.API_KEY_VAR: "k", sql_gate.THRESHOLD_VAR: "1.5"},
            "0 to 1",
        ),
        (
            {sql_gate.GATE_VAR: "jev", sql_gate.API_KEY_VAR: "k", sql_gate.ON_ERROR_VAR: "shrug"},
            "has to be",
        ),
    ],
)
def test_a_gate_that_cannot_be_built_says_which_variable_is_wrong(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: str
) -> None:
    """A misspelt flag must not be a run that quietly had no gate."""
    for name in (
        sql_gate.GATE_VAR,
        sql_gate.API_KEY_VAR,
        sql_gate.PROVIDER_VAR,
        sql_gate.THRESHOLD_VAR,
        sql_gate.ON_ERROR_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(GateConfigError, match=expected):
        sql_gate.gate_from_env()


def test_the_schema_the_gate_is_told_is_the_one_the_prompt_carries() -> None:
    """One renderer, so a column rename cannot make the gate refuse a good query."""
    summary = sql_gate.schema_summary()
    for table in sql_gate.ALLOWED_TABLES:
        assert f"{table}:" in summary
    assert "dim_player" not in summary
