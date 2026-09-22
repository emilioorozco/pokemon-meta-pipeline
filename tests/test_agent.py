"""The agent: the SQL gate on its own, then the whole loop over a real warehouse.

Two suites in one file, split by what they cost.

The fast half is `validate_sql` and the prompt. `validate_sql` is a pure
function over a string, so every refusal is one assertion with no warehouse, no
model and no network behind it, and that is the half worth the most: it is the
code that stands between a language model and a database.

The slow half is marked `dbt`, because it needs the warehouse the fixture
silver builds. It runs the real agent loop, the real `StructuredTool` and real
DuckDB, with a scripted chat model in the one slot that would otherwise need an
API key. Nothing here is mocked except the decision about which SQL to write.
"""

from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pipeline import agent
from pipeline.prompts import ALLOWED_TABLES, MAX_PROMPT_CHARS, render_schema, system_prompt
from pipeline.telemetry import ServiceMetrics, build_metrics, build_tracer_provider
from tests.agent_fakes import ScriptedChatModel, final, scripted, tool_call

MATCHUP_SQL = (
    "select archetype_name, opponent_archetype_name, games, wins, win_rate, min_games_met "
    "from mart_matchups order by games desc"
)


# ------------------------------------------------------------------ fast --


def test_a_plain_select_over_an_allowed_table_is_accepted() -> None:
    assert agent.validate_sql("select games from mart_matchups") is None
    assert agent.validate_sql("SELECT * FROM dim_archetype LIMIT 3") is None
    assert (
        agent.validate_sql(
            "with recent as (select * from mart_archetype_weekly) select * from recent"
        )
        is None
    )
    assert (
        agent.validate_sql(
            "select m.archetype_name, c.card_name from mart_cards_seen c "
            "join mart_matchups m on m.archetype_key = c.archetype_key"
        )
        is None
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("drop table mart_matchups", "DROP"),
        ("attach 'other.duckdb' as other", "ATTACH"),
        ("install vss", "INSTALL"),
        ("pragma database_list", "PRAGMA"),
        ("update mart_matchups set games = 0", "UPDATE"),
        ("", "empty"),
    ],
)
def test_a_statement_that_is_not_a_select_is_refused_by_name(sql: str, expected: str) -> None:
    """The refusal names the rule, because its reader has to write a better query."""
    refusal = agent.validate_sql(sql)
    assert refusal is not None
    assert expected in refusal


def test_a_write_hidden_after_a_select_is_still_refused() -> None:
    """The leading keyword is not the whole check; the keyword list is the rest."""
    refusal = agent.validate_sql("select 1 from mart_matchups where 1 = (delete from dim_card)")
    assert refusal is not None
    assert "DELETE" in refusal


def test_reading_a_file_is_refused_and_the_tables_are_named() -> None:
    refusal = agent.validate_sql("select * from read_parquet('/etc/passwd')")
    assert refusal is not None
    assert "read_parquet" in refusal
    assert "mart_matchups" in refusal


def test_the_fact_and_the_player_dimension_are_not_readable() -> None:
    """The two exclusions that are the point of the allowlist, asserted separately.

    `fct_game_side` is off it because the marts aggregate it correctly and a
    model writing its own group-by over a two-rows-per-game fact double counts.
    `dim_player` is off it because it is the one table keyed by a person, which
    is a privacy boundary rather than a modelling preference.
    """
    for table in ("fct_game_side", "dim_player", "stg_games", "features_turn"):
        refusal = agent.validate_sql(f"select * from {table}")
        assert refusal is not None, table
        assert table in refusal


def test_two_statements_are_refused_even_when_both_would_be_allowed() -> None:
    refusal = agent.validate_sql("select 1 from mart_matchups; select 2 from dim_card")
    assert refusal is not None
    assert "one statement" in refusal
    # A trailing semicolon on a single statement is ordinary and is not a refusal.
    assert agent.validate_sql("select 1 from mart_matchups;") is None


def test_a_keyword_inside_a_string_or_a_comment_is_not_a_keyword() -> None:
    """The checks run over the statement with literals and comments blanked out.

    Without this an archetype called "Drop Bear" would be unqueryable, which is
    the kind of false refusal that teaches a model to stop using the tool.
    """
    assert agent.validate_sql("select * from dim_archetype where archetype_name = 'Drop Bear'") is (
        None
    )
    assert agent.validate_sql("-- drop everything\nselect 1 from mart_matchups") is None
    assert agent.validate_sql("select 1 from mart_matchups /* delete this later */") is None


def test_a_limit_is_appended_when_there_is_none_and_capped_when_it_is_too_big() -> None:
    assert agent.with_limit("select 1 from mart_matchups").endswith(f"LIMIT {agent.DEFAULT_LIMIT}")
    assert agent.with_limit("select 1 from mart_matchups limit 5") == (
        "select 1 from mart_matchups limit 5"
    )
    capped = agent.with_limit("select 1 from mart_matchups limit 5000")
    assert capped.endswith(f"LIMIT {agent.MAX_LIMIT}")


def test_a_result_is_rendered_as_a_table_a_model_can_read() -> None:
    rendered = agent.markdown_table(["name", "games"], [["Alpha", 12], [None, 3]])
    lines = rendered.splitlines()
    assert lines[0] == "| name | games |"
    assert lines[2] == "| Alpha | 12 |"
    # A null is an empty cell, not the string "None".
    assert lines[3] == "|  | 3 |"


def test_the_prompt_describes_every_allowed_table_and_nothing_else() -> None:
    """The schema half is generated, so this asserts it stayed in step with the models."""
    prompt = system_prompt()
    for table in ALLOWED_TABLES:
        assert f"\n{table}:" in f"\n{prompt}", table
    for forbidden in ("fct_game_side", "dim_player", "stg_games"):
        assert f"\n{forbidden}:" not in f"\n{prompt}", forbidden
    # Columns come from schema.yml too, not from a list typed out here.
    assert "min_games_met" in prompt
    assert "opponent_archetype_name" in prompt


def test_the_prompt_carries_the_three_rules_that_keep_it_honest() -> None:
    prompt = system_prompt()
    assert "seen_rate" in prompt
    assert "not a deck inclusion rate" in prompt
    assert "min_games_met" in prompt
    assert "sample size" in prompt


def test_the_prompt_fits_its_budget() -> None:
    """A generated prompt can grow silently; this is the thing that notices."""
    assert len(system_prompt()) < MAX_PROMPT_CHARS
    assert len(system_prompt(with_card_tool=True)) < MAX_PROMPT_CHARS


def test_the_card_tool_is_described_only_when_the_agent_has_it() -> None:
    assert "lookup_cards" not in system_prompt()
    assert "lookup_cards" in system_prompt(with_card_tool=True)


def test_a_table_the_schema_file_does_not_describe_is_left_out_rather_than_raised_on() -> None:
    assert render_schema(("mart_matchups", "no_such_model")).count("\n\n") == 0


def test_a_query_against_a_warehouse_that_is_not_built_says_so(tmp_path: Path) -> None:
    text, rows = agent.run_marts_query(
        "select 1 from mart_matchups", warehouse=tmp_path / "missing.duckdb"
    )
    assert rows == 0
    assert "pipeline.gold" in text


# ------------------------------------------------------------------- dbt --


@pytest.fixture
def metrics() -> ServiceMetrics:
    """A private Prometheus registry per test, so the counters start at zero."""
    return build_metrics()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def attribute(span: ReadableSpan, name: str) -> int:
    """One integer span attribute, narrowed from the union the SDK types it as."""
    assert span.attributes is not None
    return int(str(span.attributes[name]))


def counter_value(metrics: ServiceMetrics, tool: str) -> float:
    value = metrics.registry.get_sample_value("agent_tool_calls_total", {"tool": tool})
    return float(value or 0.0)


def build(
    model: ScriptedChatModel,
    warehouse: Path,
    metrics: ServiceMetrics,
    exporter: InMemorySpanExporter,
    **extra: Any,
) -> agent.Agent:
    """The real agent, with the scripted model and an in-memory span exporter."""
    tracer = build_tracer_provider(exporter=exporter).get_tracer("tests")
    return agent.build_agent(
        model=model, warehouse=warehouse, tracer=tracer, metrics=metrics, **extra
    )


@pytest.mark.dbt
def test_the_tool_runs_a_real_query_against_the_real_marts(gold_from_fixtures: Path) -> None:
    text, rows = agent.run_marts_query(MATCHUP_SQL, warehouse=gold_from_fixtures)
    assert rows > 0
    assert rows <= agent.DEFAULT_LIMIT
    assert text.splitlines()[0].startswith("| archetype_name |")
    assert text.rstrip().endswith("row(s).")


@pytest.mark.dbt
def test_the_tool_refuses_the_forbidden_tables_against_the_real_warehouse(
    gold_from_fixtures: Path,
) -> None:
    """The refusal happens before DuckDB, on a warehouse where the table does exist.

    That is the whole claim: `fct_game_side` and `dim_player` are built and
    queryable, and the tool still will not read them.
    """
    for table in ("fct_game_side", "dim_player"):
        text, rows = agent.run_marts_query(f"select * from {table}", warehouse=gold_from_fixtures)
        assert rows == 0
        assert "refused" in text


@pytest.mark.dbt
def test_a_scripted_run_answers_a_matchup_with_the_games_count(
    gold_from_fixtures: Path, metrics: ServiceMetrics, exporter: InMemorySpanExporter
) -> None:
    """The real loop: a scripted tool call, a real query, and an answer built from it.

    The number in the answer is read out of the warehouse by the test and by
    the tool independently, so a tool that silently returned nothing would fail
    here rather than pass with an empty table.
    """
    import duckdb

    connection = duckdb.connect(str(gold_from_fixtures), read_only=True)
    row = connection.sql(
        "select archetype_name, opponent_archetype_name, games from mart_matchups "
        "order by games desc, matchup_key limit 1"
    ).fetchone()
    connection.close()
    assert row is not None
    archetype, opponent, games = str(row[0]), str(row[1]), int(row[2])

    model = scripted(
        tool_call("query_marts", "call-1", sql=MATCHUP_SQL),
        final(f"{archetype} is {games} games against {opponent} in this corpus."),
    )
    answer = build(model, gold_from_fixtures, metrics, exporter).ask(
        f"how does {archetype} do against {opponent}"
    )

    assert str(games) in answer.answer
    assert [call.tool for call in answer.tool_calls] == ["query_marts"]
    assert answer.tool_calls[0].rows > 0
    assert answer.model == "scripted-fake"
    # The tool's table really came back to the model, rather than the script
    # simply running to its end regardless.
    last_turn = model.seen[-1]
    assert any("| archetype_name |" in str(message.content) for message in last_turn)


@pytest.mark.dbt
def test_a_tool_call_is_a_span_and_a_counter(
    gold_from_fixtures: Path, metrics: ServiceMetrics, exporter: InMemorySpanExporter
) -> None:
    model = scripted(
        tool_call("query_marts", "call-1", sql="select games from mart_matchups"),
        final("done"),
    )
    build(model, gold_from_fixtures, metrics, exporter).ask("anything")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert agent.ANSWER_SPAN in spans
    tool_span = spans[f"{agent.TOOL_SPAN_PREFIX}{agent.SQL_TOOL}"]
    assert attribute(tool_span, "agent.sql.length") > 0
    assert attribute(tool_span, "agent.rows") > 0
    assert attribute(spans[agent.ANSWER_SPAN], "agent.tool_calls") == 1
    assert counter_value(metrics, agent.SQL_TOOL) == 1.0


@pytest.mark.dbt
def test_a_refused_query_comes_back_to_the_model_rather_than_raising(
    gold_from_fixtures: Path, metrics: ServiceMetrics, exporter: InMemorySpanExporter
) -> None:
    """A refusal is a turn of the conversation, not an exception out of the loop.

    The model that wrote a bad query is the only thing that can write a better
    one, so it has to see why it was refused.
    """
    model = scripted(
        tool_call("query_marts", "call-1", sql="select * from dim_player"),
        tool_call("query_marts", "call-2", sql="select games from mart_matchups"),
        final("I cannot see players, so here is the matchup instead."),
    )
    answer = build(model, gold_from_fixtures, metrics, exporter).ask("who is the best player")

    assert [call.rows for call in answer.tool_calls] == [0, answer.tool_calls[1].rows]
    assert counter_value(metrics, agent.SQL_TOOL) == 2.0
    refusal_turn = model.seen[1]
    assert any("refused" in str(message.content) for message in refusal_turn)
