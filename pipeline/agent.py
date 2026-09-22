"""The agent: a tool-calling loop over the gold marts, and nothing else.

    uv run python -m pipeline.agent "how does Dragapult ex do against Gholdengo ex"
    uv run python -m pipeline.agent --repl

One model, two tools and a read-only DuckDB connection. `query_marts` runs a
single SELECT against the warehouse the dbt stage built; `lookup_cards`, when
`pipeline.card_index` has an index to load, searches the text of printed cards.
`POST /ask` on the serving application is the same loop over HTTP.

Five choices worth knowing before reading the code.

**The tool is a gate, not a wrapper.** A language model writing SQL against a
warehouse is a useful thing and an obvious hazard, so the SQL it writes goes
through `validate_sql` before DuckDB ever sees it: one statement, and that
statement a `SELECT` or a `WITH`; every table it names on the allowlist; no
`ATTACH`, `COPY`, `INSTALL`, `LOAD`, `PRAGMA`, `SET`, and none of the file
readers (`read_parquet`, `read_csv`, `glob`) that would otherwise turn a
read-only connection into a filesystem. The connection is opened `read_only`
as well, which is the second lock rather than the first: `read_only` would
still allow `read_csv('/etc/passwd')`, and the allowlist is what does not.

`validate_sql` is a pure function over a string and a set of table names. It
takes no connection and returns the refusal as a string, so the interesting
half of this module is unit tested in the fast suite with no warehouse, no
model and no network, and every refusal names the rule it broke rather than
saying "invalid query".

**The allowlist is about privacy as well as correctness.** `dim_player`, the
member roster, is not on it, and neither is anything from silver or staging, so
nothing the agent can read carries a handle or a raw log line
(docs/data-handling.md). What it can read is `mart_player_summary`, an
aggregate over members keyed by the same irreversible token the rest of the
pipeline uses; rule 5 of the prompt is what keeps the agent from presenting
such a token as a person. `fct_game_side` is off the list for a duller reason:
the marts aggregate it correctly, and a model writing its own group-by over a
two-rows-per-game fact is where double counting starts.

**The prompt is generated.** `pipeline.prompts` renders the mart schemas out of
`dbt/models/marts/schema.yml` at import, so the column list in the prompt is
the column list dbt tests. The rules beside it are the ones this corpus needs:
cite `games`, flag `min_games_met`, and never let `seen_rate` be reported as a
deck inclusion rate.

**Every tool call is a span and a counter increment.** `agent.tool.query_marts`
carries the SQL length and the row count, `agent.answer` wraps the run with the
model, the number of tool calls and the token usage the provider reported. The
counter is `agent_tool_calls_total{tool=...}`, which `pipeline.telemetry`
declared while the serving instrumentation was being built. The SQL itself is
on the span as a length rather than as text: a query is short and harmless
here, but a span attribute is the wrong place to start putting model output.

**The model is injected.** `build_agent` takes a chat model, so the tests pass
a scripted fake that returns pre-written `AIMessage`s with `tool_calls` on them
and the real agent loop, the real tool and the real warehouse run underneath
it. Nothing in the test suite needs an API key, and the live path is the same
code with `ChatAnthropic` in the same slot.
"""

import argparse
import contextlib
import json
import logging
import os
import re
import sys
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import duckdb
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from opentelemetry import trace

from pipeline.config import WAREHOUSE_PATH
from pipeline.observability import configure_logging, emit_summary
from pipeline.prompts import ALLOWED_TABLES, system_prompt
from pipeline.telemetry import ServiceMetrics, build_metrics, build_tracer_provider

logger = logging.getLogger(__name__)

STAGE: Final = "agent"
SERVICE_NAME: Final = "pra-agent"

# Anthropic, and the small model, because every question here is "write one
# SELECT against seven tables and read a dozen rows back". `PRA_AGENT_MODEL`
# overrides it; the key is `ANTHROPIC_API_KEY`, injected by `op run`.
DEFAULT_MODEL: Final = "claude-haiku-4-5-20251001"
MODEL_VAR: Final = "PRA_AGENT_MODEL"
API_KEY_VAR: Final = "ANTHROPIC_API_KEY"

SQL_TOOL: Final = "query_marts"
CARD_TOOL: Final = "lookup_cards"
ANSWER_SPAN: Final = "agent.answer"
TOOL_SPAN_PREFIX: Final = "agent.tool."

# Appended when the model writes no LIMIT of its own, and the ceiling any LIMIT
# it does write is cut down to. Fifty rows is more than an answer needs and few
# enough to read; two hundred is the point past which a tool result is context
# spent rather than evidence gathered.
DEFAULT_LIMIT: Final = 50
MAX_LIMIT: Final = 200
# DuckDB cancels a statement that runs longer than this. A mart query over a
# corpus this size is milliseconds, so anything near five seconds is a mistake.
STATEMENT_TIMEOUT_S: Final = 5.0
# Turns of the loop. Each turn is one model call and one tool call, so this is
# generous for a question that needs two queries and a guard against a model
# that has decided to keep querying.
MAX_ITERATIONS: Final = 8
# Cells in the markdown table are cut to this, so one long `aliases` value
# cannot be most of the tool result.
MAX_CELL_CHARS: Final = 120

# Statement keywords that are never allowed, whatever else the string contains.
# Checked on word boundaries against the comment-stripped SQL, so a column
# called `copy_count` is not a refusal and `COPY (...) TO` is.
FORBIDDEN_KEYWORDS: Final[tuple[str, ...]] = (
    "attach",
    "detach",
    "copy",
    "install",
    "load",
    "pragma",
    "set",
    "reset",
    "export",
    "import",
    "call",
    "create",
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "grant",
    "revoke",
    "vacuum",
    "checkpoint",
)
# Table functions that read something other than the warehouse. DuckDB reaches
# the filesystem and the network through these, so they are refused by name.
FORBIDDEN_FUNCTIONS: Final[tuple[str, ...]] = (
    "read_parquet",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "parquet_scan",
    "csv_scan",
    "glob",
    "sniff_csv",
    "duckdb_extensions",
)

_LINE_COMMENT: Final = re.compile(r"--[^\n]*")
_BLOCK_COMMENT: Final = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL: Final = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER: Final = re.compile(r'"(?:[^"]|"")*"')
# Whatever follows FROM or JOIN: a bare name, a schema-qualified name, or an
# opening parenthesis for a subquery, which is not a table and is skipped.
_TABLE_REF: Final = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_.$]*)", re.IGNORECASE)
_FUNCTION_CALL: Final = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)
_LIMIT_CLAUSE: Final = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)
_LEADING_KEYWORD: Final = re.compile(r"^\s*([a-z_]+)", re.IGNORECASE)
# Names a CTE introduces, which are legal table references even though they are
# not on the allowlist: `with recent as (...) select * from recent`.
_CTE_NAME: Final = re.compile(r"(?:\bwith\b|,)\s+([a-z_][a-z0-9_]*)\s+as\s*\(", re.IGNORECASE)


# ------------------------------------------------------------ the SQL gate --


def strip_literals(sql: str) -> str:
    """The statement with comments, string literals and quoted identifiers blanked.

    Every check below is a search for a keyword or a name, and all three of
    these are places a keyword can appear without being one: `-- drop this`,
    `where archetype_name = 'Drop Bear'`, and a column deliberately quoted as
    `"drop"`. Blanking them rather than removing them keeps the offsets, so a
    refusal can still be reasoned about against the original string.
    """
    blanked = _BLOCK_COMMENT.sub(lambda match: " " * len(match.group(0)), sql)
    blanked = _LINE_COMMENT.sub(lambda match: " " * len(match.group(0)), blanked)
    blanked = _STRING_LITERAL.sub(_blank_inside, blanked)
    return _IDENTIFIER.sub(_blank_inside, blanked)


def _blank_inside(match: re.Match[str]) -> str:
    """A quoted run with its contents replaced by spaces and its quotes kept."""
    text = match.group(0)
    return text[0] + " " * (len(text) - 2) + text[-1]


def validate_sql(sql: str, allowed_tables: Sequence[str] = ALLOWED_TABLES) -> str | None:
    """The refusal this statement earns, or None when it may run.

    A pure function of a string and a list of names: no connection, no model,
    no filesystem, which is what makes the rules testable one at a time. The
    message names the rule rather than the symptom, because the reader is a
    language model that has to write a better query next, and "invalid query"
    tells it nothing it can act on.

    The order of the checks is the order of severity, so a statement that both
    drops a table and reads a Parquet file is refused for the drop.
    """
    text = strip_literals(sql).strip().rstrip(";").strip()
    if not text:
        return "refused: the query is empty."

    # One statement. A trailing semicolon is fine and has already been taken
    # off; a semicolon anywhere else means a second statement is being smuggled
    # past a check that only ever looks at the first one.
    if ";" in text:
        return (
            "refused: only one statement per call. Remove the `;` and send a single "
            "SELECT, or make two calls."
        )

    leading = _LEADING_KEYWORD.match(text)
    keyword = leading.group(1).lower() if leading else ""
    if keyword not in {"select", "with"}:
        return (
            f"refused: this tool runs read-only queries, and the statement starts with "
            f"`{keyword.upper() or text[:12]}`. It has to start with SELECT or WITH."
        )

    lowered = text.lower()
    for forbidden in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{forbidden}\b", lowered):
            return (
                f"refused: `{forbidden.upper()}` is not allowed here. This tool answers "
                f"questions with a single read-only SELECT and changes nothing."
            )

    for name in {match.group(1).lower() for match in _FUNCTION_CALL.finditer(text)}:
        if name in FORBIDDEN_FUNCTIONS:
            return (
                f"refused: `{name}(...)` reads outside the warehouse. Query the tables by "
                f"name instead: {', '.join(allowed_tables)}."
            )

    allowed = {name.lower() for name in allowed_tables}
    for table in referenced_tables(sql):
        if table in allowed:
            continue
        return (
            f"refused: `{table}` is not a table this tool can read. The tables are: "
            f"{', '.join(allowed_tables)}."
        )
    return None


def referenced_tables(sql: str) -> list[str]:
    """The tables a statement reads, in the order it names them, CTEs left out.

    Split out of `validate_sql` because two callers need the same answer. The
    validator asks whether every name is on the allowlist; `pipeline.eval`'s
    replay model asks whether the system prompt described them, since a model
    cannot query a table it was never told exists. Names the statement
    introduces itself with `WITH` are not tables and are dropped here rather
    than at each call site.
    """
    text = strip_literals(sql)
    ctes = {match.group(1).lower() for match in _CTE_NAME.finditer(text)}
    names: list[str] = []
    for match in _TABLE_REF.finditer(text):
        table = match.group(1).lower().split(".")[-1]
        if table not in ctes and table not in names:
            names.append(table)
    return names


def with_limit(sql: str, *, default: int = DEFAULT_LIMIT, cap: int = MAX_LIMIT) -> str:
    """The statement with a LIMIT on it, added or cut down to the cap.

    Appended rather than wrapped in a subquery, because a wrapper changes what
    the model sees back from what it wrote and makes an ORDER BY inside it
    unreliable. A statement whose only LIMIT is inside a CTE therefore gets a
    second one on the outside, which is correct and occasionally redundant.
    """
    text = sql.strip().rstrip(";").strip()
    found = list(_LIMIT_CLAUSE.finditer(text))
    if not found:
        return f"{text}\nLIMIT {default}"
    last = found[-1]
    if int(last.group(1)) <= cap:
        return text
    return f"{text[: last.start()]}LIMIT {cap}{text[last.end() :]}"


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """The result set as a compact markdown table.

    Markdown because the reader is a language model and a pipe table is the
    shape it has seen a million of; a JSON array of objects repeats every
    column name on every row, which on a fifty-row result is most of the
    tokens spent on punctuation.
    """
    header = f"| {' | '.join(columns)} |"
    rule = f"|{'|'.join('---' for _ in columns)}|"
    body = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def _cell(value: Any) -> str:
    """One value as a table cell: short, one line, and never an unescaped pipe."""
    if value is None:
        return ""
    text = str(value).replace("|", "/").replace("\n", " ")
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 3] + "..."


# ------------------------------------------------------------- tool calls --


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation, as the answer reports it back to a caller."""

    tool: str
    input_summary: str
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "input_summary": self.input_summary, "rows": self.rows}


# The calls made by the run happening on this task. A context variable rather
# than an attribute on the agent, because one agent object serves every request
# of a serving process and two concurrent `/ask` calls must not collect into
# the same list. Starlette copies the context per request, so each run sees its
# own.
_calls: ContextVar[list[ToolCall] | None] = ContextVar("pra_agent_calls", default=None)


def record_call(call: ToolCall) -> None:
    """Add a tool call to the run in progress, if anything is collecting them."""
    collected = _calls.get()
    if collected is not None:
        collected.append(call)


# ----------------------------------------------------------------- tools --


def open_warehouse(path: Path) -> duckdb.DuckDBPyConnection:
    """A read-only connection to the warehouse, with a statement timeout on it.

    Read-only is the cheap half of the protection and the allowlist in
    `validate_sql` is the real one, but it costs a keyword and it means a bug in
    the validator cannot write. The timeout is set through a configuration
    parameter and is tolerated if this DuckDB build does not have it: a version
    without the setting should serve queries, not refuse to start.
    """
    connection = duckdb.connect(str(path), read_only=True)
    with contextlib.suppress(duckdb.Error):
        connection.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT_S}s'")
    return connection


def run_marts_query(
    sql: str,
    *,
    warehouse: Path,
    allowed_tables: Sequence[str] = ALLOWED_TABLES,
) -> tuple[str, int]:
    """Validate, run and render one query. Returns the tool's text and the row count.

    A refusal and a failure both come back as text with a row count of zero,
    never as an exception: an exception out of a tool ends the agent's turn,
    and the useful outcome of a bad query is the model reading why and writing
    a better one.
    """
    refusal = validate_sql(sql, allowed_tables)
    if refusal is not None:
        logger.info("tool call refused", extra={"tool": SQL_TOOL, "reason": refusal})
        return refusal, 0
    limited = with_limit(sql)
    if not warehouse.is_file():
        return (
            f"the warehouse is not built: nothing at {warehouse.name}. "
            "Run `python -m pipeline.gold` first.",
            0,
        )
    connection = open_warehouse(warehouse)
    try:
        result = connection.sql(limited)
        columns = list(result.columns)
        rows = result.fetchall()
    except duckdb.Error as failure:
        # The class and the message, not a traceback: the model reads this.
        return f"the query failed: {type(failure).__name__}: {failure}", 0
    finally:
        connection.close()
    if not rows:
        return "0 rows. The query ran and matched nothing.", 0
    table = markdown_table(columns, rows)
    return f"{table}\n\n{len(rows)} row(s).", len(rows)


def make_query_marts_tool(
    *,
    warehouse: Path,
    tracer: trace.Tracer,
    metrics: ServiceMetrics,
    allowed_tables: Sequence[str] = ALLOWED_TABLES,
) -> BaseTool:
    """The SQL tool, bound to one warehouse and one set of instruments."""

    def query_marts(sql: str) -> str:
        """Run one read-only SELECT against the gold marts and return the rows."""
        with tracer.start_as_current_span(f"{TOOL_SPAN_PREFIX}{SQL_TOOL}") as span:
            span.set_attribute("agent.tool", SQL_TOOL)
            span.set_attribute("agent.sql.length", len(sql))
            answer, rows = run_marts_query(sql, warehouse=warehouse, allowed_tables=allowed_tables)
            span.set_attribute("agent.rows", rows)
        metrics.count_tool_call(SQL_TOOL)
        record_call(ToolCall(tool=SQL_TOOL, input_summary=summarize(sql), rows=rows))
        logger.info("tool call", extra={"tool": SQL_TOOL, "rows": rows, "sql_length": len(sql)})
        return answer

    return StructuredTool.from_function(
        func=query_marts,
        name=SQL_TOOL,
        description=(
            "Run one read-only SQL SELECT against the gold marts and get the rows back "
            "as a markdown table. DuckDB dialect. One statement, no semicolons, only "
            f"the tables described in the system prompt. A LIMIT of {DEFAULT_LIMIT} is "
            f"added when you leave one out and any LIMIT above {MAX_LIMIT} is reduced."
        ),
    )


def summarize(text: str, width: int = 100) -> str:
    """A tool input as one short line, for the response body and the log."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def marts_tools(
    *,
    warehouse: Path = WAREHOUSE_PATH,
    tracer: trace.Tracer,
    metrics: ServiceMetrics,
    card_index: Path | None = None,
) -> list[BaseTool]:
    """Every tool the agent gets: the SQL one always, the card one when there is an index.

    `pipeline.card_index` is imported here rather than at module scope because
    it pulls in the embedder, which pulls in torch, and an agent asked only for
    matchup numbers should not pay several seconds of import for a tool it will
    not call. A missing or unreadable index is a logged skip, not a failure: the
    SQL half of the agent works perfectly well without the card text.
    """
    tools = [
        make_query_marts_tool(warehouse=warehouse, tracer=tracer, metrics=metrics),
    ]
    if card_index is None:
        return tools
    from pipeline import card_index as index_module

    try:
        tools.append(
            index_module.make_lookup_cards_tool(card_index, tracer=tracer, metrics=metrics)
        )
    except (OSError, ValueError) as failure:
        logger.warning(
            "the card lookup tool is off",
            extra={"index": str(card_index), "error": f"{type(failure).__name__}: {failure}"},
        )
    return tools


# ----------------------------------------------------------------- agent --


@dataclass(frozen=True)
class Answer:
    """What one run produced: the text, what it called, and what it cost."""

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "model": self.model,
            "usage": dict(self.usage),
        }


def chat_model(model: str | None = None) -> BaseChatModel:
    """The provider, built from the environment. Imported late; it needs a key.

    `ChatAnthropic` reads `ANTHROPIC_API_KEY` itself and raises when it is
    missing, which is the right moment to find out: building the agent is what
    the command line does before it has a question to ask.
    """
    from langchain_anthropic import ChatAnthropic

    name = model or os.environ.get(MODEL_VAR, "").strip() or DEFAULT_MODEL
    # `model` rather than `model_name`: the field carries that alias, and it is
    # the name the provider's own documentation and its type signature use.
    return ChatAnthropic(model=name, timeout=60, stop=None)


class Agent:
    """A built agent: the graph, the tools it holds, and the name of its model.

    Built once and asked many times. `ask` is what both the command line and
    `POST /ask` call, and the only state it keeps between questions is none:
    each run starts from the system prompt and the question.
    """

    def __init__(
        self,
        graph: Any,
        *,
        model_name: str,
        tool_names: Sequence[str],
        tracer: trace.Tracer,
    ) -> None:
        self.graph = graph
        self.model_name = model_name
        self.tool_names = list(tool_names)
        # The same tracer the tools were built with, rather than the global
        # provider: `agent.answer` has to be the parent of the tool spans, and
        # a second provider would put them in two unrelated traces.
        self.tracer = tracer

    def ask(self, question: str) -> Answer:
        """Run the loop on one question and collect what it did."""
        collected: list[ToolCall] = []
        token = _calls.set(collected)
        try:
            with self.tracer.start_as_current_span(ANSWER_SPAN) as span:
                span.set_attribute("agent.model", self.model_name)
                state = self.graph.invoke({"messages": [HumanMessage(content=question)]})
                messages: list[BaseMessage] = list(state["messages"])
                usage = token_usage(messages)
                span.set_attribute("agent.tool_calls", len(collected))
                for name, value in usage.items():
                    span.set_attribute(f"agent.usage.{name}", value)
        finally:
            _calls.reset(token)
        answer = Answer(
            answer=final_text(messages),
            tool_calls=collected,
            model=self.model_name,
            usage=usage,
        )
        logger.info(
            "agent answered",
            extra={
                "model": self.model_name,
                "tool_calls": len(collected),
                "usage": usage,
                "answer_length": len(answer.answer),
            },
        )
        return answer


def final_text(messages: Sequence[BaseMessage]) -> str:
    """The last assistant message as plain text.

    A content block list rather than a string is the normal shape from a
    provider that can interleave text and tool calls, so the text blocks are
    joined rather than assumed to be the whole thing.
    """
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
        if isinstance(content, str):
            if content.strip():
                return content.strip()
            continue
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return joined
    return ""


def token_usage(messages: Sequence[BaseMessage]) -> dict[str, int]:
    """Input and output tokens summed over the run, when the provider reported any.

    Empty rather than zeroed when nothing said: a fake model reports no usage,
    and a response body that claimed zero tokens would be a wrong number rather
    than a missing one.
    """
    totals: dict[str, int] = {}
    for message in messages:
        metadata = getattr(message, "usage_metadata", None)
        if not isinstance(metadata, dict):
            continue
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = metadata.get(name)
            if isinstance(value, int):
                totals[name] = totals.get(name, 0) + value
    return totals


def build_agent(
    *,
    model: BaseChatModel | None = None,
    warehouse: Path = WAREHOUSE_PATH,
    card_index: Path | None = None,
    tracer: trace.Tracer | None = None,
    metrics: ServiceMetrics | None = None,
) -> Agent:
    """The agent, with its model, its warehouse and its instruments injected.

    Every argument has a default that is the real thing, and every one of them
    is replaceable, which is the same shape `pipeline.serve` gives its model
    loader. The tests pass a scripted chat model and a fixture warehouse and
    exercise everything else for real.
    """
    resolved_metrics = metrics if metrics is not None else build_metrics()
    resolved_tracer = tracer or build_tracer_provider(SERVICE_NAME).get_tracer(__name__)
    chat = model if model is not None else chat_model()
    tools = marts_tools(
        warehouse=warehouse,
        tracer=resolved_tracer,
        metrics=resolved_metrics,
        card_index=card_index,
    )
    graph = create_agent(
        model=chat,
        tools=tools,
        system_prompt=system_prompt(with_card_tool=any(t.name == CARD_TOOL for t in tools)),
    )
    return Agent(
        graph.with_config({"recursion_limit": MAX_ITERATIONS * 2}),
        tracer=resolved_tracer,
        model_name=getattr(chat, "model_name", None) or getattr(chat, "model", "") or "fake",
        tool_names=[tool.name for tool in tools],
    )


def default_card_index(path: Path | None) -> Path | None:
    """The card index to use, or None when there is no built index to load."""
    if path is None:
        return None
    return path if path.is_dir() else None


# ------------------------------------------------------------ entry point --


def render(answer: Answer) -> str:
    """One answer as the block the command line prints."""
    lines = [answer.answer, ""]
    for call in answer.tool_calls:
        lines.append(f"  [{call.tool}] {call.rows} row(s): {call.input_summary}")
    if answer.usage:
        counts = ", ".join(f"{name}={value}" for name, value in sorted(answer.usage.items()))
        lines.append(f"  tokens: {counts}")
    return "\n".join(lines).rstrip()


def repl(agent: Agent) -> int:
    """Ask questions until end of file. One run per line, no memory between them."""
    sys.stdout.write("Ask about the metagame. Ctrl-D to stop.\n")
    while True:
        sys.stdout.write("\n> ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            sys.stdout.write("\n")
            return 0
        question = line.strip()
        if not question:
            continue
        sys.stdout.write(render(agent.ask(question)) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Answer one question, or open a prompt that answers many."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.agent",
        description="Ask a question about the metagame; the agent queries the gold marts.",
        epilog=(
            "The provider key comes from the environment: "
            '`op run --env-file=.env.op -- uv run python -m pipeline.agent "..."`.'
        ),
    )
    parser.add_argument("question", nargs="?", default=None, help="the question to answer")
    parser.add_argument("--repl", action="store_true", help="ask questions until end of file")
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help=f"the DuckDB warehouse to read (default: {WAREHOUSE_PATH})",
    )
    parser.add_argument(
        "--card-index",
        type=Path,
        default=None,
        metavar="PATH",
        help="a built card index directory; without one the agent has only the SQL tool",
    )
    parser.add_argument(
        "--model", default=None, metavar="NAME", help=f"provider model (default: ${MODEL_VAR})"
    )
    parser.add_argument("--json", action="store_true", help="print the answer as one JSON object")
    args = parser.parse_args(argv)

    if not args.repl and not args.question:
        parser.exit(2, f"{parser.prog}: give a question, or --repl\n")
    configure_logging(STAGE)

    agent = build_agent(
        model=chat_model(args.model),
        warehouse=args.warehouse,
        card_index=default_card_index(args.card_index),
    )
    try:
        if args.repl:
            return repl(agent)
        answer = agent.ask(args.question)
    except Exception as failure:
        # A missing key, a rate limit and a provider outage all arrive here, and
        # all three are things the person running the command has to act on.
        # The class and the message go to stderr as one line; the traceback goes
        # to the log at exception level, so debugging is still possible and a
        # forgotten `op run` is not four screens of stack.
        logger.exception("the agent could not answer", extra={"model": agent.model_name})
        sys.stderr.write(f"{parser.prog}: {type(failure).__name__}: {failure}\n")
        return 1

    if args.json:
        sys.stdout.write(json.dumps(answer.as_dict(), indent=2) + "\n")
        return 0
    emit_summary(logger, "agent answer", answer.as_dict(), text=render(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
