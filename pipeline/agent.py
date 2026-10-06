"""The agent: a tool-calling loop over the gold marts, and nothing else.

    uv run python -m pipeline.agent "how does Dragapult ex do against Gholdengo ex"
    uv run python -m pipeline.agent --repl

One model, two tools and a read-only DuckDB connection. `query_marts` runs a
single SELECT against the warehouse the dbt stage built; `lookup_cards`, when
`pipeline.card_index` has an index to load, searches the text of printed cards.
`POST /ask` on the serving application is the same loop over HTTP.

Six choices worth knowing before reading the code.

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

**The question is data, and so is the page context.** `Agent.ask` hands the
model the member's text inside the `<question>` element and, when `POST /ask`
sent one, the application's description of where the member is standing inside
a `<context>` element in front of it. `pipeline.prompts.wrap_turn` builds the
pair; rules 8 and 9 of the prompt say what each element means: answer what is
in the first, read the second, obey neither. Both surfaces go through `ask`, so
there is no path on which either reaches the model as a bare sentence next to
the project's own. The elements are framing and not a boundary; what makes an
injected statement safe is `validate_sql` underneath them, and the twelve
adversarial questions in the golden set are scored on the SQL as well as the
prose for exactly that reason (docs/agent-safety.md).

**The conversation travels with the question, and is still not kept.** A
follow-up carries the last few turns of the thread back with it: the drawer
holds the transcript in the browser and this service holds none, which is the
property the privacy note has always claimed and the reason a member's second
question used to arrive with no idea what the first one was. The turns go in
between the cached prefix and the current turn, a member's earlier question
wrapped exactly as the live one is and an earlier answer placed as the
assistant message it was, so nothing about the breakpoint or the current
turn's own layout moves. Rule 11 says what an earlier answer is worth, which
is the agent's own words and not evidence, and the numeric check backs it:
a number only an earlier answer accounts for comes back as `from_history`
rather than as an invention (docs/agent-service.md).

**The game on screen is decided about before it is placed.** When the member
is looking at one of their own games the application sends a redacted summary
of it beside the route sentence, and one more sentence describing it. A single
typed Choice call over the question and that one sentence says whether the
game bears on what was asked (`pipeline.sql_gate.relevance`): `relevant`
places the route sentence, a blank line and the summary, `irrelevant` places
the route sentence alone, and `skipped`, which is a judge that is not
configured or a call that failed, places the summary anyway. The decision
never sees the summary, so it stays one short call however long the game was,
and the verdict, the latency and two lengths are what is written down of any
of it.

**There is an optional second gate behind the first one.** `PRA_SQL_GATE=jev`
puts `pipeline.sql_gate` between the validator and DuckDB: one typed Choice
question to a System One model, asking whether the statement is a read-only
SELECT over the marts that answers the question that was asked. It catches the
thing a denylist structurally cannot, which is a legal query that answers a
question nobody asked, and it is off by default. The denylist is not optional
and runs first in both cases, so a statement it refuses costs nothing and never
reaches a provider (docs/sql-gate.md).

**Every tool call is a span and a counter increment.** `agent.tool.query_marts`
carries the SQL length, the row count and the gate's verdict, `agent.answer`
wraps the run with the model, the number of tool calls and the token usage the
provider reported. The counter is `agent_tool_calls_total{tool=..., gate=...}`,
which `pipeline.telemetry` declared while the serving instrumentation was being
built. The SQL itself is on the span as a length rather than as text: a query
is short and harmless here, but a span attribute is the wrong place to start
putting model output.

**The prompt is a cached prefix, and the counters say whether it really is.**
The system prompt goes to the provider as content blocks with a breakpoint on
the last (`pipeline.prompts.system_blocks`), and marking a prefix is not the
same as caching one: below the model's minimum cacheable length the provider
writes nothing and returns no error. So `token_usage` reports
`cache_read_input_tokens` and `cache_creation_input_tokens` beside the plain
counts, they go on the span as `agent.usage.*` and into
`agent_prompt_tokens_total{kind=...}`, and `docs/agent-service.md` says what
the numbers should look like today. Zero reads is a reading, not a gap.

**A run keeps its evidence, not only its tally.** `tool_calls` says a query ran
and returned four rows, which is enough for a counter and not enough for a
reader deciding whether to believe the answer. So a run also collects
`Evidence`: the full text of every statement, one plain-language line saying
what that statement was for, the first rows it returned as plain JSON values,
the gate's verdict on it, and the cards the card tool matched. That is what a
"what I looked up" panel is built from, and it is the same object on both
surfaces, in `POST /ask` and under `--evidence` on the command line. The
panel renders the line and the rows and leaves the statement alone, which is
why the line is derived from the SQL by `pipeline.describe` rather than asked
of the model: a caption the model wrote would be the one part of the receipt
a reader cannot check. It is bounded on purpose, ten rows and ten cards with every
string cut to 500 characters, because an answer that carries its workings
should still be a response rather than a page.

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
import math
import os
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

import duckdb
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from opentelemetry import trace

from pipeline.config import WAREHOUSE_PATH
from pipeline.describe import LIMIT_CLAUSE, describe_sql, referenced_tables, strip_literals
from pipeline.facts import Fact, FactEvidence, check_numbers, cite_facts, clean_facts
from pipeline.observability import configure_logging, emit_summary
from pipeline.prompts import (
    ALLOWED_TABLES,
    CACHE_CONTROL,
    NO_ARCHETYPES,
    ROLE_ASSISTANT,
    ROLE_USER,
    TABLE_LIST_NOTE,
    Archetypes,
    Turn,
    clean_context,
    clean_history,
    history_chars,
    system_blocks,
    warehouse_tables,
    wrap_question,
    wrap_turn,
)
from pipeline.sql_gate import (
    GATE_OFF,
    NO_GATE,
    ContextRelevance,
    GateDecision,
    NoRelevance,
    OffGate,
    RelevanceDecision,
    SqlGate,
    gate_from_env,
    relevance_from_env,
    schema_summary,
)
from pipeline.storage import AnyLocation, Location, duckdb_connect, location
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
# Why there is no card tool, when nothing asked for one. The other reason is
# built from whatever reading the index raised, and both travel on the built
# agent so that a host can tell a finished build from half of one.
NO_CARD_INDEX: Final = "no card index is configured; this agent has only its SQL half"
ANSWER_SPAN: Final = "agent.answer"
TOOL_SPAN_PREFIX: Final = "agent.tool."

# The two cache counts, by the name this project reports them under and the
# name LangChain files them under inside `usage_metadata["input_token_details"]`.
# The reported names are the provider's, so a log line and an invoice line use
# the same words.
CACHE_TOKEN_FIELDS: Final[dict[str, str]] = {
    "cache_read_input_tokens": "cache_read",
    "cache_creation_input_tokens": "cache_creation",
}

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

# What the evidence on an answer carries, and what it will not. Ten rows is
# what a reader checking an answer scans without scrolling, and the hundredth
# row of a fifty-row scan is not evidence for anything; ten cards is more than
# any question has matched. Five hundred characters is longer than every cell
# the marts hold and short enough that ten rows of them stay a response body.
MAX_EVIDENCE_ROWS: Final = 10
MAX_EVIDENCE_CARDS: Final = 10
MAX_EVIDENCE_CHARS: Final = 500

# The four values `gate_summary` takes. It describes the answer the member
# was given rather than the worst attempt behind it: a run that guessed a
# table name, was refused, and then read the right table has not had its
# answer refused, and a badge that said so was the whole of PLA-198.
SUMMARY_OFF: Final = "off"
SUMMARY_ALLOWED: Final = "allowed"
SUMMARY_ALLOWED_LOW: Final = "allowed_low"
SUMMARY_REFUSED: Final = "refused"

# Why a query produced no rows, as one word beside the sentence that says it
# in full. The sentence is written for the model that has to write a better
# query; the code is written for the application, which has to decide between
# a badge and a line of small print, and for the evaluation, which counts
# them. A closed set, because both readers switch on it.
#
# `table_not_found` and `table_not_allowed` are the pair this ticket added and
# the distinction it turns on. The first is a name no dbt model builds, which
# is the model inventing `mart_leaderboard` and getting caught: nothing was
# protected and nothing was blocked, a guess simply missed. The second is a
# real relation that is off the allowlist, `dim_player` or `fct_game_side`,
# which is the boundary doing its job and is worth a reader's attention.
REFUSED_TABLE_NOT_FOUND: Final = "table_not_found"
REFUSED_TABLE_NOT_ALLOWED: Final = "table_not_allowed"
REFUSED_STATEMENT_NOT_ALLOWED: Final = "statement_not_allowed"
REFUSED_JUDGE_LOW_CONFIDENCE: Final = "judge_low_confidence"
REFUSED_JUDGE_REFUSED: Final = "judge_refused"
REFUSED_ERROR: Final = "error"
REFUSAL_CODES: Final[tuple[str, ...]] = (
    REFUSED_TABLE_NOT_FOUND,
    REFUSED_TABLE_NOT_ALLOWED,
    REFUSED_STATEMENT_NOT_ALLOWED,
    REFUSED_JUDGE_LOW_CONFIDENCE,
    REFUSED_JUDGE_REFUSED,
    REFUSED_ERROR,
)

# What separates the route sentence from the game summary inside the one
# `<context>` element. A blank line, so the two read as two paragraphs of one
# description rather than as one run-on sentence, and no label: a heading
# would be the project's own words inside the element rule 9 tells the model
# is somebody else's.
CONTEXT_JOINER: Final = "\n\n"

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
# Functions that read something other than the seven tables. Three groups, and
# the second and third were added when the agent was opened to members.
#
# The file and network readers come first: DuckDB reaches the filesystem and
# an http(s) URL through these, so a read-only connection without them is not
# read-only in any useful sense.
#
# Then the ones that read the process rather than the data. `getenv` is the
# one that matters: it takes no FROM clause, so a statement built out of it
# names no table at all and would sail past the allowlist check below with a
# provider key in the result set.
#
# Then the catalog. Nothing here leaks a row, but between them they enumerate
# every table, column and setting of the warehouse, which is the table list
# rule 8 says not to hand over and the obvious first step of anything that
# wants a table it was not told about.
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
    "getenv",
    "current_setting",
    "which_secret",
    "duckdb_extensions",
    "duckdb_settings",
    "duckdb_secrets",
    "duckdb_databases",
    "duckdb_tables",
    "duckdb_views",
    "duckdb_columns",
    "duckdb_schemas",
    "pragma_table_info",
    "pragma_database_list",
)

_FUNCTION_CALL: Final = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)
_LEADING_KEYWORD: Final = re.compile(r"^\s*([a-z_]+)", re.IGNORECASE)

# `strip_literals` and `referenced_tables` are imported from
# `pipeline.describe` rather than written here. They were written here, for
# the validator, and they moved when PLA-197 gave them a second caller: the
# receipt's plain-language description reads the same statement for the same
# names, and two copies of the same regular expression are two answers waiting
# to disagree.


# ------------------------------------------------------------ the SQL gate --


@dataclass(frozen=True)
class Refusal:
    """One refusal: the sentence the model reads, and the word the app reads.

    Two fields rather than one because they have two readers with nothing in
    common. `message` is prose aimed at a language model that has to write a
    better query, so it names the rule and the tables; `code` is one of
    `REFUSAL_CODES`, aimed at an application deciding whether to draw an
    error badge and at an evaluation counting how often the model guessed.
    """

    message: str
    code: str


def validate_sql(sql: str, allowed_tables: Sequence[str] = ALLOWED_TABLES) -> str | None:
    """The refusal message this statement earns, or None when it may run.

    `check_sql` with the code dropped. Kept as the name because it is the one
    every other module and every test says, and because a caller that only
    has to decide whether to run a statement should not have to know there is
    a taxonomy of reasons not to.
    """
    refusal = check_sql(sql, allowed_tables)
    return None if refusal is None else refusal.message


def check_sql(sql: str, allowed_tables: Sequence[str] = ALLOWED_TABLES) -> Refusal | None:
    """The refusal this statement earns, or None when it may run.

    A pure function of a string and a list of names: no connection, no model
    and, since PLA-198, no filesystem either, because `warehouse_tables` is a
    committed list rather than a glob. That is what makes the rules testable
    one at a time and what makes them answer the same in every container.
    The message names the rule rather than the symptom, because the reader is
    a language model that has to write a better query next, and "invalid
    query" tells it nothing it can act on.

    The order of the checks is the order of severity, so a statement that both
    drops a table and reads a Parquet file is refused for the drop.
    """
    text = strip_literals(sql).strip().rstrip(";").strip()
    if not text:
        return Refusal("refused: the query is empty.", REFUSED_STATEMENT_NOT_ALLOWED)

    # One statement. A trailing semicolon is fine and has already been taken
    # off; a semicolon anywhere else means a second statement is being smuggled
    # past a check that only ever looks at the first one.
    if ";" in text:
        return Refusal(
            "refused: only one statement per call. Remove the `;` and send a single "
            "SELECT, or make two calls.",
            REFUSED_STATEMENT_NOT_ALLOWED,
        )

    leading = _LEADING_KEYWORD.match(text)
    keyword = leading.group(1).lower() if leading else ""
    if keyword not in {"select", "with"}:
        return Refusal(
            f"refused: this tool runs read-only queries, and the statement starts with "
            f"`{keyword.upper() or text[:12]}`. It has to start with SELECT or WITH.",
            REFUSED_STATEMENT_NOT_ALLOWED,
        )

    lowered = text.lower()
    for forbidden in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{forbidden}\b", lowered):
            return Refusal(
                f"refused: `{forbidden.upper()}` is not allowed here. This tool answers "
                f"questions with a single read-only SELECT and changes nothing.",
                REFUSED_STATEMENT_NOT_ALLOWED,
            )

    for name in {match.group(1).lower() for match in _FUNCTION_CALL.finditer(text)}:
        if name in FORBIDDEN_FUNCTIONS:
            return Refusal(
                f"refused: `{name}(...)` reads outside the warehouse. Query the tables by "
                f"name instead: {', '.join(allowed_tables)}.",
                REFUSED_STATEMENT_NOT_ALLOWED,
            )

    allowed = {name.lower() for name in allowed_tables}
    for table in referenced_tables(sql):
        if table in allowed:
            continue
        if table_exists(table):
            return Refusal(
                f"refused: `{table}` is not a table this tool can read. The tables are: "
                f"{', '.join(allowed_tables)}.",
                REFUSED_TABLE_NOT_ALLOWED,
            )
        return Refusal(
            f"refused: there is no table called `{table}` in this warehouse. Pick one of the "
            f"tables the system prompt lists rather than a name that sounds right: "
            f"{', '.join(allowed_tables)}.",
            REFUSED_TABLE_NOT_FOUND,
        )
    return None


def table_exists(table: str) -> bool:
    """Whether this name is a relation the warehouse really holds.

    The authority is the dbt project, through `pipeline.prompts.warehouse_tables`:
    every model dbt builds is a file in it, and the question being asked is
    "did the model name something real", which the catalog of a warehouse that
    may not even be built cannot answer any better. The DuckDB catalog was the
    other candidate and is the worse one: it needs a connection on a path the
    validator has never been given, it is empty before the gold stage runs, and
    a validator that opened a file would stop being the pure function the
    whole of `docs/agent-safety.md` rests on.

    The list is read from the dbt project when somebody edits it and committed
    as `pipeline.warehouse_tables`, rather than globbed here. This function
    had a naming-convention fallback for the container that ships without the
    dbt project, and that container is the deployed one: the fallback was not
    the rare case, it was the only case, and it called
    `mart_archetype_summary` and `mart_weekly_archetype` real tables being
    blocked. A committed list travels with the package, so there is nothing
    left to fall back to and no second answer to be wrong (docs/sql-gate.md).
    """
    return table in warehouse_tables()


def with_limit(sql: str, *, default: int = DEFAULT_LIMIT, cap: int = MAX_LIMIT) -> str:
    """The statement with a LIMIT on it, added or cut down to the cap.

    Appended rather than wrapped in a subquery, because a wrapper changes what
    the model sees back from what it wrote and makes an ORDER BY inside it
    unreliable. A statement whose only LIMIT is inside a CTE therefore gets a
    second one on the outside, which is correct and occasionally redundant.
    """
    text = sql.strip().rstrip(";").strip()
    found = list(LIMIT_CLAUSE.finditer(text))
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
    """One tool invocation, as the answer reports it back to a caller.

    `gate` is the verdict the SQL gate reached on this call, `off` when no gate
    ran and on every `lookup_cards` call, which the gate has nothing to say
    about. It is carried here rather than summed on the answer because the
    evaluation reports the gate per question and the cost over the run, and
    both are the same list read two ways.
    """

    tool: str
    input_summary: str
    rows: int
    gate: str = GATE_OFF
    gate_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "input_summary": self.input_summary,
            "rows": self.rows,
            "gate": self.gate,
            "gate_cost_usd": self.gate_cost_usd,
        }


# ------------------------------------------------------------- evidence --


def clip(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    """One string cut to a length a response body can afford, ellipsis included."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def json_safe(value: Any, limit: int = MAX_EVIDENCE_CHARS) -> Any:
    """One warehouse value as something JSON carries, strings cut to `limit`.

    DuckDB hands back Python objects, and three of the kinds it hands back are
    not JSON: a `date` or a `timestamp`, a `Decimal`, and a float that is NaN
    or an infinity. The first two have an obvious reading, an ISO string and a
    number, and the third has none: `NaN` is not valid JSON and a parser that
    accepts it disagrees with one that does not, so it becomes null and the
    reader sees a missing number rather than a syntax error.

    Anything that can say its own ISO form does, which covers the date, the
    timestamp and the time without this function naming three types. Anything
    else that is not a container falls back to `str`, cut to the limit, so an
    unexpected column is a short string rather than a serialisation failure in
    the middle of an answer.
    """
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return clip(value, limit)
    if isinstance(value, bytes):
        return clip(value.decode("utf-8", "replace"), limit)
    if isinstance(value, list | tuple | set):
        return [json_safe(item, limit) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item, limit) for key, item in value.items()}
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return clip(str(isoformat()), limit)
    return clip(str(value), limit)


@dataclass(frozen=True)
class QueryEvidence:
    """One statement the run put to the warehouse, and what it got for it.

    The whole statement, not the one-line summary `ToolCall` carries: the
    reader of this is a person deciding whether the answer follows from the
    query, and a query cut off at a hundred characters cannot be read for
    that.

    `refused_reason` is why there are no rows, and it is filled for the three
    ways that happens: the validator refused the statement, the gate refused
    it, or DuckDB itself would not run it. Only the first two are refusals,
    which is what `refused` is for; `gate_summary` counts those and leaves a
    broken query to be read as the empty result it is.

    `refused_code` is the same thing in one word from `REFUSAL_CODES`, and it
    is here because the sentence is written for a model and the application
    has to switch on it. The two that matter most to a reader are
    `table_not_found`, a name no model builds, and `table_not_allowed`, a real
    table off the allowlist: the first is the agent guessing and the second is
    the boundary holding, and until PLA-198 they arrived as the same event.

    `description` is the same statement in a member's words, and it is a
    property rather than a field on purpose: it is derived from `sql` by a
    pure function, so there is no slot for anybody to put a different line in,
    least of all the model that wrote the query (`pipeline.describe`). The
    application draws the receipt from it and keeps the SQL off the screen.
    """

    sql: str
    row_count: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    gate: str = GATE_OFF
    refused_reason: str | None = None
    refused_code: str | None = None
    # Not serialised: the body already carries the reason and the code, and
    # this only decides whether the run's `gate_summary` is `refused`.
    refused: bool = False

    @property
    def description(self) -> str:
        """What this lookup was for, in plain language, derived from the statement."""
        return describe_sql(self.sql)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "description": self.description,
            "row_count": self.row_count,
            "rows": [dict(row) for row in self.rows],
            "gate": self.gate,
            "refused_reason": self.refused_reason,
            "refused_code": self.refused_code,
        }


@dataclass(frozen=True)
class CardEvidence:
    """One printed card the run looked up, as the panel cites it.

    The set and the number are their own fields rather than part of the text,
    because the reader cites a card by them and the application renders them
    as a heading. `text` is therefore the card without that heading.
    """

    name: str
    set_code: str
    number: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "set_code": self.set_code,
            "number": self.number,
            "text": self.text,
        }


def summarize_gate(queries: Sequence[QueryEvidence]) -> str:
    """What happened to the answer this run gave, as one word.

    The rule, since PLA-198: if anything was allowed to run, this describes
    those runs and not the attempts before them. `refused` is reserved for a
    run that got nothing, which is every query refused, or no query at all
    because the one that was tried was refused. The application draws an
    error-tone badge on `refused`, and a correct answer behind a guessed
    table name was earning one: the model wrote `mart_leaderboard`, read the
    refusal, queried `mart_player_summary` and answered, and the member was
    told the data query was refused.

    Among the queries that did run it is still a ladder, worst first, because
    the question that half answers is whether the answer needs a second look.
    A gate that allowed a statement it was not sure about, and a gate that
    errored and let the statement through under `PRA_SQL_GATE_ON_ERROR`, are
    both "allowed, with a caveat". `off` is a run that asked the warehouse
    nothing, and a run whose queries ran with no gate in front of them: in
    neither case did a gate have an opinion to report.

    The refused attempt stays in `evidence.queries` with its reason and its
    code either way. Nothing is hidden; it is read as the detour it was.
    """
    ran = [query for query in queries if not query.refused]
    if not ran:
        return SUMMARY_REFUSED if queries else SUMMARY_OFF
    worst = SUMMARY_OFF
    for query in ran:
        outcome = query.gate.partition(":")[2]
        if outcome in {"allowed_low", "error"}:
            worst = SUMMARY_ALLOWED_LOW
        elif outcome == "allowed" and worst == SUMMARY_OFF:
            worst = SUMMARY_ALLOWED
    return worst


def dedupe_cards(cards: Sequence[CardEvidence]) -> list[CardEvidence]:
    """The cards a run matched, each once, in the order it first saw them.

    Two lookups on one question routinely return the same card, and a panel
    that printed Dragapult ex twice would read as two pieces of evidence for
    something that is one. The key is the printing a reader cites, so the same
    card in two sets stays two rows.
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[CardEvidence] = []
    for card in cards:
        key = (card.name, card.set_code, card.number)
        if key in seen:
            continue
        seen.add(key)
        kept.append(
            CardEvidence(
                name=clip(card.name),
                set_code=clip(card.set_code),
                number=clip(card.number),
                text=clip(card.text),
            )
        )
        if len(kept) == MAX_EVIDENCE_CARDS:
            break
    return kept


@dataclass(frozen=True)
class Evidence:
    """What one run read: its queries, the cards it matched, and the facts it was given.

    The first two are produced by the tools as the run goes. The third is not
    produced at all: the facts arrived with the request, and they are echoed
    here with a `cited` flag so the panel can show which of them the answer
    used (`pipeline.facts`). A fact is evidence in exactly the way a row is,
    which is why it is in this object rather than beside it.
    """

    queries: list[QueryEvidence] = field(default_factory=list)
    cards: list[CardEvidence] = field(default_factory=list)
    facts: list[FactEvidence] = field(default_factory=list)

    @property
    def gate_summary(self) -> str:
        """What happened to the answer, over this run's queries."""
        return summarize_gate(self.queries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "queries": [query.as_dict() for query in self.queries],
            "cards": [card.as_dict() for card in self.cards],
            "facts": [fact.as_dict() for fact in self.facts],
        }


class EvidenceLog:
    """The evidence of the run in progress, collected as the tools produce it.

    Mutable, and one per run, because the tools hand their evidence over one
    call at a time and the answer is assembled from all of it at the end.
    `finish` is where the bounds are applied, so a tool never has to know what
    the response body can afford.
    """

    def __init__(self) -> None:
        self.queries: list[QueryEvidence] = []
        self.cards: list[CardEvidence] = []

    def finish(self, facts: Sequence[FactEvidence] = ()) -> Evidence:
        """The collected evidence, deduplicated and capped, with the facts beside it.

        The facts are a parameter rather than something the log collected,
        because nothing collected them: they came in with the request and
        the only thing the run adds is whether the answer used each one.
        """
        return Evidence(
            queries=list(self.queries), cards=dedupe_cards(self.cards), facts=list(facts)
        )


# The calls made by the run happening on this task, and the question that run
# is answering. Context variables rather than attributes on the agent, because
# one agent object serves every request of a serving process and two concurrent
# `/ask` calls must not collect into the same list. Starlette copies the
# context per request, so each run sees its own.
_calls: ContextVar[list[ToolCall] | None] = ContextVar("pra_agent_calls", default=None)
# What the person asked, so the SQL gate can judge the query against it. The
# tool is handed only the SQL, and "does this answer the question" needs the
# question; threading it through the tool's arguments would put it in the
# model's hands, which is exactly whose judgement the gate is second-guessing.
_question: ContextVar[str] = ContextVar("pra_agent_question", default="")
# What the run in progress has read, for the same reason and with the same
# per-request isolation as `_calls`. Separate from it because `tool_calls` is
# the tally the evaluation reads and this is the evidence a reader reads, and
# the two are allowed to change shape independently.
_evidence: ContextVar[EvidenceLog | None] = ContextVar("pra_agent_evidence", default=None)


def record_call(call: ToolCall) -> None:
    """Add a tool call to the run in progress, if anything is collecting them."""
    collected = _calls.get()
    if collected is not None:
        collected.append(call)


def record_query(query: QueryEvidence) -> None:
    """Add one query's evidence to the run in progress, if there is one."""
    log = _evidence.get()
    if log is not None:
        log.queries.append(query)


def record_cards(cards: Sequence[CardEvidence]) -> None:
    """Add the cards one lookup matched to the run in progress, if there is one."""
    log = _evidence.get()
    if log is not None:
        log.cards.extend(cards)


@contextlib.contextmanager
def collect_evidence() -> Iterator[EvidenceLog]:
    """Collect what the tools read inside this block, and nothing outside it.

    `Agent.ask` wraps a run in it. It is public because the tools are usable
    on their own, by `pipeline.eval` and by anything driving one directly, and
    "what did that read" is the same question there.
    """
    log = EvidenceLog()
    token = _evidence.set(log)
    try:
        yield log
    finally:
        _evidence.reset(token)


def current_question() -> str:
    """The question the run in progress is answering, or empty outside a run."""
    return _question.get()


# ----------------------------------------------------------------- tools --


def open_warehouse(path: AnyLocation) -> duckdb.DuckDBPyConnection:
    """A read-only connection to the warehouse, with a statement timeout on it.

    Read-only is the cheap half of the protection and the allowlist in
    `validate_sql` is the real one, but it costs a keyword and it means a bug in
    the validator cannot write. The timeout is set through a configuration
    parameter and is tolerated if this DuckDB build does not have it: a version
    without the setting should serve queries, not refuse to start.

    A warehouse on S3 is downloaded first and opened with `httpfs` loaded, both
    through `pipeline.storage.duckdb_connect`: DuckDB opens a database file, not
    a stream, and the ops views inside that file read the lake. The download
    happens once per process, which for a service is once per start and not once
    per question.
    """
    connection = duckdb_connect(path)
    with contextlib.suppress(duckdb.Error):
        connection.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT_S}s'")
    return connection


def gate_refusal(decision: GateDecision) -> str:
    """A gate verdict as the sentence the model reads back.

    Shaped exactly like the validator's refusals, and for the same reason: it
    opens with `refused`, it names the rule that stopped it, and it says what a
    better attempt would look like. The confidence is in there because a
    refusal at 0.55 and a refusal at 0.99 mean different things to whoever
    reads the transcript afterwards.
    """
    return (
        f"refused by the {decision.gate} gate at confidence {decision.confidence:.2f}: "
        f"{decision.reason}. Write a plain read-only SELECT over the tables in the system "
        "prompt that answers the question you were asked, or say you cannot answer it."
    )


def gate_refusal_code(decision: GateDecision) -> str:
    """A gate verdict as one of `REFUSAL_CODES`.

    Three outcomes and not one, because they ask different things of whoever
    reads them. `error` is the gate itself being unreachable or unreadable,
    which is an operational problem and not a judgement about the SQL.
    `judge_low_confidence` is the model choosing `allow` under the threshold
    with `PRA_SQL_GATE_LOW_CONFIDENCE=refuse` set, which is a threshold to
    tune. `judge_refused` is the model really saying no, which is the only
    one of the three that is the gate working as advertised.
    """
    if decision.errored:
        return REFUSED_ERROR
    if decision.low_confidence:
        return REFUSED_JUDGE_LOW_CONFIDENCE
    return REFUSED_JUDGE_REFUSED


@dataclass(frozen=True)
class QueryResult:
    """What one statement produced: the model's text, the count, and the rows.

    The text is for the model, which reads a markdown table; `rows` is for the
    reader, which reads the values. They are two renderings of one result and
    are produced together rather than by running the query twice.

    `failure` is the message when nothing ran, and None when the query ran,
    including when it ran and matched nothing.
    """

    text: str
    row_count: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None


def run_marts_query(
    sql: str,
    *,
    warehouse: AnyLocation,
    allowed_tables: Sequence[str] = ALLOWED_TABLES,
) -> tuple[str, int]:
    """Validate, run and render one query. Returns the tool's text and the row count.

    The ungated form, kept because it is the whole of the tool when
    `PRA_SQL_GATE` is unset and because a caller that has no question to judge
    a query against has nothing to give a gate. `guarded_query` is this with a
    gate in the middle.
    """
    answer, rows, _ = guarded_query(sql, warehouse=warehouse, allowed_tables=allowed_tables)
    return answer, rows


def guarded_query(
    sql: str,
    *,
    warehouse: AnyLocation,
    allowed_tables: Sequence[str] = ALLOWED_TABLES,
    gate: SqlGate | None = None,
    question: str = "",
) -> tuple[str, int, GateDecision]:
    """The denylist, then the gate, then DuckDB. Text, row count, and the verdict.

    The order is the point. `validate_sql` is free, deterministic and always
    on, so a statement it refuses is refused before anything is paid for and
    before a provider is told what the agent was asked. The gate only ever sees
    statements that already passed, which is what keeps a run's gate bill
    proportional to the queries that were going to run anyway.

    A refusal from either, and a failure from DuckDB, all come back as text
    with a row count of zero, never as an exception: an exception out of a tool
    ends the agent's turn, and the useful outcome of a bad query is the model
    reading why and writing a better one.
    """
    refusal = check_sql(sql, allowed_tables)
    if refusal is not None:
        logger.info(
            "tool call refused",
            extra={"tool": SQL_TOOL, "reason": refusal.message, "refused_code": refusal.code},
        )
        record_query(
            QueryEvidence(
                sql=sql,
                gate=NO_GATE.label,
                refused_reason=refusal.message,
                refused_code=refusal.code,
                refused=True,
            )
        )
        return refusal.message, 0, NO_GATE
    decision = NO_GATE if gate is None else gate.judge(question, sql, schema_summary())
    if not decision.allowed:
        refused = gate_refusal(decision)
        record_query(
            QueryEvidence(
                sql=sql,
                gate=decision.label,
                refused_reason=refused,
                refused_code=gate_refusal_code(decision),
                refused=True,
            )
        )
        return refused, 0, decision
    result = execute_marts_query(sql, warehouse=warehouse)
    record_query(
        QueryEvidence(
            sql=sql,
            row_count=result.row_count,
            rows=result.rows,
            gate=decision.label,
            refused_reason=result.failure,
            # Not a refusal: the statement was allowed and DuckDB would not
            # run it. It gets the code anyway, so that "there is a reason" and
            # "there is a code for it" are one fact rather than two.
            refused_code=None if result.failure is None else REFUSED_ERROR,
        )
    )
    return result.text, result.row_count, decision


def execute_marts_query(sql: str, *, warehouse: AnyLocation) -> QueryResult:
    """Run one already-checked statement and render what came back.

    Split out of `guarded_query` so that "is this allowed" and "what does it
    return" are two functions rather than two halves of one: nothing here
    checks anything, and nothing above here touches a connection.

    At most `MAX_EVIDENCE_ROWS` rows are kept as values, whatever the count
    says: the count is the answer to "how much did this match" and the rows
    are there to be read, and nobody reads the fiftieth one.
    """
    limited = with_limit(sql)
    target = location(warehouse)
    if not target.is_file():
        missing = (
            f"the warehouse is not built: nothing at {target.name}. "
            "Run `python -m pipeline.gold` first."
        )
        return QueryResult(missing, failure=missing)
    connection = open_warehouse(target)
    try:
        result = connection.sql(limited)
        columns = list(result.columns)
        rows = result.fetchall()
    except duckdb.Error as failure:
        # The class and the message, not a traceback: the model reads this.
        broke = f"the query failed: {type(failure).__name__}: {failure}"
        return QueryResult(broke, failure=broke)
    finally:
        connection.close()
    if not rows:
        return QueryResult("0 rows. The query ran and matched nothing.")
    table = markdown_table(columns, rows)
    return QueryResult(
        f"{table}\n\n{len(rows)} row(s).",
        row_count=len(rows),
        rows=[
            {str(name): json_safe(value) for name, value in zip(columns, row, strict=True)}
            for row in rows[:MAX_EVIDENCE_ROWS]
        ],
    )


def make_query_marts_tool(
    *,
    warehouse: AnyLocation,
    tracer: trace.Tracer,
    metrics: ServiceMetrics,
    allowed_tables: Sequence[str] = ALLOWED_TABLES,
    gate: SqlGate | None = None,
) -> BaseTool:
    """The SQL tool, bound to one warehouse, one set of instruments and one gate."""
    resolved_gate = gate if gate is not None else OffGate()

    def query_marts(sql: str) -> str:
        """Run one read-only SELECT against the gold marts and return the rows."""
        with tracer.start_as_current_span(f"{TOOL_SPAN_PREFIX}{SQL_TOOL}") as span:
            span.set_attribute("agent.tool", SQL_TOOL)
            span.set_attribute("agent.sql.length", len(sql))
            answer, rows, decision = guarded_query(
                sql,
                warehouse=warehouse,
                allowed_tables=allowed_tables,
                gate=resolved_gate,
                question=current_question(),
            )
            span.set_attribute("agent.rows", rows)
            span.set_attribute("gate.name", decision.gate)
            span.set_attribute("gate.allowed", decision.allowed)
            span.set_attribute("gate.confidence", decision.confidence)
            span.set_attribute("gate.cost_usd", decision.cost_usd)
        metrics.count_tool_call(SQL_TOOL, gate=decision.label)
        record_call(
            ToolCall(
                tool=SQL_TOOL,
                input_summary=summarize(sql),
                rows=rows,
                gate=decision.label,
                gate_cost_usd=decision.cost_usd,
            )
        )
        logger.info(
            "tool call",
            extra={
                "tool": SQL_TOOL,
                "rows": rows,
                "sql_length": len(sql),
                "gate": decision.label,
            },
        )
        return answer

    return StructuredTool.from_function(
        func=query_marts,
        name=SQL_TOOL,
        description=(
            "Run one read-only SQL SELECT against the gold marts and get the rows back "
            "as a markdown table. DuckDB dialect. One statement, no semicolons, only "
            f"the tables described in the system prompt. {TABLE_LIST_NOTE} A LIMIT of "
            f"{DEFAULT_LIMIT} is added when you leave one out and any LIMIT above "
            f"{MAX_LIMIT} is reduced."
        ),
    )


def summarize(text: str, width: int = 100) -> str:
    """A tool input as one short line, for the response body and the log."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


@dataclass(frozen=True)
class ToolSet:
    """The agent's tools, the one worth warming, and why the card one is absent.

    `warm` is `None` whenever there is no card tool, which is the case with no
    index built and the case where the index could not be read. When there is
    one it embeds a short fixed string through that tool's own index, so a
    caller can pay for the embedding model before a question does.

    `card_reason` is the same fact in words, and `None` when there is a card
    tool. A skip is logged here, which is enough for a command that ends and
    was not enough for a container that does not: a serving process that
    cached an agent built without the card tool answered "I have no card text"
    for hours after the index it wanted had been rebuilt. The reason travels
    with the agent so that the host holding it can tell a finished build from
    half of one and say so.
    """

    tools: list[BaseTool]
    warm: Callable[[], bool] | None = None
    card_reason: str | None = None


def marts_tools(
    *,
    warehouse: AnyLocation = WAREHOUSE_PATH,
    tracer: trace.Tracer,
    metrics: ServiceMetrics,
    card_index: AnyLocation | None = None,
    gate: SqlGate | None = None,
) -> ToolSet:
    """Every tool the agent gets: the SQL one always, the card one when there is an index.

    `pipeline.card_index` is imported here rather than at module scope because
    it pulls in pyarrow and the embedder, and an agent asked only for matchup
    numbers should not pay that import for a tool it will not call. A missing
    or unreadable index is a logged skip, not a failure: the SQL half of the
    agent works perfectly well without the card text, and that now covers a
    serving image whose query embedder was never baked in, which raises
    `QueryEmbedderError` out of the load and lands here as a `RuntimeError`.
    The skip comes back as `ToolSet.card_reason` as well as a log line, so a
    long-lived host can report it and build again rather than keep half an
    agent for the rest of its life.

    The index is loaded here and handed to the tool rather than loaded inside
    it, so that this function can keep it and hand back a warmer over the same
    object. Two copies would be two resident embedding models, which on a
    function sized for one is the opposite of the point.
    """
    tools = [
        make_query_marts_tool(warehouse=warehouse, tracer=tracer, metrics=metrics, gate=gate),
    ]
    if card_index is None:
        return ToolSet(tools, card_reason=NO_CARD_INDEX)
    from pipeline import card_index as index_module
    from pipeline.query_embedder import QueryEmbedderError

    try:
        index = index_module.CardIndex.load(card_index)
        tools.append(
            index_module.make_lookup_cards_tool(
                card_index, tracer=tracer, metrics=metrics, index=index
            )
        )
    except (OSError, ValueError, QueryEmbedderError) as failure:
        reason = f"the card index could not be read: {type(failure).__name__}: {failure}"
        logger.warning(
            "the card lookup tool is off",
            extra={"index": str(card_index), "error": reason},
        )
        return ToolSet(tools, card_reason=reason)
    return ToolSet(tools, warm=lambda: index_module.warm_index(index))


# ----------------------------------------------------------------- agent --


def prior_messages(turns: Sequence[Turn]) -> list[BaseMessage]:
    """The conversation so far, as the alternating messages it is sent back as.

    After the cached system prefix and before the current turn, which is the
    only place they can go without moving anything: the breakpoint is on the
    last system block and the live turn keeps the `<context>` and `<question>`
    layout it has always had, byte for byte, whether or not a conversation
    came with it.

    A member's earlier question is wrapped exactly as the live one is, by
    `wrap_question`, so the same rule 8 covers it and a member who typed
    `</question>` two turns ago cannot reach out of the element they typed it
    into. An earlier answer is placed as an `AIMessage` and as it stands, with
    our own delimiters taken out of it and nothing added: it is a turn of the
    conversation in the position the provider has for one, so there is no
    element to put it in and no label of ours to put beside it. What tells
    the model what such a turn is worth is rule 11, not a wrapper.

    The last of them carries the second cache breakpoint, for the reason the
    last system block carries the first: everything above it is identical
    from one call to the next within a question, and within a thread the
    whole of it repeats on the next question too. `clean_history` guarantees
    the last turn is an assistant one and never empty, so the mark always has
    a text block of its own to sit on, and it is the only one marked: a
    breakpoint per turn would be four entries written to serve one read. With
    no history there is no message here and nothing is marked, which is the
    arrangement every request had before this existed
    (docs/agent-service.md).
    """
    messages: list[BaseMessage] = []
    for turn in turns:
        if turn.role == ROLE_USER:
            messages.append(HumanMessage(content=wrap_question(turn.text)))
        else:
            messages.append(AIMessage(content=turn.text))
    if messages:
        # A list of content blocks rather than a string, because that is the
        # only shape `cache_control` has anywhere to go; langchain-anthropic
        # forwards the key on a text block to the provider untouched.
        last = messages[-1]
        messages[-1] = AIMessage(
            content=[
                {"type": "text", "text": str(last.content), "cache_control": dict(CACHE_CONTROL)}
            ]
        )
    return messages


@dataclass(frozen=True)
class Answer:
    """What one run produced: the text, what it called, what it read, and the cost.

    `tool_calls` is the tally and `evidence` is the workings. Both are here
    because they have two readers: `pipeline.eval` scores a run on which tools
    it called, and a person reading the answer wants the statement and the
    rows. Neither is derived from the other, so neither is dropped.
    """

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)
    # Whether a `<context>` element was really put in front of the question.
    # A flag and never the text: the application shows a member an "about this
    # page" chip off this, and a chip that says yes when the context was
    # dropped for being empty or for being nothing but delimiters would be the
    # interface telling a small lie about what the answer was built from.
    context_used: bool = False
    # Whether the game summary in particular was placed, which `context_used`
    # cannot say on its own: a request that sent a route sentence and a game
    # the judge dropped placed a context and did not place the game, and the
    # chip the application draws is about the game.
    context_game_used: bool = False
    # What the relevance judge said about the game on screen, or None when no
    # game was sent and there was nothing to decide. One of `relevant`,
    # `irrelevant`, `skipped` (`pipeline.sql_gate.RELEVANCE_VERDICTS`).
    context_relevance: str | None = None
    # Every number in the prose that nothing this run read can account for,
    # as it was written. A report and never a refusal: the answer is here
    # whatever is in this list, and an empty list is the ordinary case
    # (`pipeline.facts.check_numbers`, docs/agent-safety.md).
    unverified_numbers: list[str] = field(default_factory=list)
    # Numbers in the prose that only an earlier answer of this conversation
    # accounts for. Apart from `unverified_numbers` and never inside it: a
    # number this agent wrote two turns ago out of rows nobody has read
    # again is not an invention and is not a finding either, and rule 11 of
    # the prompt is what the two lists are reporting on.
    from_history: list[str] = field(default_factory=list)

    @property
    def gate_summary(self) -> str:
        """What happened to the answer, over this run's queries, as one word."""
        return self.evidence.gate_summary

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "model": self.model,
            "usage": dict(self.usage),
            "evidence": self.evidence.as_dict(),
            "gate_summary": self.gate_summary,
            "context_used": self.context_used,
            "context_game_used": self.context_game_used,
            "context_relevance": self.context_relevance,
            "unverified_numbers": list(self.unverified_numbers),
            "from_history": list(self.from_history),
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
        metrics: ServiceMetrics | None = None,
        warmer: Callable[[], bool] | None = None,
        card_tool_reason: str | None = None,
        relevance: ContextRelevance | None = None,
    ) -> None:
        self.graph = graph
        self.model_name = model_name
        self.tool_names = list(tool_names)
        # The same instruments the tools were built with, so the prompt-token
        # counter and the tool-call counter are in one registry and a scrape
        # reads both. Optional because a test builds an `Agent` around a
        # scripted graph and has nothing to scrape.
        self.metrics = metrics
        # The same tracer the tools were built with, rather than the global
        # provider: `agent.answer` has to be the parent of the tool spans, and
        # a second provider would put them in two unrelated traces.
        self.tracer = tracer
        self.warmer = warmer
        # None when this agent has the card tool, and otherwise the sentence
        # saying what stopped it. `pipeline.serve` reads it off the built agent
        # to decide whether the build is finished, so an attribute rather than
        # only a log line: a host cannot act on something it has to grep for.
        self.card_tool_reason = card_tool_reason
        # Who decides whether the game on a member's screen belongs in front
        # of their question. `NoRelevance` by default, which skips every
        # decision and attaches every game, because that is what an
        # environment with no judge configured should do.
        self.relevance = relevance if relevance is not None else NoRelevance()

    def warm(self) -> bool:
        """Make the card tool's embedding model resident, asking the provider nothing.

        `GET /warm` is the only caller. The point is the one thing a first
        question cannot avoid paying for and a keepalive ping can: the model
        behind `lookup_cards`. Nothing about the language model is touched,
        because that one is a network call per question and warming it would
        be spending money on a ping.

        False when there is no card tool, which is an agent that has only the
        SQL half and nothing to warm.
        """
        if self.warmer is None:
            return False
        return self.warmer()

    def decide_relevance(
        self, question: str, game: str, first_line: str | None
    ) -> RelevanceDecision:
        """What the judge says about the game on screen, for this question.

        The judge is handed the question and the first line, and the game
        text is handed to nothing: it is here only so that a caller cannot
        reach this with no game and get a verdict about nothing. One short
        call, whatever the summary's length, which is the whole reason the
        application sends a first line at all.
        """
        if not game:
            return RelevanceDecision(reason="no game was sent")
        return self.relevance.relevance(question, clean_context(first_line))

    def ask(
        self,
        question: str,
        context: str | None = None,
        job: str | None = None,
        context_game: str | None = None,
        context_first_line: str | None = None,
        context_facts: Sequence[Fact] | None = None,
        context_archetypes: Archetypes | None = None,
        history: Sequence[Turn] | None = None,
    ) -> Answer:
        """Run the loop on one question and collect what it did.

        The question goes to the model inside the `<question>` element rule 8
        of the prompt describes, and goes to the SQL gate as it was typed. Two
        readings on purpose: the model is being told where the member's words
        start and stop, and the gate is being asked whether a statement
        answers what was actually asked, which is a judgement about the plain
        text and not about the framing around it.

        `context` is the application's sentence or two about where the member
        is standing, and it goes in a `<context>` element in front of the
        question rather than into the question or into the prompt: rule 9 and
        `pipeline.prompts.wrap_turn` say why. It reaches the gate not at all.
        The gate's question is whether a statement answers what was asked, and
        the context is not what was asked.

        `context_game` is the other half of that element and the one this
        method decides about. It is a redacted plain-text summary of the game
        the member is looking at, built by the application from their own log;
        this service never fetches a game. `context_first_line` is one
        sentence describing the same game, and it is the only part of it the
        relevance judge is shown. When the judge says `irrelevant` the summary
        is dropped and the route sentence goes on its own; on `relevant` and
        on `skipped` the placed context is the route sentence, a blank line,
        then the summary. Failing towards attaching is deliberate: an
        irrelevant game in the context is a few hundred characters the model
        ignores, and a missing game on a question about that game is a worse
        answer.

        `context_facts` is the numbered list of analysis facts the
        application computed from the same log, and it rides with the game:
        the facts are placed when the summary is placed and dropped when the
        judge drops it, because a fact about a game that is not in front of
        the model is a sentence with nothing to attach to. They go inside the
        same `<context>` element, after the game text, as the `<facts>` list
        rule 10 describes.

        `context_archetypes` is the two decks of that same game, the
        member's own and their opponent's, either of which may be missing on
        a game whose deck was never identified. It rides with the game for
        the reason the facts do, and it is placed as one sentence between the
        summary and the facts (`pipeline.prompts.render_decks`). The summary
        the application writes has always named the opponent's deck and
        called the member's own "your deck", so without this field a review
        could read one row of `mart_archetype_pace` and not the other.

        `history` is the last few turns of the conversation, which the
        application kept in the browser and sends back with a follow-up
        because this service keeps none. They are placed as alternating
        messages between the cached system prefix and the current turn, so
        nothing about the breakpoint or about the current turn's own layout
        moves: with no history the messages are the one message they have
        always been (`prior_messages`, docs/agent-service.md). A history this
        cannot place, which is one that does not alternate or holds an empty
        turn, is placed as nothing; the service refuses such a request with a
        422 before it reaches here.

        After the answer comes back, every number in its prose is looked up
        in the rows, the cards and those fact values, and whatever is found
        nowhere is reported on the answer as `unverified_numbers`. A number
        that only an earlier answer of the conversation accounts for is
        reported apart, as `from_history`, because it is this agent quoting
        itself rather than inventing or reading (rule 11). Both are a report
        and not a refusal: the answer returns either way, and what is written
        down is the two counts (`pipeline.facts`).

        `job` is the application's own router label for the question, one of
        `pipeline.prompts.JOBS`. Since PLA-205 it does something: it is
        written into the human turn as one line, `Routed as: my_mistake`,
        above the `<context>` element when there is one and above the
        `<question>` element when there is not, and it names which of the
        playbooks in the second system block the model works from. A label
        that is not one of the six is no line at all, and the turn is then
        the bytes it has always been (`pipeline.prompts.route_line`). It is
        still on the log line and on the span, which is what it was carried
        for before it changed anything.

        Neither the context, the game summary, the first line, the question
        nor a word of the conversation is logged or put on a span, here or
        anywhere below. What is recorded of them is four lengths, a turn
        count, a label, a verdict and a duration (docs/agent-safety.md).

        This is the one place either surface wraps anything. `POST /ask` and
        the command line both arrive here with bare strings, so there is no
        second path on which a question could reach the model unwrapped.
        """
        collected: list[ToolCall] = []
        token = _calls.set(collected)
        asked = _question.set(question)
        route = clean_context(context)
        game = clean_context(context_game)
        decision = self.decide_relevance(question, game, context_first_line) if game else None
        if decision is not None and self.metrics is not None:
            self.metrics.observe_context_relevance(decision.verdict, decision.latency_ms / 1000)
        attached = game if decision is not None and decision.attach else ""
        placed = CONTEXT_JOINER.join(part for part in (route, attached) if part)
        # With the game and never without it, which is one decision and not
        # two: the judge already said whether what the member is looking at
        # belongs in front of the question.
        facts = clean_facts(context_facts) if attached else ()
        # With the game, for the same one decision: two deck names in front
        # of a model that cannot see the game they belong to is a pair of
        # labels with nothing under them.
        decks = (context_archetypes or NO_ARCHETYPES) if attached else NO_ARCHETYPES
        # Cleaned here and never rejected here: the service has already
        # refused a malformed conversation with a 422, and the floor under
        # the other callers is a history that is placed as nothing rather
        # than placed in pieces.
        prior = clean_history(history)
        try:
            with collect_evidence() as log, self.tracer.start_as_current_span(ANSWER_SPAN) as span:
                span.set_attribute("agent.model", self.model_name)
                span.set_attribute("agent.question.length", len(question))
                # The context as it was placed, so a context that was nothing
                # but delimiters reads as the nothing it became, and a game
                # the judge dropped is not counted in the length.
                span.set_attribute("agent.context_chars", len(placed))
                span.set_attribute("agent.context_game_chars", len(attached))
                span.set_attribute("agent.context_relevance", decision.verdict if decision else "")
                span.set_attribute("agent.relevance_ms", decision.latency_ms if decision else 0)
                span.set_attribute("agent.job", job or "")
                span.set_attribute("agent.facts", len(facts))
                # Whether each side's deck was named, and never which deck it
                # was: an archetype is not a member, but it is still a thing
                # about one game that nothing needs in a log to be read.
                span.set_attribute("agent.deck_mine_named", bool(decks.mine))
                span.set_attribute("agent.deck_theirs_named", bool(decks.theirs))
                # The conversation as two numbers and never as text: how many
                # turns came back with the question and how long they were.
                span.set_attribute("agent.history_turns", len(prior))
                span.set_attribute("agent.history_chars", history_chars(prior))
                if self.metrics is not None:
                    self.metrics.count_history_turns(len(prior))
                turn = HumanMessage(
                    content=wrap_turn(question, placed, [fact.text for fact in facts], job, decks)
                )
                state = self.graph.invoke({"messages": [*prior_messages(prior), turn]})
                messages: list[BaseMessage] = list(state["messages"])
                usage = token_usage(messages)
                span.set_attribute("agent.tool_calls", len(collected))
                # Every reported count, which since the cache breakpoint went
                # in includes `agent.usage.cache_read_input_tokens` and
                # `agent.usage.cache_creation_input_tokens`.
                for name, value in usage.items():
                    span.set_attribute(f"agent.usage.{name}", value)
                if self.metrics is not None:
                    self.metrics.count_prompt_tokens(usage)
                # Inside the span, because the count is one of its
                # attributes and the check is a scan of a string that has
                # already been produced: nothing here asks the provider
                # anything or adds a call to a member's wait.
                text = final_text(messages)
                evidence = log.finish(cite_facts(text, facts))
                check = check_numbers(
                    text,
                    rows=[row for query in evidence.queries for row in query.rows],
                    cards=[f"{card.number} {card.text}" for card in evidence.cards],
                    facts=facts,
                    history=[
                        prior_turn.text for prior_turn in prior if prior_turn.role == ROLE_ASSISTANT
                    ],
                )
                span.set_attribute("agent.unverified_numbers", len(check.unverified))
                span.set_attribute("agent.from_history", len(check.from_history))
                if self.metrics is not None:
                    self.metrics.count_unverified_numbers(len(check.unverified))
        finally:
            _question.reset(asked)
            _calls.reset(token)
        answer = Answer(
            answer=text,
            tool_calls=collected,
            model=self.model_name,
            usage=usage,
            evidence=evidence,
            context_used=bool(placed),
            context_game_used=bool(attached),
            context_relevance=decision.verdict if decision else None,
            unverified_numbers=list(check.unverified),
            from_history=list(check.from_history),
        )
        logger.info(
            "agent answered",
            extra={
                "model": self.model_name,
                "tool_calls": len(collected),
                "usage": usage,
                "answer_length": len(answer.answer),
                "gate_summary": answer.gate_summary,
                # Lengths, a label, a verdict and a duration. Neither the
                # context, the game summary nor the sentence the verdict was
                # reached on is written down, at this level or any other.
                "context_chars": len(placed),
                "context_game_chars": len(attached),
                "context_relevance": decision.verdict if decision else "",
                "relevance_ms": decision.latency_ms if decision else 0,
                "job": job or "",
                # The count and never the numbers: a number an answer wrote
                # is the answer, and this line has never carried any of it.
                "facts": len(facts),
                "unverified_numbers": len(check.unverified),
                # Two counts and a length, and not a word of what was said
                # in any of those turns.
                "history_turns": len(prior),
                "history_chars": history_chars(prior),
                "from_history": len(check.from_history),
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

    The two cache counts come from a nested place. LangChain keeps the plain
    counts at the top of `usage_metadata` and puts the cache ones under
    `input_token_details` as `cache_read` and `cache_creation`; the names
    reported here are the provider's own, `cache_read_input_tokens` and
    `cache_creation_input_tokens`, because those are what the platform's
    documentation and its pricing table call them and a reader comparing a log
    line with a bill should not have to translate.

    They are reported as zero whenever anything was reported at all, which is
    the one place this function does fill a count in. A missing cache field on
    a provider that answered is a real zero, not an absence: it means the
    prefix was not cached, which is exactly the thing worth seeing. A run that
    reported no usage at all still gets an empty dictionary.
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
        details = metadata.get("input_token_details")
        details = details if isinstance(details, dict) else {}
        for name, reported in CACHE_TOKEN_FIELDS.items():
            value = details.get(reported)
            totals[name] = totals.get(name, 0) + (value if isinstance(value, int) else 0)
    return totals


def build_agent(
    *,
    model: BaseChatModel | None = None,
    warehouse: AnyLocation = WAREHOUSE_PATH,
    card_index: AnyLocation | None = None,
    tracer: trace.Tracer | None = None,
    metrics: ServiceMetrics | None = None,
    gate: SqlGate | None = None,
    relevance: ContextRelevance | None = None,
) -> Agent:
    """The agent, with its model, its warehouse, its instruments and its gate injected.

    Every argument has a default that is the real thing, and every one of them
    is replaceable, which is the same shape `pipeline.serve` gives its model
    loader. The tests pass a scripted chat model and a fixture warehouse and
    exercise everything else for real.

    The gate defaults to whatever `PRA_SQL_GATE` asks for, which is `OffGate`
    unless something has turned it on, and is read here rather than inside the
    tool so that a misconfigured gate fails while the agent is being built
    instead of in the middle of a question.

    The relevance judge is resolved from the gate first and from the
    environment second, so a deployment with the gate on holds one Jev client
    and asks it two kinds of question rather than holding two
    (`pipeline.sql_gate.relevance_from_env`).
    """
    resolved_metrics = metrics if metrics is not None else build_metrics()
    resolved_tracer = tracer or build_tracer_provider(SERVICE_NAME).get_tracer(__name__)
    resolved_gate = gate if gate is not None else gate_from_env()
    resolved_relevance = relevance if relevance is not None else relevance_from_env(resolved_gate)
    chat = model if model is not None else chat_model()
    toolset = marts_tools(
        warehouse=warehouse,
        tracer=resolved_tracer,
        metrics=resolved_metrics,
        card_index=card_index,
        gate=resolved_gate,
    )
    tools = toolset.tools
    # A `SystemMessage` of content blocks rather than a plain string, so the
    # last block can carry the cache breakpoint; `create_agent` accepts either
    # and langchain-anthropic forwards `cache_control` on a text block as is.
    graph = create_agent(
        model=chat,
        tools=tools,
        system_prompt=SystemMessage(
            content=system_blocks(with_card_tool=any(t.name == CARD_TOOL for t in tools))
        ),
    )
    return Agent(
        graph.with_config({"recursion_limit": MAX_ITERATIONS * 2}),
        tracer=resolved_tracer,
        metrics=resolved_metrics,
        model_name=getattr(chat, "model_name", None) or getattr(chat, "model", "") or "fake",
        tool_names=[tool.name for tool in tools],
        warmer=toolset.warm,
        card_tool_reason=toolset.card_reason,
        relevance=resolved_relevance,
    )


def default_card_index(path: AnyLocation | None) -> Location | None:
    """The card index to use, or None when there is no built index to load."""
    if path is None:
        return None
    resolved = location(path)
    return resolved if resolved.is_dir() else None


# ------------------------------------------------------------ entry point --


def render(answer: Answer) -> str:
    """One answer as the block the command line prints."""
    lines = [answer.answer, ""]
    for call in answer.tool_calls:
        gate = "" if call.gate == GATE_OFF else f" [gate: {call.gate}]"
        lines.append(f"  [{call.tool}] {call.rows} row(s):{gate} {call.input_summary}")
    judged = [call for call in answer.tool_calls if call.gate != GATE_OFF]
    if judged:
        spent = sum(call.gate_cost_usd for call in judged)
        lines.append(f"  gate cost: ${spent:.6f} ({len(judged)} call(s))")
    if answer.usage:
        counts = ", ".join(f"{name}={value}" for name, value in sorted(answer.usage.items()))
        lines.append(f"  tokens: {counts}")
    return "\n".join(lines).rstrip()


def render_evidence(answer: Answer) -> str:
    """The evidence as JSON, in the shape `POST /ask` puts it in its body.

    The same object rather than a prettier one: the point of printing it here
    is that a question answered on the command line and the same question
    answered over HTTP can be compared without allowing for two renderings.
    """
    return json.dumps(
        {"evidence": answer.evidence.as_dict(), "gate_summary": answer.gate_summary},
        indent=2,
    )


def rendered(answer: Answer, *, evidence: bool) -> str:
    """One answer as the block the command line prints, with or without the workings."""
    text = render(answer)
    return f"{text}\n\n{render_evidence(answer)}" if evidence else text


def repl(agent: Agent, *, evidence: bool = False) -> int:
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
        sys.stdout.write(rendered(agent.ask(question), evidence=evidence) + "\n")


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
        type=location,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help=(
            "the DuckDB warehouse to read, a file or an s3:// object that is "
            f"downloaded to read (default: {WAREHOUSE_PATH})"
        ),
    )
    parser.add_argument(
        "--card-index",
        type=location,
        default=None,
        metavar="PATH",
        help="a built card index directory or s3:// prefix; without one the agent has "
        "only the SQL tool",
    )
    parser.add_argument(
        "--model", default=None, metavar="NAME", help=f"provider model (default: ${MODEL_VAR})"
    )
    parser.add_argument("--json", action="store_true", help="print the answer as one JSON object")
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="print the queries, their rows and the cards the answer rests on; "
        "--json always carries them",
    )
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
            return repl(agent, evidence=args.evidence)
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
    # The evidence goes to stdout and not into the log record. It is rows of
    # mart data rather than a measurement, a log line is the wrong place to
    # start putting query results, and `gate_summary` is the part of it a
    # collector can chart.
    fields = {name: value for name, value in answer.as_dict().items() if name != "evidence"}
    emit_summary(logger, "agent answer", fields, text=rendered(answer, evidence=args.evidence))
    return 0


if __name__ == "__main__":
    # `python -m pipeline.agent` loads this file as `__main__`, and the card tool
    # then imports `pipeline.agent` a second time as itself: two copies of the
    # module, two `_calls` context variables, and a `lookup_cards` call that is
    # recorded into a list nobody reads. Running the real module's `main` keeps
    # one copy, which is what every other entry point sees when it imports us.
    from pipeline.agent import main as agent_main

    raise SystemExit(agent_main())
