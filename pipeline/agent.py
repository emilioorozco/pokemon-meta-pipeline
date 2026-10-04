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
`Evidence`: the full text of every statement, the first rows it returned as
plain JSON values, the gate's verdict on it, and the cards the card tool
matched. That is what a "what I looked up" panel is built from, and it is the
same object on both surfaces, in `POST /ask` and under `--evidence` on the
command line. It is bounded on purpose, ten rows and ten cards with every
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
from pipeline.observability import configure_logging, emit_summary
from pipeline.prompts import ALLOWED_TABLES, clean_context, system_blocks, wrap_turn
from pipeline.sql_gate import (
    GATE_OFF,
    NO_GATE,
    GateDecision,
    OffGate,
    SqlGate,
    gate_from_env,
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

# The four values `gate_summary` takes: the worst thing that happened to a
# query in this run, which is what a banner over the panel is drawn from.
SUMMARY_OFF: Final = "off"
SUMMARY_ALLOWED: Final = "allowed"
SUMMARY_ALLOWED_LOW: Final = "allowed_low"
SUMMARY_REFUSED: Final = "refused"

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
    """

    sql: str
    row_count: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    gate: str = GATE_OFF
    refused_reason: str | None = None
    # Not serialised: the body already carries the reason, and this only
    # decides whether the run's `gate_summary` is `refused`.
    refused: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "row_count": self.row_count,
            "rows": [dict(row) for row in self.rows],
            "gate": self.gate,
            "refused_reason": self.refused_reason,
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
    """The worst thing that happened to a query in this run, as one word.

    A ladder rather than a count, because the question it answers is whether
    anything in the answer needs a second look. A refusal, from the validator
    or from the gate, is the worst and ends the walk. A gate that allowed a
    statement it was not sure about, and a gate that errored and let the
    statement through under `PRA_SQL_GATE_ON_ERROR`, are both "allowed, with
    a caveat". `off` is a run that asked the warehouse nothing, and a run
    whose queries ran with no gate in front of them: in neither case did a
    gate have an opinion to report.
    """
    worst = SUMMARY_OFF
    for query in queries:
        if query.refused:
            return SUMMARY_REFUSED
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
    """What one run read: its queries in call order, and the cards it matched."""

    queries: list[QueryEvidence] = field(default_factory=list)
    cards: list[CardEvidence] = field(default_factory=list)

    @property
    def gate_summary(self) -> str:
        """The worst gate outcome over this run's queries."""
        return summarize_gate(self.queries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "queries": [query.as_dict() for query in self.queries],
            "cards": [card.as_dict() for card in self.cards],
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

    def finish(self) -> Evidence:
        """The collected evidence, deduplicated and capped."""
        return Evidence(queries=list(self.queries), cards=dedupe_cards(self.cards))


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
    refusal = validate_sql(sql, allowed_tables)
    if refusal is not None:
        logger.info("tool call refused", extra={"tool": SQL_TOOL, "reason": refusal})
        record_query(
            QueryEvidence(sql=sql, gate=NO_GATE.label, refused_reason=refusal, refused=True)
        )
        return refusal, 0, NO_GATE
    decision = NO_GATE if gate is None else gate.judge(question, sql, schema_summary())
    if not decision.allowed:
        refused = gate_refusal(decision)
        record_query(
            QueryEvidence(sql=sql, gate=decision.label, refused_reason=refused, refused=True)
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
            f"the tables described in the system prompt. A LIMIT of {DEFAULT_LIMIT} is "
            f"added when you leave one out and any LIMIT above {MAX_LIMIT} is reduced."
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

    @property
    def gate_summary(self) -> str:
        """The worst gate outcome over this run's queries, as one word."""
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

    def ask(self, question: str, context: str | None = None, job: str | None = None) -> Answer:
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

        `job` is the application's own router label for the question, carried
        so that the log line and the span can be read by job. It changes
        nothing about the answer today; the playbooks are a later ticket.

        Neither the context nor the question is logged or put on a span, here
        or anywhere below. What is recorded of them is two lengths and a label
        (docs/agent-safety.md).

        This is the one place either surface wraps anything. `POST /ask` and
        the command line both arrive here with bare strings, so there is no
        second path on which a question could reach the model unwrapped.
        """
        collected: list[ToolCall] = []
        token = _calls.set(collected)
        asked = _question.set(question)
        placed = clean_context(context)
        try:
            with collect_evidence() as log, self.tracer.start_as_current_span(ANSWER_SPAN) as span:
                span.set_attribute("agent.model", self.model_name)
                span.set_attribute("agent.question.length", len(question))
                # The context as it was placed, so a context that was nothing
                # but delimiters reads as the nothing it became.
                span.set_attribute("agent.context_chars", len(placed))
                span.set_attribute("agent.job", job or "")
                turn = HumanMessage(content=wrap_turn(question, context))
                state = self.graph.invoke({"messages": [turn]})
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
        finally:
            _question.reset(asked)
            _calls.reset(token)
        answer = Answer(
            answer=final_text(messages),
            tool_calls=collected,
            model=self.model_name,
            usage=usage,
            evidence=log.finish(),
            context_used=bool(placed),
        )
        logger.info(
            "agent answered",
            extra={
                "model": self.model_name,
                "tool_calls": len(collected),
                "usage": usage,
                "answer_length": len(answer.answer),
                "gate_summary": answer.gate_summary,
                # A length and a label. The context itself is never written
                # down, at this level or any other.
                "context_chars": len(placed),
                "job": job or "",
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
    """
    resolved_metrics = metrics if metrics is not None else build_metrics()
    resolved_tracer = tracer or build_tracer_provider(SERVICE_NAME).get_tracer(__name__)
    resolved_gate = gate if gate is not None else gate_from_env()
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
