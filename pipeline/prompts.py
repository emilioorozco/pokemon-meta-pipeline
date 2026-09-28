"""The agent's system prompt, with the schema half generated from the dbt models.

A prompt that lists table and column names is documentation, and documentation
that is typed twice goes stale on the first rename. So the schema section is
rendered at import time from `dbt/models/marts/schema.yml`, the same file dbt
builds and tests the models from: a column renamed in the model is renamed in
the prompt on the next import, and a column that never existed cannot be
described here at all.

Only the allowed tables are rendered. The prompt and the SQL tool's allowlist
read the same constant, so a table the tool would refuse is never advertised,
which is what keeps a refusal rare rather than routine.

Descriptions are cut to their first sentence, and to a character budget inside
that. The full text in `schema.yml` is written for a person reading the model
and runs to several thousand tokens across the seven tables, which would be
most of a small model's context spent on prose it re-reads on every turn. The
first sentence is the definition; the rest is the reasoning, which belongs in
the file and not in a context window.

The rules section is hand written, and it is the part that matters. Three of
the seven rules exist because the corpus is small and honest reporting about a
small corpus is the whole point of the project: cite the sample size, say when
the mart itself flags the cell as thin, and never turn an observation rate into
an inclusion rate. A fourth forbids inventing a number when a query comes back
empty. The fifth is a boundary rather than a style note: nothing the agent can
reach carries a name or a handle, so it cannot answer a question about a
person even if it is asked nicely.

The last two are about the SQL rather than the sentence, and both were written
against a failure the golden evaluation caught on its first run with a real
model (docs/evals.md). A question with "the most" in it invites `LIMIT 1`, and
`LIMIT 1` over a tie reports one of two right answers as the answer; and a name
the questioner capitalised their own way, matched with `=`, comes back empty
and reads exactly like an archetype with no games. Both are general: a user
asking either question deserves the tie and the row, not a tidy wrong answer.

The whole prompt can be replaced from outside, by pointing
`PRA_AGENT_SYSTEM_PROMPT_FILE` at a file. That hook exists for one purpose: the
golden evaluation in `pipeline.eval` claims the rules above are load bearing,
and the only way to show it is to run the same questions against a prompt with
the rules taken out and watch the score fall (docs/evals.md). A missing file
raises rather than falling back, because an experiment that quietly ran the
good prompt would report the wrong conclusion, and a service started with the
variable set by accident should fail loudly on the first agent it builds.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

from pipeline.config import REPO_ROOT

MARTS_SCHEMA: Final = REPO_ROOT / "dbt" / "models" / "marts" / "schema.yml"
SCHEMA_FILES: Final[tuple[Path, ...]] = (MARTS_SCHEMA,)

# A file whose contents replace the whole prompt, schema and rules included.
# Read on every call rather than at import, because the evaluation sets it and
# then builds an agent inside the same process.
PROMPT_FILE_VAR: Final = "PRA_AGENT_SYSTEM_PROMPT_FILE"

# The tables the SQL tool will run against, and therefore the only ones the
# prompt describes: the four marts of the gold layer, and the three dimensions
# a mart's keys join to.
#
# `fct_game_side` is absent because the marts already aggregate it and a fact
# at the (game, seat) grain is where a careless query starts double counting.
# `dim_player` is absent because it is the roster: one row per member, and the
# question it answers is "who is here", which the agent has no business asking.
# `mart_player_summary` is present, and is the one table the agent can read that
# is keyed by a person. It is an aggregate of members only, keyed by the same
# irreversible token, and it holds no handle; rule 5 of the prompt is what keeps
# the agent from presenting that token as a person (docs/data-handling.md).
# `dim_season` and `dim_format` are absent because both are placeholders today
# and neither is joined by any mart. The ops models (`run_metrics`,
# `mart_pipeline_health`) are absent too: they are the pipeline's own telemetry,
# they answer no question about the metagame, and leaving them out keeps the
# prompt to one schema file and inside its token budget.
ALLOWED_TABLES: Final[tuple[str, ...]] = (
    "mart_matchups",
    "mart_archetype_weekly",
    "mart_cards_seen",
    "mart_player_summary",
    "dim_archetype",
    "dim_card",
    "dim_date",
)

# How much of a description survives into the prompt: one sentence, and at most
# this many characters of it. Two of the table descriptions open with a sentence
# that is itself a paragraph, so the sentence split alone is not a budget.
TABLE_SENTENCES: Final = 1
COLUMN_SENTENCES: Final = 1
MAX_TABLE_CHARS: Final = 120
# 56 rather than 62 because the hand written rules grew and the whole prompt
# has to stay inside `MAX_PROMPT_CHARS`. Every line the six characters shortens
# was already ending in an ellipsis, so what they buy a rule costs a column
# description nothing a reader of `schema.yml` cannot get back in full.
MAX_COLUMN_CHARS: Final = 56
# A ceiling the prompt test asserts against. Four characters per token is the
# usual rough conversion, so this is the ~2,000 token budget the ticket set.
MAX_PROMPT_CHARS: Final = 8_000

_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE: Final = re.compile(r"\s+")


def first_sentences(text: str, count: int, limit: int) -> str:
    """The first `count` sentences of a description, on one line, cut to `limit`.

    The yml uses folded block scalars, so a description arrives with its source
    line breaks in it; collapsing the whitespace first is what makes a sentence
    split on ". " work at all. A sentence longer than the limit is cut at the
    last word boundary and given an ellipsis, because the alternative is one
    model's description setting the size of the whole prompt.
    """
    flat = _WHITESPACE.sub(" ", text).strip()
    if not flat:
        return ""
    taken = " ".join(_SENTENCE_END.split(flat)[:count]).strip()
    if len(taken) <= limit:
        return taken
    return taken[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."


def read_models(paths: tuple[Path, ...] = SCHEMA_FILES) -> dict[str, dict[str, object]]:
    """Every model in the given dbt schema files, keyed by model name."""
    models: dict[str, dict[str, object]] = {}
    for path in paths:
        if not path.is_file():
            continue
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for model in parsed.get("models", []):
            name = model.get("name")
            if name:
                models[str(name)] = model
    return models


def render_schema(
    tables: tuple[str, ...] = ALLOWED_TABLES,
    paths: tuple[Path, ...] = SCHEMA_FILES,
) -> str:
    """The allowed tables as a compact schema listing, in the allowlist's order.

    A table named in the allowlist but missing from the schema files is skipped
    rather than raised on: the prompt has to render in a checkout where a model
    was renamed but the allowlist has not caught up yet, and a broken import
    would take the whole serving process down with it.
    """
    models = read_models(paths)
    blocks: list[str] = []
    for table in tables:
        model = models.get(table)
        if model is None:
            continue
        summary = first_sentences(
            str(model.get("description", "")), TABLE_SENTENCES, MAX_TABLE_CHARS
        )
        lines = [f"{table}: {summary}" if summary else f"{table}:"]
        columns = model.get("columns") or []
        assert isinstance(columns, list)
        for column in columns:
            name = column.get("name")
            if not name:
                continue
            note = first_sentences(
                str(column.get("description", "")), COLUMN_SENTENCES, MAX_COLUMN_CHARS
            )
            lines.append(f"  {name} - {note}" if note else f"  {name}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


RULES: Final = """\
Rules you follow on every answer.

1. Cite the sample size. Every number comes with the `games` count it was
   computed over, in the same sentence: a rate without its denominator is not
   an answer on a corpus this small.
2. Say when the mart flags a thin cell. `min_games_met` false means the row is
   under the project's threshold: say so in words, not only the rate.
3. `seen_rate` is the share of games in which a card was observed. It is
   not a deck inclusion rate: a stock export reveals only what was played, so
   report it as an observation and a lower bound. `inclusion_rate` covers only
   the seats that shared a full decklist; when it is null, give the seen count
   and rate under that caveat rather than refusing the question.
4. Never guess a number. If a query returns no rows, or the tool refuses it,
   say what you asked for and that the warehouse does not answer it; never
   estimate or recall a number from outside these tables.
5. Player identity is not available to you. No table holds a name or a handle:
   `mart_player_summary` is keyed by an irreversible token and `dim_player` is
   not readable. Describe the distribution if asked, never a token as a person:
   handles become one-way tokens before anything is written, so there is no
   answer to give.
6. A top is not one row. Asked for the most or the best, name every row tied on
   the top value; `LIMIT 1` turns a tie into an ordering the data does not
   support.
7. Names are stored as they were written, not as a question capitalises them,
   so match them with `ILIKE` or `lower()` rather than `=`. An empty result is
   a spelling to widen before it is an absence to report.

How to work. One SELECT at a time against the tables below: read the rows that
come back and answer from them. The tool appends a LIMIT when you leave one
out. Prefer `archetype_name` over `archetype_key` in the answer, and keep it to
a few sentences."""

# Added only when the retriever's index has been built and the tool is really
# registered. A prompt that advertises a tool the agent does not have is how a
# model ends up describing a lookup it never made.
CARD_TOOL_NOTE: Final = """\
`lookup_cards(query, k)` searches printed card text: abilities, attacks, rules.
It is a reference, not game data, so nothing it returns says how often a card
is played. Use it for what a card does and `query_marts` for every number."""


def override_path() -> Path | None:
    """The replacement prompt file named by the environment, if one is."""
    configured = os.environ.get(PROMPT_FILE_VAR, "").strip()
    return Path(configured) if configured else None


def system_prompt(with_card_tool: bool = False) -> str:
    """The prompt the agent is built with: the generated one, or a replacement.

    Uncached, and cheap anyway: with no override set this is one environment
    lookup in front of the cached builder below, which is the thing that reads
    the schema file. With one set it is a file read per agent built, which is
    once per process on the serving path and once per question in an
    evaluation that is deliberately measuring the prompt.
    """
    override = override_path()
    if override is None:
        return generated_prompt(with_card_tool)
    # Deliberately not merged with anything: an override is the whole prompt,
    # including the card-tool note, so that "the rules are missing" means the
    # rules really are missing.
    return override.read_text(encoding="utf-8")


@lru_cache(maxsize=2)
def generated_prompt(with_card_tool: bool = False) -> str:
    """The whole prompt: what the agent is, the rules, its tools and the schema.

    Cached because it reads a file and a process builds more than one agent: the
    schema cannot change inside a run, and re-reading it per request would put a
    disk read on the serving path for no benefit.
    """
    cards = f"\n\n{CARD_TOOL_NOTE}" if with_card_tool else ""
    return (
        "You answer questions about a Pokemon Trading Card Game metagame from a "
        "small warehouse of parsed battle logs, using the tools you are given. "
        "You are precise about sample size and about what the data cannot say.\n\n"
        f"{RULES}{cards}\n\n"
        "Tables you can query with `query_marts` (DuckDB SQL, read only):\n\n"
        f"{render_schema()}\n"
    )
