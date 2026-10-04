"""The plain-language line under each lookup, and the rule it has to keep.

Four suites and no warehouse, because `describe_sql` is a pure function of a
string: the shape of a line for a statement whose parts are all recognisable,
the way it gives up rather than guesses when they are not, the rule the whole
field exists for, which is that no relation name, no column name and no
upper-case SQL keyword may ever reach it, and the vocabulary those lines are
built out of surviving a run with no dbt project beside the package.

The third one is the suite worth having. It is run over every statement in
`evals/transcript.yaml`, which is a recording of the queries a competent run
really writes, rather than over examples written to pass it; the matching
assertion over the statements a live replay produces is in
`tests/test_eval.py`, where the warehouse already exists.
"""

import ast
import re
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from pipeline import describe, prompts
from pipeline.prompts import ALLOWED_TABLES, read_models, warehouse_tables

TRANSCRIPT: Final = Path(__file__).parents[1] / "evals" / "transcript.yaml"

# The keywords a description may not contain in upper case. Not the whole of
# DuckDB's grammar: these are the words a statement is written out of, so a
# line with one of them in capitals is a line that has leaked a fragment of
# SQL rather than one that happens to use an English word.
SQL_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "GROUP",
        "ORDER",
        "BY",
        "LIMIT",
        "HAVING",
        "WITH",
        "AS",
        "AND",
        "OR",
        "NOT",
        "IN",
        "ON",
        "UNION",
        "DISTINCT",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "NULL",
        "LIKE",
        "ILIKE",
        "BETWEEN",
        "DESC",
        "ASC",
        "DATE",
        "INTERVAL",
        "IS",
        "ALL",
        "EXISTS",
    }
)

_WORD: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def transcript_statements() -> list[tuple[str, str]]:
    """Every recorded `query_marts` statement, with the run id it belongs to."""
    parsed: dict[str, Any] = yaml.safe_load(TRANSCRIPT.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for run_id, run in parsed["runs"].items():
        for turn in run.get("turns") or []:
            sql = (turn.get("args") or {}).get("sql")
            if sql:
                found.append((str(run_id), str(sql)))
    return found


def assert_reads_as_plain_language(line: str) -> None:
    """The one rule, as the two assertions it is made of.

    Shared by this module and by `tests/test_eval.py`, so that the replay and
    the transcript are held to the same bar and the bar is written once.
    """
    forbidden = describe.raw_names()
    for word in _WORD.finditer(line):
        assert word.group(0).lower() not in forbidden, f"{line!r} names {word.group(0)!r}"
        assert word.group(0) not in SQL_KEYWORDS, f"{line!r} shouts {word.group(0)!r}"
    assert len(line) <= describe.MAX_DESCRIPTION_CHARS, f"{line!r} is {len(line)} characters"


# ----------------------------------------------------------- the vocabulary --


def test_every_table_the_agent_may_read_has_words_of_its_own() -> None:
    """A table on the allowlist with no phrase is a lookup the receipt cannot name."""
    assert set(describe.MART_PHRASES) == set(ALLOWED_TABLES)


def test_every_named_column_is_a_column_of_a_table_the_agent_may_read() -> None:
    """The hand table is checked against dbt, so a rename is a red test and not a typo.

    `schema.yml` is the one description of the warehouse this project has
    written down, and the prompt is generated from it for the same reason. A
    phrase here for a column that no longer exists is a phrase that can never
    be used and nothing else would notice.
    """
    columns: set[str] = set()
    for table, model in read_models().items():
        if table not in ALLOWED_TABLES:
            continue
        listed = model.get("columns") or []
        assert isinstance(listed, list)
        columns.update(str(column.get("name") or "") for column in listed)
    assert set(describe.COLUMN_WORDS) <= columns
    assert set(describe.DATE_COLUMNS) <= columns
    assert set(describe.RATE_HEADS) <= columns


def test_no_phrase_in_either_table_says_a_name_out_loud() -> None:
    """The vocabulary is the thing the rule rests on, so the rule is checked on it."""
    for phrase in (*describe.MART_PHRASES.values(), *describe.COLUMN_WORDS.values()):
        assert_reads_as_plain_language(phrase)


# ------------------------------------------------------------- the shapes --


def test_a_filtered_rate_reads_as_the_rate_and_the_filters() -> None:
    line = describe.describe_sql(
        "select archetype_name, win_rate from mart_matchups "
        "where archetype_name ilike 'Dragapult control' "
        "and opponent_archetype_name ilike 'Alakazam / Toucannon'"
    )
    assert line == (
        "Win rate over matchup results, for the deck = Dragapult control "
        "and the opposing deck = Alakazam / Toucannon"
    )


def test_a_counting_query_says_so_and_names_the_mart() -> None:
    assert describe.describe_sql("select count(*) from mart_player_summary") == (
        "A count over per-player summaries"
    )


def test_an_order_and_a_limit_read_as_a_top() -> None:
    assert describe.describe_sql(
        "select card_name, seen_rate from mart_cards_seen order by seen_rate desc limit 10"
    ) == (
        "How often cards were seen over which cards showed up, top 10 by how often a card was seen"
    )


def test_an_order_with_no_limit_says_which_end_is_first() -> None:
    assert describe.describe_sql(
        "select archetype_name, sum(games) as games from mart_archetype_weekly "
        "group by archetype_name order by games desc"
    ) == (
        "A total of matches over how each deck did week by week, ordered by matches, highest first"
    )


def test_a_date_filter_becomes_a_week_and_a_range_becomes_a_window() -> None:
    assert describe.describe_sql(
        "select archetype_name from mart_archetype_weekly where week_start = date '2026-09-14'"
    ) == ("A lookup over how each deck did week by week, for the week beginning 2026-09-14")
    assert describe.describe_sql(
        "select play_date from dim_date where play_date >= date '2026-09-01' "
        "and play_date <= date '2026-09-30'"
    ) == ("A lookup over the calendar, from 2026-09-01 to 2026-09-30")


def test_a_month_is_named_rather_than_numbered() -> None:
    """`month` and `year` are columns of the calendar as well as ordinary words."""
    assert describe.describe_sql("select day_name from dim_date where month = 9") == (
        "A lookup over the calendar, in September"
    )


def test_a_doubled_quote_comes_back_as_the_apostrophe_it_is() -> None:
    line = describe.describe_sql(
        "select games from mart_matchups where archetype_name ilike 'Cynthia''s Garchomp'"
    )
    assert "Cynthia's Garchomp" in line


def test_a_boolean_filter_reads_as_yes_or_no() -> None:
    assert describe.describe_sql(
        "select archetype_name from mart_matchups where min_games_met = true"
    ) == ("A lookup over matchup results, for enough matches to count = yes")


# ---------------------------------------------------- giving up gracefully --


def test_a_table_the_tool_cannot_read_is_never_named() -> None:
    """A guessed name and a blocked one both land here, and neither reaches the screen.

    This is the description a refused query carries. Why it was refused is
    `refused_code`, which the application has its own words for; the line
    itself says what the lookup was over and stops.
    """
    for sql in (
        "select * from mart_leaderboard",
        "select * from dim_player",
        "select * from fct_game_side join dim_player on true",
        "select getenv('ANTHROPIC_API_KEY')",
        "",
    ):
        assert describe.describe_sql(sql) == describe.UNKNOWN_LOOKUP


def test_an_either_or_filter_is_left_out_rather_than_reported_as_both() -> None:
    """ "A and B" would be a lie about a statement that asked for either."""
    assert describe.describe_sql(
        "select archetype_name from mart_matchups where archetype_name ilike 'a' "
        "or archetype_name ilike 'b'"
    ) == ("A lookup over matchup results")


def test_a_filter_on_a_column_with_no_words_is_dropped_not_guessed() -> None:
    line = describe.describe_sql(
        "select archetype_name from mart_matchups where matchup_key = 'a vs b'"
    )
    assert line == "A lookup over matchup results"


def test_a_long_line_is_cut_at_a_word_and_never_mid_name() -> None:
    sql = (
        "select win_rate from mart_matchups join dim_archetype on true "
        "join dim_date on true join dim_card on true "
        "where archetype_name ilike 'a deck with a very long and quite unhelpful name' "
        "and opponent_archetype_name ilike 'another deck with an equally long name'"
    )
    line = describe.describe_sql(sql)
    assert len(line) <= describe.MAX_DESCRIPTION_CHARS
    assert line.endswith("...")
    assert_reads_as_plain_language(line)


# --------------------------------------------------------------- the rule --


@pytest.mark.parametrize(("run_id", "sql"), transcript_statements())
def test_no_recorded_statement_produces_a_line_with_a_name_in_it(run_id: str, sql: str) -> None:
    """The rule, over the queries a competent run really writes.

    `evals/transcript.yaml` is the recording the whole evaluation replays, so
    these are not examples chosen to pass: they are the statements the golden
    set is built on, and a phrase added to `pipeline.describe` that puts a
    column name back on the screen fails here first.
    """
    line = describe.describe_sql(sql)
    assert line, run_id
    assert_reads_as_plain_language(line)


def test_the_same_statement_always_produces_the_same_line() -> None:
    """Pure, which is the half of the design that makes the line checkable."""
    sql = "select win_rate from mart_matchups where archetype_name ilike 'Dragapult control'"
    assert describe.describe_sql(sql) == describe.describe_sql(sql)


# ------------------------------------------- the vocabulary without a checkout --


def _clear_caches() -> None:
    """Every cache the two vocabularies sit behind, emptied.

    All three, because they feed each other: `describe_sql` caches a line,
    `schema_column_words` caches the words that line was built out of, and
    `raw_names` caches the set both of them are checked against.
    """
    describe.describe_sql.cache_clear()
    describe.schema_column_words.cache_clear()
    describe.raw_names.cache_clear()


def _file_based_column_words() -> dict[str, str]:
    """`schema_column_words` as it read the schema file before PLA-207.

    The oracle for the move to `pipeline.marts_schema`: the same filter in the
    same order over the parse of `dbt/models/marts/schema.yml`, so a committed
    module that has drifted from the file shows up as a vocabulary that
    differs rather than as a receipt line nobody is looking at.
    """
    forbidden = describe.raw_names()
    words: dict[str, str] = {}
    for model in read_models().values():
        listed = model.get("columns") or []
        assert isinstance(listed, list)
        for column in listed:
            name = str(column.get("name") or "")
            if not name or name in describe.COLUMN_WORDS:
                continue
            sentence = describe.first_sentences(
                str(column.get("description", "")),
                describe.COLUMN_SENTENCES,
                describe.MAX_VALUE_CHARS,
            )
            phrase = sentence.rstrip(".").strip()
            if not phrase or phrase.endswith("...") or len(phrase) > 36:
                continue
            if any(word.group(0).lower() in forbidden for word in _WORD.finditer(phrase)):
                continue
            words[name] = phrase[0].lower() + phrase[1:]
    return words


def _file_based_raw_names() -> frozenset[str]:
    """`raw_names` as it read the schema file before PLA-207, the same oracle."""
    names = set(ALLOWED_TABLES) | set(warehouse_tables())
    for table, model in read_models().items():
        if table not in ALLOWED_TABLES:
            continue
        listed = model.get("columns") or []
        assert isinstance(listed, list)
        names.update(str(column.get("name") or "").lower() for column in listed)
    names.discard("")
    return frozenset(names)


def test_the_borrowed_words_are_the_dbt_schema_read_from_the_committed_module() -> None:
    """The module the receipt reads says what the file says, column for column.

    `pipeline.marts_schema` is generated from `dbt/models/marts/schema.yml`
    and a test in `tests/test_agent.py` keeps the two in step, so this is a
    second reading of the same promise from the side that depends on it: the
    words are the ones the file would have produced, and the move off the
    filesystem changed no line.
    """
    assert describe.schema_column_words() == _file_based_column_words()
    assert describe.raw_names() == _file_based_raw_names()


def test_every_recorded_statement_reads_the_same_without_the_dbt_tree() -> None:
    """The deployed image, as the only thing that is different about it.

    `Dockerfile.agent` copies `pipeline/` and `scripts/`, so there is no
    `dbt/` under the function's root at all. Pointing the two constants at a
    directory that is not there is that image in one line, and it is a
    better test than deleting the files because it leaves the checkout alone:
    the suite stays parallel-safe and a failure here is a missing read rather
    than a missing fixture.

    Every statement in the transcript rather than one, because a vocabulary
    that came back short would show up as a filter quietly dropped from a
    line and not as an exception.
    """
    before = {sql: describe.describe_sql(sql) for _run_id, sql in transcript_statements()}
    names = describe.raw_names()
    words = describe.schema_column_words()
    assert before
    assert names
    assert words
    _clear_caches()
    try:
        with pytest.MonkeyPatch.context() as patch:
            nowhere = Path("/nonexistent-dbt-tree-for-pla-207")
            patch.setattr(prompts, "MARTS_SCHEMA", nowhere / "models" / "marts" / "schema.yml")
            patch.setattr(prompts, "SCHEMA_FILES", (nowhere / "models" / "marts" / "schema.yml",))
            patch.setattr(prompts, "DBT_MODELS_DIR", nowhere / "models")
            assert read_models(prompts.SCHEMA_FILES) == {}
            assert describe.raw_names() == names
            assert describe.schema_column_words() == words
            after = {sql: describe.describe_sql(sql) for _run_id, sql in transcript_statements()}
        assert after == before
    finally:
        _clear_caches()


def test_no_serving_module_reaches_for_the_dbt_tree() -> None:
    """Nothing under `pipeline/` reads the dbt project except the two build-time parsers.

    A source scan and not a mock, because the failure this guards against is
    silent by construction: a read of `dbt/` that finds nothing on the image
    returns an empty answer rather than raising, and the symptom is a shorter
    prompt or a thinner receipt in production and a green suite everywhere
    else. PLA-198 was that bug in `warehouse_tables`, PLA-206 in the prompt's
    schema listing and PLA-207 in these two vocabularies, so the third time it
    is a rule.

    The names are matched as code rather than as text, so a docstring may go
    on saying where the committed data came from. The allowlist is the two
    constants' own module: they are defined there, `dbt_model_names` globs the
    directory for `scripts/generate_warehouse_tables.py`, and `read_models`
    and `marts_models_from_files` parse the file for
    `scripts/generate_marts_schema.py`. Every one of those runs in a checkout
    and none of them on a serving path.
    """
    scanned = {
        "MARTS_SCHEMA",
        "SCHEMA_FILES",
        "DBT_MODELS_DIR",
        "read_models",
        "dbt_model_names",
        "marts_models_from_files",
    }
    allowed = {
        ("prompts.py", "<module>"),
        ("prompts.py", "dbt_model_names"),
        ("prompts.py", "read_models"),
        ("prompts.py", "marts_models_from_files"),
    }
    package = Path(describe.__file__).parent
    found: set[tuple[str, str]] = set()
    for path in sorted(package.glob("**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner, node in _owned_nodes(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in scanned:
                found.add((str(path.relative_to(package)), owner))
    assert found == allowed, (
        f"unexpected dbt-tree readers under pipeline/: {sorted(found - allowed)}"
    )


def _owned_nodes(tree: ast.Module) -> list[tuple[str, ast.Name | ast.Attribute]]:
    """Every name and attribute in a module, with the top-level definition it sits in.

    `<module>` for anything outside a `def` or a `class`, and the outermost
    definition's own name for everything else, which is the granularity the
    allowlist above is written at.
    """
    owned: list[tuple[str, ast.Name | ast.Attribute]] = []
    for top in tree.body:
        owner = getattr(top, "name", "<module>")
        for node in ast.walk(top):
            if isinstance(node, ast.Name | ast.Attribute):
                owned.append((str(owner), node))
    return owned
