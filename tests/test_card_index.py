"""The retriever: the index format and the search, then the tool inside the agent.

Three suites, split the way the rest of the project splits them.

The fast half swaps the embedder. `HashingEmbedder` is deterministic and needs
no download, so the build, the Parquet round trip, the cosine search, the tool
and its instrumentation all run in `uv run pytest` with nothing fetched from
anywhere. What it cannot test is retrieval quality, because it is a bag of
hashed words and not a semantic model.

That is what the `ml` half is for: one test, with the real
`sentence-transformers` model, asserting that "bench damage" finds the card
whose attack puts damage counters on the Bench. It is marked because the first
run downloads a model, which is the same reason the trainer's tests are marked.

The `dbt` half is the one that matters most: the agent holding both tools at
once, answering from the marts and the card text in a single scripted run.
"""

import json
from pathlib import Path
from typing import Final

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pipeline import agent, card_index
from pipeline.telemetry import ServiceMetrics, build_metrics, build_tracer_provider
from tests.agent_fakes import final, scripted, tool_call
from tests.test_agent import attribute

CORPUS: Final = Path(__file__).parent / "card_text.jsonl"
BENCH_CARD: Final = "Dragapult ex"
OTHER_CARD: Final = "Gholdengo ex"


@pytest.fixture(scope="module")
def cards() -> list[card_index.Card]:
    return card_index.read_cards(CORPUS)


@pytest.fixture
def metrics() -> ServiceMetrics:
    return build_metrics()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def hashed_index(cards: list[card_index.Card], tmp_path: Path) -> Path:
    """The fixture corpus indexed with the deterministic embedder."""
    out = tmp_path / "card_index"
    assert card_index.build_index(cards, card_index.HashingEmbedder(), out) == len(cards)
    return out


# ------------------------------------------------------------------ fast --


def test_the_committed_corpus_is_the_shape_the_index_expects(
    cards: list[card_index.Card],
) -> None:
    """The fixture is invented but it has to look like what the fetcher writes."""
    assert len(cards) >= 12
    names = {card.name for card in cards}
    assert {BENCH_CARD, OTHER_CARD} <= names
    dragapult = next(card for card in cards if card.name == BENCH_CARD)
    assert dragapult.hp == 320
    assert any("Bench" in attack["effect"] for attack in dragapult.attacks)
    for card in cards:
        assert card.card_id and card.source_url
        assert card.document().startswith(card.name)


def test_building_writes_vectors_and_a_note_of_which_model_made_them(
    hashed_index: Path, cards: list[card_index.Card]
) -> None:
    """The embedder's name is stored, because a query from another one is nonsense."""
    meta = json.loads((hashed_index / card_index.META_FILE).read_text(encoding="utf-8"))
    assert meta["cards"] == len(cards)
    assert meta["embedder"].startswith(card_index.HASHING_NAME)
    assert meta["dimensions"] == card_index.HASHING_DIM
    assert (hashed_index / card_index.VECTORS_FILE).is_file()


def test_an_index_reloads_without_being_told_which_embedder_built_it(
    hashed_index: Path, cards: list[card_index.Card]
) -> None:
    loaded = card_index.CardIndex.load(hashed_index)
    assert len(loaded.cards) == len(cards)
    assert loaded.embedder.name.startswith(card_index.HASHING_NAME)
    # The whole record survives the round trip, not just the name. A loaded
    # entry is one distinct card with its printings, so the text is under `card`.
    dragapult = next(entry for entry in loaded.cards if entry.name == BENCH_CARD)
    assert dragapult.card.attacks[0]["name"] == "Phantom Dive"
    assert dragapult.card.rules
    assert dragapult.printings and dragapult.printings[0].where


def test_a_missing_index_says_how_to_build_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="card_index build"):
        card_index.CardIndex.load(tmp_path / "nothing")


def test_the_hashing_search_ranks_the_card_that_says_bench_first(hashed_index: Path) -> None:
    """Not a claim about semantics: the one card whose text says Bench wins on words."""
    hits = card_index.CardIndex.load(hashed_index).search("bench damage", k=3)
    assert hits[0].card.name == BENCH_CARD
    assert hits[0].score > 0


def test_the_search_is_bounded_and_deterministic(hashed_index: Path) -> None:
    index = card_index.CardIndex.load(hashed_index)
    assert len(index.search("energy", k=100)) == min(card_index.MAX_K, len(index.cards))
    first = [hit.card.card_id for hit in index.search("draw cards", k=5)]
    second = [hit.card.card_id for hit in index.search("draw cards", k=5)]
    assert first == second


def test_the_tool_returns_readable_card_text_and_counts_its_call(
    hashed_index: Path, metrics: ServiceMetrics, exporter: InMemorySpanExporter
) -> None:
    tracer = build_tracer_provider(exporter=exporter).get_tracer("tests")
    tool = card_index.make_lookup_cards_tool(hashed_index, tracer=tracer, metrics=metrics)
    text = tool.invoke({"query": "bench damage", "k": 2})

    assert BENCH_CARD in text
    assert "Phantom Dive" in text
    spans = {span.name: span for span in exporter.get_finished_spans()}
    tool_span = spans[f"{card_index.TOOL_SPAN_PREFIX}{card_index.CARD_TOOL}"]
    assert attribute(tool_span, "agent.rows") == 2
    # `gate` is `off` on every card lookup: the SQL gate judges statements, and
    # this tool sends none, so there is no verdict to report.
    value = metrics.registry.get_sample_value(
        "agent_tool_calls_total", {"tool": card_index.CARD_TOOL, "gate": "off"}
    )
    assert value == 1.0


def test_an_empty_corpus_is_refused_rather_than_indexed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no cards"):
        card_index.build_index([], card_index.HashingEmbedder(), tmp_path / "index")


def test_a_build_with_no_corpus_file_is_a_logged_skip_and_not_a_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run_all` runs this stage unconditionally, so a missing corpus exits zero."""
    code = card_index.main(
        ["build", "--source", str(tmp_path / "absent.jsonl"), "--out", str(tmp_path / "index")]
    )
    assert code == 0
    assert "no card text" in capsys.readouterr().out


def test_the_command_line_builds_and_queries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "index"
    assert (
        card_index.main(
            ["build", "--source", str(CORPUS), "--out", str(out), "--embedder", "hashing"]
        )
        == 0
    )
    assert "indexed 12 cards" in capsys.readouterr().out
    assert card_index.main(["query", "bench damage", "-k", "1", "--index", str(out)]) == 0
    assert BENCH_CARD in capsys.readouterr().out


# -------------------------------------------------------------------- ml --


@pytest.mark.ml
def test_the_real_embedder_finds_the_bench_damage_card_by_meaning(
    cards: list[card_index.Card], tmp_path: Path
) -> None:
    """The retrieval claim, with the model that makes it. Downloads on first run."""
    pytest.importorskip("sentence_transformers", reason="the embedder is not installed")
    out = tmp_path / "card_index"
    embedder = card_index.SentenceTransformerEmbedder(card_index.DEFAULT_MODEL)
    card_index.build_index(cards, embedder, out)

    hits = card_index.CardIndex.load(out, embedder).search("bench damage", k=3)
    assert BENCH_CARD in [hit.card.name for hit in hits[:3]]


# ------------------------------------------------------------------- dbt --


@pytest.mark.dbt
def test_the_agent_uses_both_tools_in_one_run(
    gold_from_fixtures: Path,
    hashed_index: Path,
    metrics: ServiceMetrics,
    exporter: InMemorySpanExporter,
) -> None:
    """One question, one query against the marts and one lookup against the cards.

    The matchup number is read out of the warehouse by the test, so a tool that
    returned nothing would fail this rather than let a scripted answer through.
    """
    import duckdb

    connection = duckdb.connect(str(gold_from_fixtures), read_only=True)
    row = connection.sql(
        "select archetype_name, games from mart_matchups order by games desc, matchup_key limit 1"
    ).fetchone()
    connection.close()
    assert row is not None
    archetype, games = str(row[0]), int(row[1])

    model = scripted(
        tool_call(
            agent.SQL_TOOL,
            "call-1",
            sql="select archetype_name, games, win_rate from mart_matchups order by games desc",
        ),
        tool_call(card_index.CARD_TOOL, "call-2", query="bench damage", k=2),
        final(f"{archetype} has {games} games in its top matchup, and {BENCH_CARD} is why."),
    )
    tracer = build_tracer_provider(exporter=exporter).get_tracer("tests")
    built = agent.build_agent(
        model=model,
        warehouse=gold_from_fixtures,
        card_index=hashed_index,
        tracer=tracer,
        metrics=metrics,
    )
    assert built.tool_names == [agent.SQL_TOOL, card_index.CARD_TOOL]

    answer = built.ask("what carries this matchup")

    assert str(games) in answer.answer
    assert BENCH_CARD in answer.answer
    assert [call.tool for call in answer.tool_calls] == [agent.SQL_TOOL, card_index.CARD_TOOL]
    names = {span.name for span in exporter.get_finished_spans()}
    assert f"{card_index.TOOL_SPAN_PREFIX}{card_index.CARD_TOOL}" in names
