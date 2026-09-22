"""The retriever: card text embedded locally, searched by cosine, read as a tool.

    uv run python -m pipeline.card_index build
    uv run python -m pipeline.card_index query "bench damage" -k 5

`scripts/fetch_card_text.py` writes the corpus, one JSON object per line per
card, from a public card API. This module turns that into an index and answers
`lookup_cards(query, k)`, the agent's second tool.

**Local embeddings, no API.** The vectors come from `sentence-transformers`
with `BAAI/bge-small-en-v1.5`, a 384-dimensional model that runs on a laptop
CPU in a few seconds for the whole corpus. Card text is short, domain-specific
and never changes, so the index is built once and read many times, and an
embedding API would add a key, a bill and a network hop to something that is
already fast. `all-MiniLM-L6-v2` is the fallback if bge is not available and is
selected with `--embedder`; both are 384 dimensions and either can be swapped
without touching the storage format.

**Parquet plus a numpy dot product, not DuckDB's `vss`.** The choice was
between an approximate-nearest-neighbour index in the warehouse and a brute
force scan over a matrix, and at this size the scan wins on every axis that
matters. The corpus is about 25,000 cards at 384 float32, which is 38 MB and a
single 25,000 by 384 matrix multiplication per query: well under a millisecond,
exact rather than approximate, with no recall to tune. `vss` would add an
extension to install at build time (a download, so a machine with no network
could not build an index), an HNSW index whose persistence in a file-backed
database is still behind an experimental flag, and a second copy of the card
text inside the warehouse the SQL tool is deliberately restricted from reading.
The day this indexes millions of rows the trade goes the other way, and the
storage is one file and one loader, so it is a small day's work.

**The embedder is an interface with two implementations.** `HashingEmbedder` is
deterministic, dependency-free and needs no download: it hashes word tokens
into a fixed number of buckets. It is not a semantic model and it is not
pretending to be one; it exists so the index format, the build, the search and
the tool all run in the fast test suite, and so `--embedder hashing` gives a
usable keyword-ish search on a machine that cannot download a model. The
marker on the real-model tests is `ml` for the same reason the trainer's is.

The index directory holds two files: `vectors.parquet`, one row per card with
its display fields, the document that was embedded and its vector, and
`meta.json`, which records which embedder built it. A query embedded by a
different model than the index would return confident nonsense, so the loader
checks the name and refuses instead.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from opentelemetry import trace

from pipeline.config import CARD_INDEX_DIR, CARD_TEXT_PATH
from pipeline.observability import configure_logging, emit_summary, stage_run
from pipeline.telemetry import ServiceMetrics

logger = logging.getLogger(__name__)

STAGE: Final = "card_index"

VECTORS_FILE: Final = "vectors.parquet"
META_FILE: Final = "meta.json"

DEFAULT_MODEL: Final = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
# bge is trained with an instruction on the query side and none on the document
# side, and skipping it costs real accuracy: without the prefix, "put damage
# counters on the bench" does not rank the card that does exactly that first,
# and with it, it does. Only the bge family wants it, which is why it is applied
# by model name rather than to everything.
BGE_QUERY_PREFIX: Final = "Represent this sentence for searching relevant passages: "
HASHING_NAME: Final = "hashing"
HASHING_DIM: Final = 256
# How much of a word the hashing embedder keeps as a stem. Five is enough to
# make "bench" and "Benched" the same feature and short enough that it does not
# collapse "damage" onto "damaged" by accident, which it also does, on purpose.
STEM_CHARS: Final = 5

DEFAULT_K: Final = 5
MAX_K: Final = 20
# Effect text is the long field, and a tool result holding five whole cards is
# already a page. Cut each effect rather than the number of cards: the model
# asked for k cards and silently getting three would be the worse surprise.
MAX_EFFECT_CHARS: Final = 220

CARD_TOOL: Final = "lookup_cards"
TOOL_SPAN_PREFIX: Final = "agent.tool."

_TOKEN: Final = re.compile(r"[a-z0-9']+")


# ----------------------------------------------------------------- corpus --


@dataclass(frozen=True)
class Card:
    """One printed card, as `scripts/fetch_card_text.py` writes it.

    Everything is optional except the identifier and the name, because the API
    this is built from describes a Trainer, a basic Energy and a Stage 2
    Pokemon with the same object and most of the fields are absent on two of
    the three.
    """

    card_id: str
    name: str
    set_name: str = ""
    number: str = ""
    types: list[str] = field(default_factory=list)
    hp: int | None = None
    stage: str = ""
    abilities: list[dict[str, str]] = field(default_factory=list)
    attacks: list[dict[str, Any]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    retreat: int | None = None
    regulation_mark: str = ""
    source_url: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Card":
        """One JSONL record as a card, tolerating a missing optional field."""
        return cls(
            card_id=str(record.get("card_id", "")),
            name=str(record.get("name", "")),
            set_name=str(record.get("set") or ""),
            number=str(record.get("number") or ""),
            types=[str(value) for value in record.get("types") or []],
            hp=_as_int(record.get("hp")),
            stage=str(record.get("stage") or ""),
            abilities=[
                {"name": str(item.get("name", "")), "effect": str(item.get("effect", ""))}
                for item in record.get("abilities") or []
            ],
            attacks=[
                {
                    "name": str(item.get("name", "")),
                    "cost": [str(value) for value in item.get("cost") or []],
                    "damage": str(item.get("damage") or ""),
                    "effect": str(item.get("effect", "")),
                }
                for item in record.get("attacks") or []
            ],
            rules=[str(value) for value in record.get("rules") or []],
            retreat=_as_int(record.get("retreat")),
            regulation_mark=str(record.get("regulation_mark") or ""),
            source_url=str(record.get("source_url") or ""),
        )

    def document(self) -> str:
        """The text that gets embedded: everything a rules question could match on.

        Name first and repeated nowhere, effect text in full, and the costs and
        hit points left out of the sentence they would pad. A document is one
        card, not one attack, because "which card does X" is the question and a
        per-attack document would return the same card three times.
        """
        parts = [self.name]
        header = " ".join(filter(None, [self.stage, "/".join(self.types)]))
        if header:
            parts.append(header)
        if self.hp:
            parts.append(f"{self.hp} HP")
        for ability in self.abilities:
            parts.append(f"Ability {ability['name']}: {ability['effect']}")
        for attack in self.attacks:
            damage = f" {attack['damage']} damage" if attack.get("damage") else ""
            parts.append(f"Attack {attack['name']}:{damage} {attack.get('effect', '')}".rstrip())
        parts.extend(self.rules)
        return ". ".join(part.strip() for part in parts if part and part.strip())

    def render(self) -> str:
        """The card as the tool prints it: a heading and its text, a few lines."""
        where = " ".join(filter(None, [self.set_name, self.number]))
        head = f"**{self.name}**" + (f" ({where})" if where.strip() else "")
        lines = [head]
        traits = ", ".join(
            filter(
                None,
                [
                    self.stage,
                    "/".join(self.types),
                    f"{self.hp} HP" if self.hp else "",
                    f"regulation {self.regulation_mark}" if self.regulation_mark else "",
                ],
            )
        )
        if traits:
            lines.append(traits)
        for ability in self.abilities:
            lines.append(f"Ability {ability['name']}: {_clip(ability['effect'])}")
        for attack in self.attacks:
            cost = "".join(item[:1].upper() for item in attack.get("cost") or [])
            damage = attack.get("damage") or "-"
            lines.append(
                f"Attack [{cost or '-'}] {attack['name']}, {damage}: "
                f"{_clip(str(attack.get('effect', '')))}".rstrip()
            )
        for rule in self.rules:
            lines.append(_clip(rule))
        return "\n".join(lines)


def _as_int(value: Any) -> int | None:
    """An integer field that may arrive as an int, a numeric string or nothing."""
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _clip(text: str, limit: int = MAX_EFFECT_CHARS) -> str:
    """Effect text on one line, cut to a length a tool result can afford."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def read_cards(source: Path) -> list[Card]:
    """Every card in a JSONL corpus. A malformed line is skipped and counted."""
    cards: list[Card] = []
    skipped = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                skipped += 1
                continue
            card = Card.from_record(record)
            if card.card_id and card.name:
                cards.append(card)
            else:
                skipped += 1
    if skipped:
        logger.warning("skipped unreadable card records", extra={"skipped": skipped})
    return cards


# -------------------------------------------------------------- embedders --


class Embedder(Protocol):
    """What the index needs from an embedder: a name, a width and a matrix."""

    @property
    def name(self) -> str:
        """The identifier written into `meta.json` and checked on load."""

    @property
    def dimensions(self) -> int:
        """How wide a vector is."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """One L2-normalized row per text, as float32. Documents, not queries."""

    def embed_query(self, text: str) -> np.ndarray:
        """One vector for a search query, in the same space as the documents."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length, so a dot product is a cosine.

    A zero row stays zero rather than becoming a division by zero: a card with
    no text is a card that matches nothing, which is the honest answer.
    """
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normalized: np.ndarray = (matrix / lengths).astype(np.float32)
    return normalized


class HashingEmbedder:
    """Word tokens hashed into fixed buckets. Deterministic, and no download.

    Not a semantic model: "bench damage" finds a card that says bench and
    damage, not one that says "your opponent's other Pokemon". That is enough
    for the fast test suite, which is about the index format, the search and
    the tool rather than about retrieval quality, and it is a usable keyword
    search on a machine that cannot fetch a model.

    Sublinear term frequency, `1 + log(count)`, for the same reason every bag
    of words uses it: a card whose effect says "damage" four times is not four
    times more about damage than one that says it once.

    A token longer than the stem length is hashed twice, once whole and once
    cut to its first five characters. It is the crudest possible stemmer and it
    is there for exactly one thing: printed card text says "Benched" and a
    person searching says "bench", and a bag of whole words scores those two at
    zero. The real embedder has no such problem, which is why this one is only
    ever the fallback.
    """

    def __init__(self, dimensions: int = HASHING_DIM) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"{HASHING_NAME}-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for token in _TOKEN.findall(text.lower()):
                for feature in self._features(token):
                    bucket = self._bucket(feature)
                    counts[bucket] = counts.get(bucket, 0) + 1
            for bucket, count in counts.items():
                matrix[row, bucket] = 1.0 + math.log(count)
        return _normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        """The same bag of hashed tokens: this embedder has no query side."""
        row: np.ndarray = self.embed([text])[0]
        return row

    @staticmethod
    def _features(token: str) -> tuple[str, ...]:
        """The token, and its five-character stem when it has one to spare."""
        if len(token) > STEM_CHARS:
            return (token, token[:STEM_CHARS])
        return (token,)

    def _bucket(self, token: str) -> int:
        """A stable bucket for a token, across processes and Python versions.

        `hash()` is salted per process, so an index built in one run would not
        be searchable from the next. blake2b is not, and is fast enough that
        hashing a few hundred thousand tokens is not the slow part of anything.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimensions


class SentenceTransformerEmbedder:
    """`sentence-transformers` over a small local model. The real one.

    The model is loaded on first use rather than in the constructor, so
    building the object costs nothing and a process that ends up not embedding
    anything never pays for torch.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._dimensions = 0

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        if not self._dimensions:
            self._dimensions = int(self._loaded().get_sentence_embedding_dimension())
        return self._dimensions

    def _loaded(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading the embedding model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_query(self, text: str) -> np.ndarray:
        """The query with its model's instruction prefix, when its model wants one."""
        prefix = BGE_QUERY_PREFIX if "bge" in self._model_name.lower() else ""
        row: np.ndarray = self.embed([prefix + text])[0]
        return row

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._loaded().encode(
            list(texts),
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    """The embedder named on the command line, or in an index's `meta.json`."""
    if name.startswith(HASHING_NAME):
        _, _, width = name.partition("-")
        return HashingEmbedder(int(width) if width.isdigit() else HASHING_DIM)
    return SentenceTransformerEmbedder(name)


# ----------------------------------------------------------------- index --


_SCHEMA: Final = pa.schema(
    [
        pa.field("card_id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("record_json", pa.string(), nullable=False),
        pa.field("document", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32()), nullable=False),
    ]
)


def build_index(cards: Sequence[Card], embedder: Embedder, out_dir: Path) -> int:
    """Embed every card and write the index. Returns the number of rows written.

    The whole record travels into the Parquet as JSON beside its vector, so a
    search result is a card rather than an identifier that has to be looked up
    again in the source file the index may well outlive.
    """
    if not cards:
        raise ValueError("there are no cards to index")
    documents = [card.document() for card in cards]
    vectors = embedder.embed(documents)
    if vectors.shape[0] != len(cards):
        raise ValueError(f"the embedder returned {vectors.shape[0]} rows for {len(cards)} cards")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "card_id": [card.card_id for card in cards],
            "name": [card.name for card in cards],
            "record_json": [json.dumps(_as_record(card), sort_keys=True) for card in cards],
            "document": documents,
            "vector": [row.tolist() for row in vectors],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, out_dir / VECTORS_FILE)
    (out_dir / META_FILE).write_text(
        json.dumps(
            {
                "embedder": embedder.name,
                "dimensions": int(vectors.shape[1]),
                "cards": len(cards),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(cards)


def _as_record(card: Card) -> dict[str, Any]:
    """A card back in the shape the corpus file uses."""
    return {
        "card_id": card.card_id,
        "name": card.name,
        "set": card.set_name,
        "number": card.number,
        "types": card.types,
        "hp": card.hp,
        "stage": card.stage,
        "abilities": card.abilities,
        "attacks": card.attacks,
        "rules": card.rules,
        "retreat": card.retreat,
        "regulation_mark": card.regulation_mark,
        "source_url": card.source_url,
    }


@dataclass(frozen=True)
class Hit:
    """One search result: the card and how close it was."""

    card: Card
    score: float


class CardIndex:
    """A built index in memory: the cards, their vectors, and the embedder's name."""

    def __init__(self, cards: Sequence[Card], vectors: np.ndarray, embedder: Embedder) -> None:
        self.cards = list(cards)
        self.vectors = vectors
        self.embedder = embedder

    @classmethod
    def load(cls, directory: Path, embedder: Embedder | None = None) -> "CardIndex":
        """Read an index off disk, with the embedder its `meta.json` names.

        An embedder passed in wins, which is how a test builds with the hashing
        one and searches with it too; otherwise the name in the metadata
        decides, because a query vector from a different model is not in the
        same space as the index and the results would look plausible and be
        meaningless.
        """
        meta_path = directory / META_FILE
        vectors_path = directory / VECTORS_FILE
        if not meta_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError(
                f"no card index at {directory}: run `python -m pipeline.card_index build`"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        table = pq.read_table(vectors_path)
        cards = [
            Card.from_record(json.loads(str(value)))
            for value in table.column("record_json").to_pylist()
        ]
        vectors = np.asarray(table.column("vector").to_pylist(), dtype=np.float32)
        resolved = embedder if embedder is not None else make_embedder(str(meta["embedder"]))
        if embedder is not None and embedder.name != meta["embedder"]:
            logger.warning(
                "searching an index with a different embedder than built it",
                extra={"index_embedder": meta["embedder"], "query_embedder": embedder.name},
            )
        return cls(cards, vectors, resolved)

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        """The k closest cards by cosine, best first.

        One matrix-vector product over the whole corpus. Exact, and at 25,000
        rows by 384 columns fast enough that there is nothing to optimize.
        """
        wanted = max(1, min(k, MAX_K))
        if not self.cards:
            return []
        vector = self.embedder.embed_query(query)
        scores = self.vectors @ vector
        order = np.argsort(-scores)[:wanted]
        return [Hit(card=self.cards[int(i)], score=float(scores[int(i)])) for i in order]


def render_hits(hits: Sequence[Hit]) -> str:
    """Search results as the text the tool returns."""
    if not hits:
        return "no cards matched."
    blocks = [f"{hit.card.render()}\n(similarity {hit.score:.2f})" for hit in hits]
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ tool --


def make_lookup_cards_tool(
    index_dir: Path,
    *,
    tracer: trace.Tracer,
    metrics: ServiceMetrics,
    embedder: Embedder | None = None,
) -> Any:
    """The card-text tool, bound to one built index.

    The index is loaded here, when the tool is made, rather than on the first
    call: an agent that is going to fail because its index is missing should
    fail while it is being built, not in the middle of answering.
    """
    from langchain_core.tools import StructuredTool

    from pipeline.agent import ToolCall, record_call, summarize

    index = CardIndex.load(index_dir, embedder)

    def lookup_cards(query: str, k: int = DEFAULT_K) -> str:
        """Find printed cards whose text matches a description."""
        with tracer.start_as_current_span(f"{TOOL_SPAN_PREFIX}{CARD_TOOL}") as span:
            span.set_attribute("agent.tool", CARD_TOOL)
            span.set_attribute("agent.query.length", len(query))
            hits = index.search(query, k)
            span.set_attribute("agent.rows", len(hits))
        metrics.count_tool_call(CARD_TOOL)
        record_call(ToolCall(tool=CARD_TOOL, input_summary=summarize(query), rows=len(hits)))
        logger.info("tool call", extra={"tool": CARD_TOOL, "rows": len(hits)})
        return render_hits(hits)

    return StructuredTool.from_function(
        func=lookup_cards,
        name=CARD_TOOL,
        description=(
            "Search printed card text by meaning: names, types, abilities, attacks and "
            "rules. Returns the top k cards with their text. This is a card reference, "
            "not game data: it says nothing about how often a card is played."
        ),
    )


# ------------------------------------------------------------ entry point --


def build(source: Path, out_dir: Path, embedder: Embedder) -> int:
    """Read the corpus, build the index, and say what was written."""
    cards = read_cards(source)
    written = build_index(cards, embedder, out_dir)
    logger.info(
        "card index built",
        extra={
            "cards": written,
            "embedder": embedder.name,
            "source": str(source),
            "out": str(out_dir),
        },
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """Build the index, or query it, from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.card_index",
        description="Build and query the card-text index the agent's retriever uses.",
    )
    subcommands = parser.add_subparsers(dest="command")

    builder = subcommands.add_parser("build", help="embed the card corpus into an index")
    builder.add_argument(
        "--source",
        type=Path,
        default=CARD_TEXT_PATH,
        metavar="PATH",
        help=f"the JSONL corpus to read (default: {CARD_TEXT_PATH})",
    )
    builder.add_argument(
        "--out",
        type=Path,
        default=CARD_INDEX_DIR,
        metavar="PATH",
        help=f"where to write the index (default: {CARD_INDEX_DIR})",
    )
    builder.add_argument(
        "--embedder",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=(
            f"model name, `{HASHING_NAME}` for the offline one, or {FALLBACK_MODEL} "
            f"(default: {DEFAULT_MODEL})"
        ),
    )

    query = subcommands.add_parser("query", help="search a built index")
    query.add_argument("text", help="what to search for")
    query.add_argument("-k", type=int, default=DEFAULT_K, help=f"results (default: {DEFAULT_K})")
    query.add_argument(
        "--index",
        type=Path,
        default=CARD_INDEX_DIR,
        metavar="PATH",
        help=f"the index to read (default: {CARD_INDEX_DIR})",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    configure_logging(STAGE)

    if args.command == "query":
        index = CardIndex.load(args.index)
        hits = index.search(args.text, args.k)
        sys.stdout.write(render_hits(hits) + "\n")
        return 0

    source: Path = args.source
    if not source.is_file():
        # Not an error. The corpus is fetched from a public API and is not
        # committed, so a clone that has not fetched it has nothing to index,
        # which is the same kind of nothing a missing card catalog is.
        logger.warning(
            "no card text to index",
            extra={"source": str(source), "hint": "run scripts/fetch_card_text.py"},
        )
        emit_summary(
            logger,
            "card index skipped",
            {"source": str(source), "reason": "the card text corpus is missing"},
            text=f"no card text at {source}; run scripts/fetch_card_text.py first",
        )
        return 0

    with stage_run(STAGE) as metrics:
        written = build(source, args.out, make_embedder(args.embedder))
        metrics.rows_in = written
        metrics.rows_out = written
        metrics.rows_quarantined = 0
        metrics.extra = {"embedder": args.embedder, "out": str(args.out)}

    emit_summary(
        logger,
        "card index built",
        {"cards": written, "embedder": args.embedder, "out": str(args.out)},
        text=f"indexed {written} cards into {args.out} with {args.embedder}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
