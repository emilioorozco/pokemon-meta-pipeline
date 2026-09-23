"""The retriever: card text embedded locally, searched by cosine, read as a tool.

    uv run python -m pipeline.card_index build
    uv run python -m pipeline.card_index query "bench damage" -k 5

`scripts/fetch_card_text.py` writes the corpus, one JSON object per line per
printing, from a public card API. This module turns that into an index and
answers `lookup_cards(query, k)`, the agent's second tool.

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
matters. The Standard corpus is a few thousand passages at 384 float32, which
is a handful of megabytes and a single matrix-vector product per query: well
under a millisecond, exact rather than approximate, with no recall to tune.
`vss` would add an extension to install at build time (a download, so a machine
with no network could not build an index), an HNSW index whose persistence in a
file-backed database is still behind an experimental flag, and a second copy of
the card text inside the warehouse the SQL tool is deliberately restricted from
reading. The day this indexes millions of rows the trade goes the other way,
and the storage is one file and one loader, so it is a small day's work.

**One row per passage, aggregated back to the card.** An earlier version
embedded a whole card as a single blob and argued that a per-attack document
would return the same card three times. The blob was the bug: a Stage 2 with
two attacks, an ability and an ex rule is a paragraph, and the one sentence a
rules question is aiming at is a fifth of it, so the card loses to a Stadium
whose entire text is that one idea. Dragapult ex scored 0.73 against a verbatim
quote of its own attack and ranked 39th. The fix is to embed each ability,
attack and rule on its own, prefixed with the card name so a passage is
self-describing, plus one identity passage for the name, stage, types and hit
points. Returning the same card three times is a ranking problem, not a reason
to keep the blob: passages are scored, the best passage per card is taken as
the card's score, and a card appears once, with the passage that matched named
in the result so the answer can say which attack it means.

**Reprints collapse to one card.** The corpus is printings, and the Standard
format reprints heavily: 2,264 printings are 1,145 names. Five rows of
Dragapult ex with identical text are one card that happens to have five
printings, and leaving them separate spends five of the k slots on the same
answer. Rows are keyed by normalized name plus the card's full text, so a
reprint collapses and a card that shares a name with genuinely different text
does not. The printings ride along as metadata and are rendered compactly.

**A lexical signal beside the embedding.** Card text is exact vocabulary,
"Benched", "damage counters", "Prize cards", "Supporter", and a 384-dimensional
model distilled from web text is weakest exactly there: it knows the topic and
not the term. So the same passages are also scored with BM25 over a hand-rolled
tokenizer (no new dependency; the tokenizer folds accents, because the corpus
says "Pokemon" and a person types "Pokemon"), and the two rankings are fused
with reciprocal rank fusion at k=60. RRF rather than a weighted sum of scores
because a cosine and a BM25 score are not on the same scale and the weight
would need retuning every time either side changed.

Measured on the Standard corpus, the rank of Dragapult ex:

| query                                    | blob | passages | fused |
|------------------------------------------|------|----------|-------|
| bench damage                             |  306 |       12 |     4 |
| put damage counters on the bench         |  129 |        4 |     2 |
| "Put 6 damage counters ... you like."    |   39 |        1 |     1 |

Passage aggregation is what does the work; fusion is worth a few more places
and holds the two control queries ("search your deck for a Supporter card",
"draw cards until you have 7 in hand") at rank 1. `--no-lexical` turns fusion
off and searches the embeddings alone, which is how that table was measured.

**The embedder is an interface with two implementations.** `HashingEmbedder` is
deterministic, dependency-free and needs no download: it hashes word tokens
into a fixed number of buckets. It is not a semantic model and it is not
pretending to be one; it exists so the index format, the build, the search and
the tool all run in the fast test suite, and so `--embedder hashing` gives a
usable keyword-ish search on a machine that cannot download a model. The
marker on the real-model tests is `ml` for the same reason the trainer's is.

The index directory holds three files: `cards.parquet`, one row per distinct
card with its record and its printings; `vectors.parquet`, one row per passage
with its card, its text and its vector; and `meta.json`, which records the
layout, the counts and which embedder built it. A query embedded by a different
model than the index would return confident nonsense, so the loader checks the
name and refuses instead, and an index written by an older layout is refused
with the command to rebuild it rather than read as though it were this one.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
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

CARDS_FILE: Final = "cards.parquet"
VECTORS_FILE: Final = "vectors.parquet"
META_FILE: Final = "meta.json"
# Bumped when the files or their columns change. An index from before the
# passage layout has the right file names and the wrong contents, and reading
# it as though it were this one would answer with nonsense instead of failing.
INDEX_FORMAT_VERSION: Final = 2

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
# How much of a word the tokenizer keeps as a stem. Five is enough to make
# "bench" and "Benched" the same feature and short enough that it does not
# collapse "damage" onto "damaged" by accident, which it also does, on purpose.
STEM_CHARS: Final = 5

DEFAULT_K: Final = 5
MAX_K: Final = 20
# Effect text is the long field, and a tool result holding five whole cards is
# already a page. Cut each effect rather than the number of cards: the model
# asked for k cards and silently getting three would be the worse surprise.
MAX_EFFECT_CHARS: Final = 220
# How many reprints are spelled out before the rest become a count. Three keeps
# the heading to one line for every card in the current Standard format.
MAX_SHOWN_PRINTINGS: Final = 3

# Reciprocal rank fusion, with the constant from the paper that introduced it.
# 60 is large enough that the top of one list cannot dominate the other outright
# and small enough that the difference between rank 1 and rank 10 still matters.
RRF_K: Final = 60
# How deep each ranking is read before fusion. A passage below this contributes
# nothing, which is the point: the tail of a BM25 list is noise, and giving it a
# reciprocal rank would let it outvote a real match on the other side.
FUSION_DEPTH: Final = 200
# The usual BM25 constants: k1 is where term frequency saturates, b is how hard
# length normalization bites. Passages here are one or two sentences and vary by
# a factor of ten, so the default b of 0.75 is doing real work.
BM25_K1: Final = 1.2
BM25_B: Final = 0.75

CARD_TOOL: Final = "lookup_cards"
TOOL_SPAN_PREFIX: Final = "agent.tool."

PASSAGE_IDENTITY: Final = "identity"
PASSAGE_ABILITY: Final = "ability"
PASSAGE_ATTACK: Final = "attack"
PASSAGE_RULE: Final = "rule"

# A Trainer keeps its effect text in `rules`, so its rule passage reads better
# labelled with what the card is than with the word "rule".
TRAINER_STAGES: Final = frozenset({"item", "supporter", "stadium", "tool", "pokemon tool"})

_TOKEN: Final = re.compile(r"[a-z0-9']+")
_PARAGRAPH: Final = re.compile(r"\n\s*\n")


# -------------------------------------------------------------- tokenizer --


def fold(text: str) -> str:
    """Lower case with the accents taken off, so "Pokemon" matches "Pokemon".

    The corpus spells the name with an accent and nobody types it. Decomposing
    to NFKD and dropping the combining marks is the whole trick, and it costs
    nothing on the ASCII that the rest of the card text already is.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def tokenize(text: str) -> list[str]:
    """Word tokens: folded, split on anything that is not a letter, digit or apostrophe."""
    return _TOKEN.findall(fold(text))


def with_stems(tokens: Iterable[str]) -> list[str]:
    """Every token, plus a five-character stem for the ones long enough to have one.

    The crudest possible stemmer, there for exactly one thing: printed card text
    says "Benched" and a person searching says "bench", and matching whole words
    scores those two at zero. Keeping the whole token as well as the stem means
    an exact hit still counts twice as much as a stem-only hit, which is how
    "attack" stays distinguishable from "attach" even though both stem to
    "attac".
    """
    out: list[str] = []
    for token in tokens:
        out.append(token)
        if len(token) > STEM_CHARS:
            out.append(token[:STEM_CHARS])
    return out


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
        """Everything printed on the card as one string.

        This is no longer what gets embedded, because a whole card in one vector
        buries the one sentence a question is about; `passages()` is. It is what
        two printings are compared on, so that a reprint with the same text
        collapses into one indexed card and a card that shares a name with
        different text does not.
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

    def passages(self) -> list["Passage"]:
        """The card cut into the units a question actually asks about.

        One identity passage for the name, stage, types and hit points, then one
        per ability, one per attack and one per paragraph of rules text. Every
        passage says which card and which line it is, so a vector knows whose
        attack "Phantom Dive" is and a passage read back out of a hit is a
        sentence rather than a fragment.

        The naming goes at the end, in a parenthesis, and it is measured rather
        than chosen: against a verbatim quote of Phantom Dive, bge scores the
        bare effect sentence at 0.96, the same sentence tagged at the end at
        0.89, and the same sentence behind "Dragapult ex, attack Phantom Dive:
        200 damage." at 0.75, which is below a Stadium that merely talks about
        the Bench. A short sentence has few tokens to spare and a mean-pooled
        model spends them on whatever it is given first.
        """
        out: list[Passage] = []
        traits = ", ".join(
            filter(
                None,
                [self.stage, "/".join(self.types), f"{self.hp} HP" if self.hp else ""],
            )
        )
        out.append(
            Passage(
                kind=PASSAGE_IDENTITY,
                label="name and type line",
                text=f"{self.name}, {traits}." if traits else f"{self.name}.",
            )
        )
        for ability in self.abilities:
            label = f"ability {ability['name']}".strip()
            out.append(
                Passage(
                    kind=PASSAGE_ABILITY,
                    label=label,
                    text=_tagged(ability["effect"], f"{self.name}, {label}"),
                )
            )
        for attack in self.attacks:
            label = f"attack {attack['name']}".strip()
            damage = f", {attack['damage']} damage" if attack.get("damage") else ""
            out.append(
                Passage(
                    kind=PASSAGE_ATTACK,
                    label=label,
                    text=_tagged(str(attack.get("effect", "")), f"{self.name}, {label}{damage}"),
                )
            )
        rule_label = self.stage if fold(self.stage) in TRAINER_STAGES else "rule"
        for rule in self.rules:
            for paragraph in _PARAGRAPH.split(rule):
                if paragraph.strip():
                    out.append(
                        Passage(
                            kind=PASSAGE_RULE,
                            label=rule_label,
                            text=_tagged(paragraph, f"{self.name}, {rule_label}"),
                        )
                    )
        return out

    def printing(self) -> "Printing":
        """Where this copy of the card was printed."""
        return Printing(
            card_id=self.card_id,
            set_name=self.set_name,
            number=self.number,
            regulation_mark=self.regulation_mark,
            source_url=self.source_url,
        )


@dataclass(frozen=True)
class Printing:
    """One place a card was printed. A hit is a card; these are where to find it."""

    card_id: str
    set_name: str = ""
    number: str = ""
    regulation_mark: str = ""
    source_url: str = ""

    @property
    def where(self) -> str:
        """The set and number as a reader cites them, or the identifier if neither."""
        cited = " ".join(filter(None, [self.set_name.strip(), self.number.strip()]))
        return cited or self.card_id

    def as_record(self) -> dict[str, str]:
        """The printing as the JSON stored beside the card."""
        return {
            "card_id": self.card_id,
            "set": self.set_name,
            "number": self.number,
            "regulation_mark": self.regulation_mark,
            "source_url": self.source_url,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Printing":
        return cls(
            card_id=str(record.get("card_id", "")),
            set_name=str(record.get("set") or ""),
            number=str(record.get("number") or ""),
            regulation_mark=str(record.get("regulation_mark") or ""),
            source_url=str(record.get("source_url") or ""),
        )


@dataclass(frozen=True)
class Passage:
    """One indexed unit of a card: an identity line, an ability, an attack or a rule."""

    kind: str
    label: str
    text: str


@dataclass(frozen=True)
class IndexedCard:
    """One distinct card, with every printing of it that the corpus holds."""

    card: Card
    printings: list[Printing]

    @property
    def name(self) -> str:
        return self.card.name

    @property
    def card_id(self) -> str:
        """The first printing's identifier, which is the one the card renders under."""
        return self.card.card_id

    def passages(self) -> list[Passage]:
        return self.card.passages()

    def printings_line(self) -> str:
        """The printings on one line: the first spelled out, the rest abbreviated."""
        first, rest = self.printings[0], self.printings[1:]
        if not rest:
            return first.where
        shown = ", ".join(printing.where for printing in rest[:MAX_SHOWN_PRINTINGS])
        hidden = len(rest) - min(len(rest), MAX_SHOWN_PRINTINGS)
        tail = f" and {hidden} more" if hidden else ""
        return f"{first.where}, also {shown}{tail}"

    def render(self) -> str:
        """The card as the tool prints it: a heading and its text, a few lines."""
        card = self.card
        where = self.printings_line()
        lines = [f"**{card.name}**" + (f" ({where})" if where.strip() else "")]
        traits = ", ".join(
            filter(
                None,
                [
                    card.stage,
                    "/".join(card.types),
                    f"{card.hp} HP" if card.hp else "",
                    f"regulation {card.regulation_mark}" if card.regulation_mark else "",
                ],
            )
        )
        if traits:
            lines.append(traits)
        for ability in card.abilities:
            lines.append(f"Ability {ability['name']}: {_clip(ability['effect'])}")
        for attack in card.attacks:
            cost = "".join(item[:1].upper() for item in attack.get("cost") or [])
            damage = attack.get("damage") or "-"
            lines.append(
                f"Attack [{cost or '-'}] {attack['name']}, {damage}: "
                f"{_clip(str(attack.get('effect', '')))}".rstrip()
            )
        for rule in card.rules:
            lines.append(_clip(rule))
        return "\n".join(lines)


def _tagged(body: str, tag: str) -> str:
    """A passage as "<text> (<card>, <what it is>)", or just the tag when there is no text."""
    flat = " ".join(body.split())
    return f"{flat} ({tag})" if flat else f"{tag}."


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
    """Every printing in a JSONL corpus. A malformed line is skipped and counted."""
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


def collapse_printings(cards: Sequence[Card]) -> list[IndexedCard]:
    """Printings grouped into cards, in the order the corpus first mentions each.

    Two rows are the same card when the folded name and the whole printed text
    agree. Name alone would merge the two Silvally that share a name and not an
    attack; text alone would merge nothing, because the text starts with the
    name anyway. The first printing seen is the one that renders, and the rest
    follow it as metadata.
    """
    groups: dict[tuple[str, str], IndexedCard] = {}
    for card in cards:
        key = (fold(card.name).strip(), card.document())
        entry = groups.get(key)
        if entry is None:
            groups[key] = IndexedCard(card=card, printings=[card.printing()])
        else:
            entry.printings.append(card.printing())
    return list(groups.values())


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

    A zero row stays zero rather than becoming a division by zero: a passage
    with no text is a passage that matches nothing, which is the honest answer.
    """
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normalized: np.ndarray = (matrix / lengths).astype(np.float32)
    return normalized


class HashingEmbedder:
    """Word tokens hashed into fixed buckets. Deterministic, and no download.

    Not a semantic model: "bench damage" finds a passage that says bench and
    damage, not one that says "your opponent's other Pokemon". That is enough
    for the fast test suite, which is about the index format, the search and
    the tool rather than about retrieval quality, and it is a usable keyword
    search on a machine that cannot fetch a model.

    Sublinear term frequency, `1 + log(count)`, for the same reason every bag
    of words uses it: a card whose effect says "damage" four times is not four
    times more about damage than one that says it once.

    It shares the module's tokenizer, so it gets the accent folding and the
    five-character stem that BM25 gets; the stem is what makes the printed
    "Benched" and the typed "bench" the same feature. The real embedder has no
    such problem, which is why this one is only ever the fallback.
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
            for feature in with_stems(tokenize(text)):
                bucket = self._bucket(feature)
                counts[bucket] = counts.get(bucket, 0) + 1
            for bucket, count in counts.items():
                matrix[row, bucket] = 1.0 + math.log(count)
        return _normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        """The same bag of hashed tokens: this embedder has no query side."""
        row: np.ndarray = self.embed([text])[0]
        return row

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


# ------------------------------------------------------------------ bm25 --


class BM25:
    """Okapi BM25 over the passages, built in memory from their text.

    Hand-rolled rather than a dependency: the whole model is a postings table,
    an inverse document frequency and one saturating term, and a few thousand
    short passages fit in a dictionary that takes milliseconds to build. It is
    built when the first lexical query arrives rather than at load time, so the
    pure-embedding path pays nothing for it.

    The scored terms are the tokenizer's, stems included, so a query for
    "bench" reaches a passage that says "Benched" here as well as in the
    hashing embedder.
    """

    def __init__(self, passages: Sequence[str], *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self._k1 = k1
        self._b = b
        self._count = len(passages)
        lengths = np.zeros(self._count, dtype=np.float32)
        postings: dict[str, dict[int, int]] = {}
        for row, text in enumerate(passages):
            terms = with_stems(tokenize(text))
            lengths[row] = len(terms)
            for term in terms:
                bucket = postings.setdefault(term, {})
                bucket[row] = bucket.get(row, 0) + 1
        self._average = float(lengths.mean()) if self._count and lengths.sum() else 1.0
        self._lengths = lengths
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        for term, bucket in postings.items():
            rows = np.fromiter(bucket.keys(), dtype=np.int32, count=len(bucket))
            counts = np.fromiter(bucket.values(), dtype=np.float32, count=len(bucket))
            self._postings[term] = (rows, counts)
            frequency = len(bucket)
            self._idf[term] = math.log(1.0 + (self._count - frequency + 0.5) / (frequency + 0.5))

    @property
    def passages(self) -> int:
        return self._count

    def scores(self, query: str) -> np.ndarray:
        """One BM25 score per passage. Zero where no query term appears."""
        scores = np.zeros(self._count, dtype=np.float32)
        if not self._count:
            return scores
        penalty = self._k1 * (1.0 - self._b + self._b * self._lengths / self._average)
        for term in with_stems(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            rows, counts = posting
            scores[rows] += self._idf[term] * (counts * (self._k1 + 1.0) / (counts + penalty[rows]))
        return scores


def reciprocal_rank_fusion(
    rankings: Sequence[np.ndarray], size: int, *, depth: int = FUSION_DEPTH, k: int = RRF_K
) -> np.ndarray:
    """Fuse rankings of the same rows into one score per row.

    Each ranking is row indices, best first. A row contributes `1 / (k + rank)`
    from each list it reaches the top `depth` of, and nothing from the lists it
    does not: the tail of a ranking is noise and a reciprocal rank is not small
    enough to make noise harmless. Ranks are one-based, so the top of a list is
    worth `1 / (k + 1)` and no row is ever divided by zero.
    """
    fused = np.zeros(size, dtype=np.float32)
    for ranking in rankings:
        head = ranking[:depth]
        fused[head] += 1.0 / (k + 1.0 + np.arange(len(head), dtype=np.float32))
    return fused


# ----------------------------------------------------------------- index --


_CARD_SCHEMA: Final = pa.schema(
    [
        pa.field("card_id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("record_json", pa.string(), nullable=False),
        pa.field("printings_json", pa.string(), nullable=False),
    ]
)

_PASSAGE_FIELDS: list[pa.Field[Any]] = [
    pa.field("card_row", pa.int32(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),
    pa.field("label", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("vector", pa.list_(pa.float32()), nullable=False),
]
_PASSAGE_SCHEMA: Final = pa.schema(_PASSAGE_FIELDS)


def build_index(cards: Sequence[Card], embedder: Embedder, out_dir: Path) -> int:
    """Embed every passage and write the index. Returns the number of cards written.

    Printings collapse into cards first, so the count returned is distinct cards
    and not rows read. The whole record travels into the Parquet as JSON beside
    its printings, so a search result is a card rather than an identifier that
    has to be looked up again in the source file the index may well outlive.
    """
    if not cards:
        raise ValueError("there are no cards to index")
    entries = collapse_printings(cards)
    rows: list[int] = []
    passages: list[Passage] = []
    for row, entry in enumerate(entries):
        for passage in entry.passages():
            rows.append(row)
            passages.append(passage)
    vectors = embedder.embed([passage.text for passage in passages])
    if vectors.shape[0] != len(passages):
        raise ValueError(
            f"the embedder returned {vectors.shape[0]} rows for {len(passages)} passages"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(
            {
                "card_id": [entry.card_id for entry in entries],
                "name": [entry.name for entry in entries],
                "record_json": [
                    json.dumps(_as_record(entry.card), sort_keys=True) for entry in entries
                ],
                "printings_json": [
                    json.dumps([printing.as_record() for printing in entry.printings])
                    for entry in entries
                ],
            },
            schema=_CARD_SCHEMA,
        ),
        out_dir / CARDS_FILE,
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "card_row": rows,
                "kind": [passage.kind for passage in passages],
                "label": [passage.label for passage in passages],
                "text": [passage.text for passage in passages],
                "vector": [row.tolist() for row in vectors],
            },
            schema=_PASSAGE_SCHEMA,
        ),
        out_dir / VECTORS_FILE,
    )
    (out_dir / META_FILE).write_text(
        json.dumps(
            {
                "format_version": INDEX_FORMAT_VERSION,
                "embedder": embedder.name,
                "dimensions": int(vectors.shape[1]),
                "cards": len(entries),
                "printings": len(cards),
                "passages": len(passages),
                "files": {"cards": CARDS_FILE, "passages": VECTORS_FILE},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(entries)


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
    """One search result: the card, the passage that matched, and how close it was."""

    card: IndexedCard
    score: float
    passage: Passage
    fused: bool = False

    @property
    def score_label(self) -> str:
        """What the number means, which is not the same on the two paths."""
        return "fused score" if self.fused else "similarity"


class CardIndex:
    """A built index in memory: the cards, their passages and vectors, the embedder."""

    def __init__(
        self,
        cards: Sequence[IndexedCard],
        passages: Sequence[Passage],
        passage_card: np.ndarray,
        vectors: np.ndarray,
        embedder: Embedder,
        *,
        lexical: bool = True,
    ) -> None:
        self.cards = list(cards)
        self.passages = list(passages)
        self.passage_card = passage_card
        self.vectors = vectors
        self.embedder = embedder
        self.lexical = lexical
        self._bm25: BM25 | None = None

    @classmethod
    def load(cls, directory: Path, embedder: Embedder | None = None, **options: Any) -> "CardIndex":
        """Read an index off disk, with the embedder its `meta.json` names.

        An embedder passed in wins, which is how a test builds with the hashing
        one and searches with it too; otherwise the name in the metadata
        decides, because a query vector from a different model is not in the
        same space as the index and the results would look plausible and be
        meaningless. An index from an older layout is refused for the same
        reason, and with the same bluntness.
        """
        meta_path = directory / META_FILE
        cards_path = directory / CARDS_FILE
        vectors_path = directory / VECTORS_FILE
        if not meta_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError(
                f"no card index at {directory}: run `python -m pipeline.card_index build`"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = _as_int(meta.get("format_version")) or 1
        if version != INDEX_FORMAT_VERSION or not cards_path.is_file():
            raise ValueError(
                f"the card index at {directory} is format {version}, this reads "
                f"{INDEX_FORMAT_VERSION}: rebuild it with `python -m pipeline.card_index build`"
            )
        card_table = pq.read_table(cards_path)
        entries = [
            IndexedCard(
                card=Card.from_record(json.loads(str(record))),
                printings=[Printing.from_record(item) for item in json.loads(str(printings))],
            )
            for record, printings in zip(
                card_table.column("record_json").to_pylist(),
                card_table.column("printings_json").to_pylist(),
                strict=True,
            )
        ]
        passage_table = pq.read_table(vectors_path)
        passages = [
            Passage(kind=str(kind), label=str(label), text=str(text))
            for kind, label, text in zip(
                passage_table.column("kind").to_pylist(),
                passage_table.column("label").to_pylist(),
                passage_table.column("text").to_pylist(),
                strict=True,
            )
        ]
        passage_card = np.asarray(passage_table.column("card_row").to_pylist(), dtype=np.int32)
        vectors = np.asarray(passage_table.column("vector").to_pylist(), dtype=np.float32)
        resolved = embedder if embedder is not None else make_embedder(str(meta["embedder"]))
        if embedder is not None and embedder.name != meta["embedder"]:
            logger.warning(
                "searching an index with a different embedder than built it",
                extra={"index_embedder": meta["embedder"], "query_embedder": embedder.name},
            )
        return cls(entries, passages, passage_card, vectors, resolved, **options)

    def bm25(self) -> BM25:
        """The lexical model over the same passages, built once, on first use."""
        if self._bm25 is None:
            self._bm25 = BM25([passage.text for passage in self.passages])
        return self._bm25

    def search(self, query: str, k: int = DEFAULT_K, *, lexical: bool | None = None) -> list[Hit]:
        """The k best cards, best first, each with the passage that matched.

        Passages are scored, not cards. With `lexical` off that is one
        matrix-vector product over the whole corpus, exact and well under a
        millisecond; with it on, a BM25 pass runs beside it and the two
        rankings are fused by reciprocal rank. Either way the passage scores
        collapse to cards by taking each card's best passage, which is what
        keeps a five-attack Pokemon from filling the result with itself.
        """
        wanted = max(1, min(k, MAX_K))
        if not self.cards or not self.passages:
            return []
        use_lexical = self.lexical if lexical is None else lexical
        dense = self.vectors @ self.embedder.embed_query(query)
        if use_lexical:
            lexical_scores = self.bm25().scores(query)
            lexical_order = np.argsort(-lexical_scores, kind="stable")
            # A passage no query term reaches scores zero, and there are
            # thousands of those. Ranking them would hand the top of the BM25
            # list to whichever card happens to be stored first.
            lexical_order = lexical_order[lexical_scores[lexical_order] > 0]
            fused = reciprocal_rank_fusion(
                [np.argsort(-dense, kind="stable"), lexical_order],
                size=len(self.passages),
            )
            # Cosine breaks the ties, and there are many: every passage outside
            # both candidate lists is fused at zero, and fusion itself puts
            # whole groups of passages on the same rung.
            order = np.lexsort((-dense, -fused))
            scores = fused
        else:
            order = np.argsort(-dense, kind="stable")
            scores = dense
        hits: list[Hit] = []
        seen: set[int] = set()
        for position in order:
            row = int(self.passage_card[position])
            if row in seen:
                continue
            seen.add(row)
            hits.append(
                Hit(
                    card=self.cards[row],
                    score=float(scores[position]),
                    passage=self.passages[int(position)],
                    fused=use_lexical,
                )
            )
            if len(hits) == wanted:
                break
        return hits


def render_hits(hits: Sequence[Hit]) -> str:
    """Search results as the text the tool returns."""
    if not hits:
        return "no cards matched."
    blocks = [
        f"{hit.card.render()}\nmatched {hit.passage.label} ({hit.score_label} {hit.score:.3f})"
        for hit in hits
    ]
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ tool --


def make_lookup_cards_tool(
    index_dir: Path,
    *,
    tracer: trace.Tracer,
    metrics: ServiceMetrics,
    embedder: Embedder | None = None,
    lexical: bool = True,
) -> Any:
    """The card-text tool, bound to one built index.

    The index is loaded here, when the tool is made, rather than on the first
    call: an agent that is going to fail because its index is missing should
    fail while it is being built, not in the middle of answering.
    """
    from langchain_core.tools import StructuredTool

    from pipeline.agent import ToolCall, record_call, summarize

    index = CardIndex.load(index_dir, embedder, lexical=lexical)

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
            "rules. Returns the top k cards with their text and the line that matched. "
            "This is a card reference, not game data: it says nothing about how often a "
            "card is played."
        ),
    )


# ------------------------------------------------------------ entry point --


def build(source: Path, out_dir: Path, embedder: Embedder) -> int:
    """Read the corpus, build the index, and say what was written."""
    printings = read_cards(source)
    written = build_index(printings, embedder, out_dir)
    meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    logger.info(
        "card index built",
        extra={
            "cards": written,
            "printings": len(printings),
            "passages": meta["passages"],
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
    query.add_argument(
        "--no-lexical",
        dest="lexical",
        action="store_false",
        help="search the embeddings alone, without the BM25 half of the ranking",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    configure_logging(STAGE)

    if args.command == "query":
        index = CardIndex.load(args.index, lexical=args.lexical)
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
        meta = json.loads((args.out / META_FILE).read_text(encoding="utf-8"))
        metrics.rows_in = int(meta["printings"])
        metrics.rows_out = written
        metrics.rows_quarantined = 0
        metrics.extra = {
            "embedder": args.embedder,
            "out": str(args.out),
            "passages": int(meta["passages"]),
        }

    emit_summary(
        logger,
        "card index built",
        {"cards": written, "embedder": args.embedder, "out": str(args.out)},
        text=f"indexed {written} cards into {args.out} with {args.embedder}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
