"""Enrich the local card catalog with printed card text, from the TCGdex API.

    uv run python scripts/fetch_card_text.py
    uv run python scripts/fetch_card_text.py --catalog tests/catalog.json --limit 50

The catalog the silver stage joins against (`scripts/fetch_catalog.py`) carries
a name, a set, a number, a type, hit points and a regulation mark, which is
everything a join needs and nothing a question needs: it cannot say what a card
does. The retriever needs the text, so this fills the gap from a public source
and writes `data/catalog/card_text.jsonl`, one JSON object per card, which
`python -m pipeline.card_index build` turns into the agent's second tool.

**Source and terms.** The text comes from TCGdex (https://tcgdex.net), a free,
open card database with a public REST API at `https://api.tcgdex.net/v2/en`.
Its data is community maintained and offered for open use; the card names,
attack names and rules text it serves are Pokemon Trading Card Game content
owned by Nintendo, Creatures Inc. and GAME FREAK, and this project is not
affiliated with any of them. What this script writes is a local working copy
for a local index: the dump is gitignored and is never committed, republished
or served as data in its own right, and the tool that reads it quotes a card's
text alongside an attribution back to `source_url`. Requests are made a few at
a time with a `User-Agent` naming the project, which is what the API asks for.

**Matching.** The local catalog is the list of cards this corpus has actually
seen, so it decides what is fetched: the sets it names are resolved against
TCGdex's set list by identifier and then by name, each matched set is listed
once, and a catalog entry finds its card by (set, collector number) first, by
name among the listed sets second, and by an exact-name query against the whole
database third, which is what covers a set code the two sources spell
differently. A card that matches nothing is counted and skipped, never guessed
at, because a wrong card's text is worse in a retriever than a missing one: a
missing card returns nothing, a wrong one returns a confident answer about the
wrong card.

**Being a good client.** Four requests in flight, three retries with a jittered
backoff on a timeout, a 429 or a 5xx, one listing request per set rather than
one per card, and a `--limit` for trying the thing out without walking the
whole catalog.

**Only the format that matters.** The catalog lists every printing the client
knows, some 25,000 back to 2011, and the games this pipeline sees are Standard
format, where a card is legal by its regulation mark. So the fetch keeps only
the printings whose mark is in `STANDARD_REGULATION_MARKS` (a few thousand)
unless `--reg` says otherwise; `--reg all` walks everything. A retriever full
of cards nobody can play would rank them beside the ones people do.
"""

import argparse
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from pipeline.config import (
    CARD_TEXT_PATH,
    CATALOG_PATH,
    REPO_ROOT,
    STANDARD_REGULATION_MARKS,
)
from pipeline.observability import configure_logging, emit_summary, stage_run

logger = logging.getLogger(__name__)

STAGE: Final = "fetch_card_text"
API_ROOT: Final = "https://api.tcgdex.net/v2/en"
USER_AGENT: Final = "pokemon-meta-pipeline/0.1 (card text for a local retriever index)"
# The fixture catalog, used when the real one has not been fetched. It is ten
# entries, so a run against it is a handful of requests and a real check that
# the matching works end to end.
FALLBACK_CATALOG: Final = REPO_ROOT / "tests" / "catalog.json"

WORKERS: Final = 4
TIMEOUT_S: Final = 20.0
RETRIES: Final = 3
BACKOFF_S: Final = 1.5
RETRY_STATUS: Final = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """A request that failed every attempt. The message names the URL and the reason."""


# ------------------------------------------------------------------ HTTP --


def get_json(path: str, *, timeout: float = TIMEOUT_S, retries: int = RETRIES) -> Any:
    """One GET against the API, retried with a backoff on the transient failures.

    A 404 is not transient and is not retried: it means the identifier is
    wrong, and asking three more times is rude rather than hopeful.
    """
    url = f"{API_ROOT}/{path.lstrip('/')}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as failure:
            if failure.code not in RETRY_STATUS:
                raise FetchError(f"{url}: HTTP {failure.code}") from failure
            last = f"HTTP {failure.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as failure:
            last = f"{type(failure).__name__}: {failure}"
        if attempt < retries:
            # Jittered, so four workers that all hit a rate limit at the same
            # moment do not all come back at the same moment either.
            time.sleep(BACKOFF_S * attempt + random.random() * 0.5)
    raise FetchError(f"{url}: {last}")


# -------------------------------------------------------------- matching --


@dataclass(frozen=True)
class CatalogEntry:
    """One row of the local catalog: what this corpus has seen, and where to find it."""

    key: str
    name: str
    set_code: str
    number: str
    reg: str = ""


def read_catalog(path: Path) -> list[CatalogEntry]:
    """The local catalog as entries, whichever of its two shapes it is in.

    The real catalog is keyed by the client's card identifier and the committed
    fixture is keyed by a lowercased name; both map to an object carrying the
    name, the set and the number, which is all this needs.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[CatalogEntry] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        entries.append(
            CatalogEntry(
                key=str(key),
                name=str(value.get("name") or "").strip(),
                set_code=str(value.get("set") or "").strip(),
                number=str(value.get("number") or "").strip(),
                reg=str(value.get("reg") or "").strip().upper(),
            )
        )
    return entries


def in_format(entries: Iterable[CatalogEntry], marks: Iterable[str]) -> list[CatalogEntry]:
    """The entries whose regulation mark is one of `marks`; every entry when `marks` is empty.

    An entry with no mark at all is kept only when nothing is being filtered:
    the committed fixture catalog carries none, and a real printing without one
    predates regulation marks entirely, which puts it outside any format that
    uses them.
    """
    wanted = {mark.strip().upper() for mark in marks if mark.strip()}
    if not wanted:
        return list(entries)
    return [entry for entry in entries if entry.reg in wanted]


def normalize_number(number: str) -> str:
    """A collector number as TCGdex writes it, so `1`, `001` and `1/191` all match.

    TCGdex's `localId` is unpadded for most sets and alphanumeric for promos, so
    the comparison is on the leading digits with the padding removed, and on the
    raw string when there are no digits at all.
    """
    head = number.split("/")[0].strip()
    digits = "".join(char for char in head if char.isdigit())
    return digits.lstrip("0") or head.lower()


def normalize_name(name: str) -> str:
    """A card name reduced to what two sources can be expected to agree on."""
    return " ".join(name.lower().replace("’", "'").split())


def resolve_sets(codes: Iterable[str]) -> dict[str, str]:
    """Local set codes mapped to TCGdex set identifiers, by id then by name.

    One request for the whole set list rather than one per code: there are a few
    hundred sets, the response is small, and a lookup table is a better neighbour
    than a few dozen probes.
    """
    listing = get_json("sets")
    by_id = {str(item["id"]).lower(): str(item["id"]) for item in listing}
    by_name = {normalize_name(str(item.get("name", ""))): str(item["id"]) for item in listing}
    resolved: dict[str, str] = {}
    for code in codes:
        if not code:
            continue
        lowered = code.lower()
        found = by_id.get(lowered) or by_name.get(normalize_name(code))
        if found:
            resolved[code] = found
        else:
            logger.info("no TCGdex set matches this code", extra={"set_code": code})
    return resolved


@dataclass
class SetIndex:
    """Every card of the fetched sets, indexed the two ways a catalog entry looks up."""

    by_set_number: dict[tuple[str, str], str]
    by_name: dict[str, str]

    @classmethod
    def build(cls, set_ids: Iterable[str]) -> "SetIndex":
        by_set_number: dict[tuple[str, str], str] = {}
        by_name: dict[str, str] = {}
        for set_id in set_ids:
            try:
                detail = get_json(f"sets/{urllib.parse.quote(set_id)}")
            except FetchError as failure:
                logger.warning("a set could not be listed", extra={"error": str(failure)})
                continue
            for brief in detail.get("cards", []):
                card_id = str(brief.get("id", ""))
                if not card_id:
                    continue
                local = normalize_number(str(brief.get("localId", "")))
                by_set_number[(set_id.lower(), local)] = card_id
                # First printing wins, so a reprint does not displace the card
                # a catalog entry that matched by name was most likely about.
                by_name.setdefault(normalize_name(str(brief.get("name", ""))), card_id)
        return cls(by_set_number=by_set_number, by_name=by_name)

    def find(self, entry: CatalogEntry, sets: dict[str, str]) -> str | None:
        """The TCGdex card id for one catalog entry, by set and number then by name.

        Three attempts in order of confidence: the set and the collector number,
        which identifies a printing; the name among the sets that were listed;
        and the name against the whole database, which is one request and is
        cached, because a set code the local catalog spells differently is the
        common reason the first two miss.
        """
        set_id = sets.get(entry.set_code)
        if set_id and entry.number:
            found = self.by_set_number.get((set_id.lower(), normalize_number(entry.number)))
            if found:
                return found
        local = self.by_name.get(normalize_name(entry.name))
        return local if local is not None else find_by_name(entry.name)


@lru_cache(maxsize=4096)
def find_by_name(name: str) -> str | None:
    """The oldest printing TCGdex has of a card with this exact name, or None.

    The oldest rather than the newest, deliberately: the text of a reprint is
    the same text, and the first printing is the stable identifier of the two.
    Cached because a catalog of 25,000 entries holds many rows that are the
    same card, and because a miss is worth asking about exactly once.
    """
    cleaned = name.strip()
    if not cleaned:
        return None
    try:
        matches = get_json(f"cards?name={urllib.parse.quote(f'eq:{cleaned}')}")
    except FetchError as failure:
        logger.warning("a name lookup failed", extra={"error": str(failure)})
        return None
    ids = sorted(str(item["id"]) for item in matches if item.get("id"))
    return ids[0] if ids else None


# --------------------------------------------------------------- records --


def to_record(card: dict[str, Any]) -> dict[str, Any]:
    """One TCGdex card as the corpus record `pipeline.card_index` reads.

    `stage` carries whichever of the three a card has: a Pokemon's evolution
    stage, a Trainer's kind (Item, Supporter, Stadium, Tool) or `Energy`. One
    field rather than three, because the index treats it as a label and the
    reader wants to see "Supporter" in the same place it sees "Stage 2".
    """
    card_id = str(card.get("id", ""))
    rules = [str(value) for value in card.get("rules") or []]
    for extra in (card.get("effect"), card.get("description")):
        if extra:
            rules.append(str(extra))
    category = str(card.get("category", ""))
    stage = str(
        card.get("stage") or card.get("trainerType") or (category if category == "Energy" else "")
    )
    return {
        "card_id": card_id,
        "name": str(card.get("name", "")),
        "set": str((card.get("set") or {}).get("name", "")),
        "number": str(card.get("localId", "")),
        "types": [str(value) for value in card.get("types") or []],
        "hp": card.get("hp"),
        "stage": stage,
        "abilities": [
            {"name": str(item.get("name", "")), "effect": str(item.get("effect", ""))}
            for item in card.get("abilities") or []
        ],
        "attacks": [
            {
                "name": str(item.get("name", "")),
                "cost": [str(value) for value in item.get("cost") or []],
                "damage": str(item.get("damage") or ""),
                "effect": str(item.get("effect", "")),
            }
            for item in card.get("attacks") or []
        ],
        "rules": rules,
        "retreat": card.get("retreat"),
        "regulation_mark": str(card.get("regulationMark") or ""),
        "source_url": f"{API_ROOT}/cards/{card_id}",
    }


def fetch_cards(card_ids: list[str]) -> list[dict[str, Any]]:
    """Every card, a few requests at a time, skipping the ones that fail.

    A failure is logged and dropped rather than raised: a run that fetched
    24,900 of 25,000 cards has built a usable index, and stopping on the
    hundredth would throw away the other 24,899.
    """
    records: list[dict[str, Any]] = []

    def one(card_id: str) -> dict[str, Any] | None:
        try:
            return to_record(get_json(f"cards/{urllib.parse.quote(card_id)}"))
        except FetchError as failure:
            logger.warning("a card could not be fetched", extra={"error": str(failure)})
            return None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for record in pool.map(one, card_ids):
            if record is not None:
                records.append(record)
    return records


def write_records(records: list[dict[str, Any]], out: Path) -> int:
    """The corpus as JSON lines, sorted by card id so two runs produce one diff."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: str(item["card_id"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


# ------------------------------------------------------------ entry point --


def main(argv: list[str] | None = None) -> int:
    """Resolve the catalog's sets, fetch their cards, and write the corpus."""
    parser = argparse.ArgumentParser(
        prog="uv run python scripts/fetch_card_text.py",
        description="Download printed card text from TCGdex for the cards the catalog names.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"the local catalog (default: {CATALOG_PATH}, else {FALLBACK_CATALOG})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CARD_TEXT_PATH,
        metavar="PATH",
        help=f"where to write the corpus (default: {CARD_TEXT_PATH})",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N", help="stop after N cards; 0 means every one"
    )
    parser.add_argument(
        "--reg",
        default=",".join(STANDARD_REGULATION_MARKS),
        metavar="MARKS",
        help=(
            "comma-separated regulation marks to keep, or `all` "
            f"(default: the Standard format, {','.join(STANDARD_REGULATION_MARKS)})"
        ),
    )
    args = parser.parse_args(argv)
    marks: list[str] = [] if args.reg.strip().lower() == "all" else args.reg.split(",")
    configure_logging(STAGE)

    catalog_path: Path = args.catalog or (
        CATALOG_PATH if CATALOG_PATH.is_file() else FALLBACK_CATALOG
    )
    if not catalog_path.is_file():
        logger.error("no catalog to enrich", extra={"path": str(catalog_path)})
        return 1

    with stage_run(STAGE) as metrics:
        catalog_entries = read_catalog(catalog_path)
        entries = in_format(catalog_entries, marks)
        logger.info(
            "catalog filtered to the format",
            extra={"marks": marks or "all", "kept": len(entries), "of": len(catalog_entries)},
        )
        sets = resolve_sets({entry.set_code for entry in entries})
        index = SetIndex.build(sorted(set(sets.values())))
        matched: list[str] = []
        unmatched = 0
        for entry in entries:
            found = index.find(entry, sets)
            if found is None:
                unmatched += 1
                continue
            if found not in matched:
                matched.append(found)
        if args.limit > 0:
            matched = matched[: args.limit]
        records = fetch_cards(matched)
        written = write_records(records, args.out)
        metrics.rows_in = len(entries)
        metrics.rows_out = written
        metrics.rows_quarantined = unmatched
        metrics.extra = {
            "catalog": str(catalog_path),
            "regulation_marks": marks or "all",
            "sets_resolved": len(sets),
            "unmatched": unmatched,
            "out": str(args.out),
        }

    emit_summary(
        logger,
        "card text fetched",
        {
            "catalog": str(catalog_path),
            "entries": len(entries),
            "matched": len(matched),
            "unmatched": unmatched,
            "written": written,
            "out": str(args.out),
        },
        text=(
            f"{len(entries)} catalog entries, {len(matched)} matched, {unmatched} unmatched; "
            f"wrote {written} cards to {args.out}"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
