"""The card-text fetch: the set-code spellings, and the count of what it skipped.

Nothing here touches the network. `get_json` is replaced by a stub serving a
trimmed set listing, which is what the resolver is supposed to check its guesses
against, and the name query the matcher falls back to is replaced as well, so a
test that misses stays a test rather than becoming a request.

The two things under test are the two the corpus depended on and did not have.
The client writes a set as `SV6`, `SV8-5` or `MEBSP` and TCGdex writes it as
`sv06`, `sv08.5` or `mep`, so every code of the current Standard format missed,
every entry fell through to a name query against the whole database, and the
index filled up with the oldest printing of each name from some other era. And
an entry whose set never resolved was dropped without being counted, so the
summary said nothing was unmatched while four fifths of the catalog went
missing.
"""

import logging
from typing import Any, Final

import pytest

from scripts.fetch_card_text import (
    NO_SET_CODE,
    CatalogEntry,
    Matches,
    SetIndex,
    match_entries,
    normalize_set_code,
    resolve_sets,
    set_code_candidates,
)

# The listing as the API serves it, cut to the sets the Standard format names
# plus two older ones, which is enough for a resolver whose whole job is to ask
# whether a spelling is a set that exists.
SET_LISTING: Final[list[dict[str, Any]]] = [
    {"id": "sv03.5", "name": "151"},
    {"id": "sv06", "name": "Twilight Masquerade"},
    {"id": "sv07", "name": "Stellar Crown"},
    {"id": "sv08", "name": "Surging Sparks"},
    {"id": "sv08.5", "name": "Prismatic Evolutions"},
    {"id": "sv09", "name": "Journey Together"},
    {"id": "sv10", "name": "Destined Rivals"},
    {"id": "sv10.5b", "name": "Black Bolt"},
    {"id": "sv10.5w", "name": "White Flare"},
    {"id": "me01", "name": "Mega Evolution"},
    {"id": "me02", "name": "Phantasmal Flames"},
    {"id": "me02.5", "name": "Ascended Heroes"},
    {"id": "me03", "name": "Perfect Order"},
    {"id": "me04", "name": "Chaos Rising"},
    {"id": "me05", "name": "Pitch Black"},
    {"id": "mep", "name": "MEP Black Star Promos"},
]

# Every set code the real catalog names for the Standard format, and the TCGdex
# identifier each one is. Two of them are the halves of one special set: the
# client splits Reshiram from Zekrom, TCGdex serves White Flare and Black Bolt.
STANDARD_CODES: Final[dict[str, str]] = {
    "SV6": "sv06",
    "SV7": "sv07",
    "SV8": "sv08",
    "SV8-5": "sv08.5",
    "SV9": "sv09",
    "ME1": "me01",
    "ME2-5": "me02.5",
    "ME3": "me03",
    "ME4": "me04",
    "ME5": "me05",
    "MEBSP": "mep",
    "RSV10-5": "sv10.5w",
    "ZSV10-5": "sv10.5b",
}


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """`get_json` serving the trimmed set listing and refusing anything else."""

    def stub(path: str, **_: Any) -> Any:
        assert path == "sets", f"the resolver asked for {path}, which is not the set list"
        return SET_LISTING

    monkeypatch.setattr("scripts.fetch_card_text.get_json", stub)
    return SET_LISTING


@pytest.fixture
def no_name_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-database name query, answering nothing, without a request."""
    monkeypatch.setattr("scripts.fetch_card_text.find_by_name", lambda name: None)


# ----------------------------------------------------------- set codes --


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("SV6", "sv06"),
        ("SV7", "sv07"),
        ("SV9", "sv09"),
        ("SV8-5", "sv08.5"),
        ("ME1", "me01"),
        ("ME2-5", "me02.5"),
        ("ME10", "me10"),
        ("sv06", "sv06"),
        (" SV6 ", "sv06"),
        ("RSV10-5", "rsv10.5"),
        ("MEBSP", None),
        ("", None),
    ],
)
def test_normalize_set_code_spells_the_identifier(code: str, expected: str | None) -> None:
    """Pad the series to two digits, write the `-5` tail as a decimal, or give up."""
    assert normalize_set_code(code) == expected


def test_set_code_candidates_put_the_table_first() -> None:
    """The explicit table outranks the rule, and a code is never tried twice."""
    assert set_code_candidates("MEBSP")[0] == "mep"
    assert set_code_candidates("ZSV10-5") == ["sv10.5b", "zsv10-5", "zsv10.5"]
    assert set_code_candidates("sv06") == ["sv06"]


def test_resolve_sets_maps_every_code_of_the_format(listing: list[dict[str, Any]]) -> None:
    """The thirteen codes the Standard catalog names all reach their TCGdex set."""
    assert resolve_sets(STANDARD_CODES) == STANDARD_CODES


def test_resolve_sets_matches_by_name_when_no_spelling_does(
    listing: list[dict[str, Any]],
) -> None:
    """A code that is really the set's name still resolves, as it did before."""
    assert resolve_sets(["Twilight Masquerade"]) == {"Twilight Masquerade": "sv06"}


def test_resolve_sets_keeps_only_identifiers_the_listing_serves(
    listing: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    """A guess is a guess until the set list confirms it, and a miss is a warning.

    `SV11` normalizes to a perfectly plausible `sv11` and there is no such set,
    so the code resolves to nothing rather than to an identifier that would 404
    one card at a time. The empty code is not a code and is passed over in
    silence; the catalog carries it for printings whose set the client did not
    record, and `match_entries` is what counts those.
    """
    with caplog.at_level(logging.WARNING, logger="scripts.fetch_card_text"):
        assert resolve_sets(["SV11", "SV6", ""]) == {"SV6": "sv06"}
    assert [record.set_code for record in caplog.records] == ["SV11"]  # type: ignore[attr-defined]
    assert all(record.levelno == logging.WARNING for record in caplog.records)


# -------------------------------------------------------- what was skipped --


def entry(name: str, set_code: str, number: str) -> CatalogEntry:
    """One catalog row, keyed the way the real catalog keys them."""
    return CatalogEntry(
        key=f"{set_code.lower()}_{number}", name=name, set_code=set_code, number=number, reg="H"
    )


def test_match_entries_finds_the_printing_and_counts_each_card_once(
    no_name_query: None,
) -> None:
    """A card matched by two entries is fetched once, in the order it was seen."""
    index = SetIndex(
        by_set_number={("sv06", "130"): "sv06-130", ("sv06", "131"): "sv06-131"},
        by_name={"dragapult ex": "sv06-130"},
    )
    entries = [
        entry("Dragapult ex", "SV6", "130"),
        # The reverse holo of the same printing: one more row, the same card.
        entry("Dragapult ex", "SV6", "130"),
        entry("Dusknoir", "SV6", "131"),
    ]
    found = match_entries(entries, {"SV6": "sv06"}, index)
    assert found == Matches(card_ids=["sv06-130", "sv06-131"])


def test_match_entries_counts_the_entries_whose_set_never_resolved(
    no_name_query: None,
) -> None:
    """An unresolved set is unmatched entries, broken down by the code that failed.

    This is the accounting the summary used to get wrong: the run reported no
    unmatched entries at all while every row of an unresolved set disappeared
    between the catalog and the corpus.
    """
    index = SetIndex(by_set_number={("sv06", "130"): "sv06-130"}, by_name={})
    entries = [
        entry("Dragapult ex", "SV6", "130"),
        entry("Mega Gardevoir ex", "ME9", "1"),
        entry("Mega Gardevoir ex", "ME9", "1"),
        entry("Oddish", "", "1"),
    ]
    found = match_entries(entries, {"SV6": "sv06"}, index)
    assert found.card_ids == ["sv06-130"]
    assert found.unmatched == 3
    assert dict(found.unresolved_by_set) == {"ME9": 2, NO_SET_CODE: 1}


def test_match_entries_counts_an_ordinary_miss_too(no_name_query: None) -> None:
    """A resolved set whose card is not there is unmatched, and not against a set.

    The per-set breakdown is about sets that could not be reached at all, so an
    entry whose set was listed and whose card was simply not found belongs in
    the total and nowhere else.
    """
    index = SetIndex(by_set_number={}, by_name={})
    found = match_entries([entry("A Card Nobody Has", "SV6", "999")], {"SV6": "sv06"}, index)
    assert found.card_ids == []
    assert found.unmatched == 1
    assert dict(found.unresolved_by_set) == {}
