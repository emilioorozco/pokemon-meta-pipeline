"""The Pydantic contract tracks contract/parsed-blob.schema.json: keys, optionality, enums."""

import copy
import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import jsonschema
import pytest
from pydantic import BaseModel

from pipeline.config import REPO_ROOT
from pipeline.contract import (
    CONTRACT_VERSION,
    ActionKind,
    CardRef,
    CompetitiveElo,
    ContractError,
    Decklist,
    DeckMeta,
    Entry,
    GameSummary,
    ObservedCards,
    ParsedBlobV1,
    ParsedBlobV2,
    RoleStats,
    Segment,
    SideStats,
    SubEntry,
    parse_blob,
)

SCHEMA_PATH = REPO_ROOT / "contract" / "parsed-blob.schema.json"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())

SUMMARY_PATH = ("properties", "summary")
SEGMENT_PATH = ("properties", "segments", "items")
ENTRY_PATH = (*SEGMENT_PATH, "properties", "entries", "items")
SUB_ENTRY_PATH = (*ENTRY_PATH, "properties", "subs", "items")

# (id, path of the object inside the schema, model that mirrors it)
OBJECTS: list[tuple[str, tuple[str, ...], type[BaseModel]]] = [
    ("top level", (), ParsedBlobV2),
    ("summary", SUMMARY_PATH, GameSummary),
    ("summary.stats", (*SUMMARY_PATH, "properties", "stats"), RoleStats),
    ("summary.observedCards", (*SUMMARY_PATH, "properties", "observedCards"), ObservedCards),
    ("summary.myDeckMeta", (*SUMMARY_PATH, "properties", "myDeckMeta"), DeckMeta),
    ("summary.opponentDeckMeta", (*SUMMARY_PATH, "properties", "opponentDeckMeta"), DeckMeta),
    ("summary.elo", (*SUMMARY_PATH, "properties", "elo"), CompetitiveElo),
    ("segment", SEGMENT_PATH, Segment),
    ("entry", ENTRY_PATH, Entry),
    ("sub-entry", SUB_ENTRY_PATH, SubEntry),
    ("side stats", ("properties", "statsByPlayer", "additionalProperties"), SideStats),
    ("decklist", ("properties", "myDecklist"), Decklist),
    ("card ref", ("properties", "myDecklist", "properties", "cards", "items"), CardRef),
]


def _at(*path: str) -> dict[str, Any]:
    node: Any = SCHEMA
    for key in path:
        node = node[key]
    assert isinstance(node, dict)
    return node


def _find_enum_with(node: Any, member: str) -> list[str] | None:
    """First `enum` list anywhere under `node` that contains `member`."""
    if isinstance(node, dict):
        enum = node.get("enum")
        if isinstance(enum, list) and member in enum:
            return enum
        children: list[Any] = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_enum_with(child, member)
        if found is not None:
            return found
    return None


def _allowed_values(annotation: Any) -> set[Any] | None:
    """Values an enumerated annotation accepts (Literal args or enum values); None if open."""
    if get_origin(annotation) in (UnionType, Union):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) != 1:
            return None
        annotation = members[0]
    if get_origin(annotation) is Literal:
        return set(get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {member.value for member in annotation}
    return None


def _compare_object(node: dict[str, Any], model: type[BaseModel]) -> list[str]:
    """Differences between one schema object and one model; empty when they agree."""
    problems: list[str] = []
    props: dict[str, Any] = node["properties"]
    schema_required = set(node.get("required", []))
    schema_optional = set(props) - schema_required

    model_required: set[str] = set()
    model_optional: set[str] = set()
    by_alias = {}
    for name, info in model.model_fields.items():
        alias = info.alias or name
        by_alias[alias] = info
        (model_required if info.is_required() else model_optional).add(alias)

    if schema_required != model_required:
        problems.append(f"required: {sorted(schema_required)} vs {sorted(model_required)}")
    if schema_optional != model_optional:
        problems.append(f"optional: {sorted(schema_optional)} vs {sorted(model_optional)}")

    schema_forbids = node.get("additionalProperties") is False
    model_forbids = model.model_config.get("extra") == "forbid"
    if schema_forbids != model_forbids:
        problems.append(f"extra keys: forbids {schema_forbids} vs {model_forbids}")

    for key, prop in props.items():
        if "enum" in prop and key in by_alias:
            allowed = _allowed_values(by_alias[key].annotation)
            if allowed != set(prop["enum"]):
                problems.append(f"{key} enum: {sorted(prop['enum'])} vs {allowed}")
    return problems


def _validate_schema(instance: Any, node: dict[str, Any] = SCHEMA) -> None:
    jsonschema.Draft7Validator(node).validate(instance)


# ---- hand-built blobs ----


def _side_stats() -> dict[str, Any]:
    return {
        "cardsDrawn": 7,
        "energyAttached": 1,
        "damageDealt": 30,
        "knockouts": 0,
        "prizesTaken": 0,
        "mulligans": 0,
        "turnsTaken": 1,
        "pokemonPlayed": ["Pikachu"],
        "cardsPlayed": [],
        "evolutions": [],
        "attacks": ["Thunder Shock"],
    }


def _sub_entry(kind: str = "draw") -> dict[str, Any]:
    return {"line": 4, "text": "- P1 drew a card.", "kind": kind, "fields": {"n": 1}, "details": []}


def _entry(kind: str = "play_pokemon") -> dict[str, Any]:
    return {
        "line": 3,
        "text": "P1 played Pikachu to the Bench.",
        "kind": kind,
        "actor": "P1",
        "fields": {"card": "Pikachu", "to": "Bench", "energy": False},
        "subs": [_sub_entry()],
    }


def _summary() -> dict[str, Any]:
    return {
        "gameId": "0123456789abcdef",
        "userId": "user-1",
        "uploadedAt": "2026-09-01T12:00:00.000Z",
        "playedAt": "2026-09-01T11:30:00.000Z",
        "exportVariant": "stock",
        "parserVersion": 7,
        "unparsedCount": 0,
        "players": ["P1", "P2"],
        "mySide": 0,
        "result": "win",
        "endReason": "prizes",
        "turnCount": 1,
        "stats": {"me": _side_stats()},
        "hasFullDecklists": False,
    }


def _v1_blob() -> dict[str, Any]:
    return {
        "segments": [
            {"kind": "setup", "title": "Setup", "entries": []},
            {
                "kind": "turn",
                "turnNumber": 1,
                "player": "P1",
                "title": "P1's Turn",
                "entries": [_entry()],
            },
        ],
        "statsByPlayer": {"P1": _side_stats(), "P2": _side_stats()},
        "unparsedLines": [],
        "extras": {},
    }


def _v2_blob() -> dict[str, Any]:
    return {"schemaVersion": CONTRACT_VERSION, "summary": _summary(), **_v1_blob()}


# ---- schema and model agree ----


def test_contract_version_matches_schema() -> None:
    jsonschema.Draft7Validator.check_schema(SCHEMA)
    assert SCHEMA["title"] == "ParsedBlobV2"
    assert SCHEMA["properties"]["schemaVersion"]["const"] == CONTRACT_VERSION


def test_action_kinds_match_schema() -> None:
    kinds = _find_enum_with(SCHEMA, "opening_hand")
    assert kinds is not None
    assert len(kinds) == len(set(kinds))
    assert {kind.value for kind in ActionKind} == set(kinds)
    assert _find_enum_with(_at(*SUB_ENTRY_PATH), "opening_hand") == kinds


@pytest.mark.parametrize(
    ("path", "model"),
    [(path, model) for _, path, model in OBJECTS],
    ids=[name for name, _, _ in OBJECTS],
)
def test_object_keys_and_enums_match(path: tuple[str, ...], model: type[BaseModel]) -> None:
    assert _compare_object(_at(*path), model) == []


def test_compare_object_reports_drift() -> None:
    """The helper itself must notice a missing key, or every test above is vacuous."""
    node = copy.deepcopy(_at(*SEGMENT_PATH))
    node["properties"]["extraKey"] = {"type": "string"}
    node["required"].append("extraKey")
    node["properties"]["kind"]["enum"].append("bonus")
    node["additionalProperties"] = True
    problems = _compare_object(node, Segment)
    assert [p.split(":")[0] for p in problems] == ["required", "extra keys", "kind enum"]


# ---- valid blobs ----


def test_minimal_v2_blob_passes_schema_and_model() -> None:
    blob = _v2_blob()
    _validate_schema(blob)
    parsed = ParsedBlobV2.model_validate(blob)
    assert parsed.schema_version == CONTRACT_VERSION
    assert parsed.summary.has_full_decklists is False
    assert parsed.summary.stats.opponent is None
    assert parsed.segments[1].turn_number == 1
    assert parsed.segments[1].entries[0].subs[0].kind is ActionKind.DRAW
    assert set(parsed.stats_by_player) == {"P1", "P2"}
    # the wire names survive a round trip
    assert parsed.model_dump(mode="json", by_alias=True, exclude_unset=True) == blob


def test_v2_blob_with_the_long_tail() -> None:
    blob = _v2_blob()
    card = {"cardId": "sv6_25", "baseCardId": "sv6_25", "name": "Pikachu", "count": 4}
    decklist = {"cards": [card], "cardCount": 4, "complete": False, "source": "paste"}
    blob["myDecklist"] = decklist
    blob["opponentDecklist"] = {**decklist, "source": "inferred"}
    blob["extras"] = {"Region": ["us-east-1"]}
    blob["summary"].update(
        {
            "playedAtSource": "log",
            "mySideSource": "handle",
            "opponentName": "P2",
            "winner": "P1",
            "wentFirst": True,
            "wonCoinToss": False,
            "uploadSource": "overlay-mac",
            "uploadClient": "tcgl-shortcut/2",
            "myDeckMeta": {"tcglDeckId": "d1", "size": 60},
            "myDecklistSource": "paste",
            "observedCards": {"me": [card], "opponent": []},
            "inferredPrizes": [{"name": "Rare Candy", "count": 1}],
            "opponentArchetype": "Charizard / Pidgeot",
            "opponentArchetypeSource": "auto",
            "opponentArchetypeResolution": "exact",
            "seasonId": "s1",
            "seasonSource": "log",
            "elo": {
                "seasonId": "s1",
                "previousElo": 1500,
                "newElo": 1516,
                "delta": 16,
                "modeElos": {"standard": 1516},
            },
            "excludedFromStats": False,
        }
    )
    _validate_schema(blob)
    parsed = parse_blob(blob)
    assert isinstance(parsed, ParsedBlobV2)
    assert parsed.my_decklist is not None
    assert parsed.my_decklist.cards[0].card_id == "sv6_25"
    assert parsed.summary.elo is not None
    assert parsed.summary.elo.mode_elos == {"standard": 1516}
    assert parsed.model_dump(mode="json", by_alias=True, exclude_unset=True) == blob


def test_v1_blob_dispatches_on_missing_schema_version() -> None:
    blob = _v1_blob()
    parsed = parse_blob(blob)
    assert isinstance(parsed, ParsedBlobV1)
    assert parsed.model_dump(mode="json", by_alias=True, exclude_unset=True) == blob
    # the same body with the two v2 keys is the v2 blob, and the schema agrees
    v2 = {"schemaVersion": 2, "summary": _summary(), **blob}
    _validate_schema(v2)
    assert isinstance(parse_blob(v2), ParsedBlobV2)


# ---- malformed blobs ----


def _drop_summary_field(blob: dict[str, Any]) -> None:
    del blob["summary"]["gameId"]


def _unknown_kind(blob: dict[str, Any]) -> None:
    blob["segments"][1]["entries"][0]["kind"] = "teleport"


def _wrong_turn_number_type(blob: dict[str, Any]) -> None:
    blob["segments"][1]["turnNumber"] = "1"


def _unknown_top_level_key(blob: dict[str, Any]) -> None:
    blob["bogus"] = 1


def _unknown_sub_entry_key(blob: dict[str, Any]) -> None:
    blob["segments"][1]["entries"][0]["subs"][0]["subs"] = []


def _nested_field_value(blob: dict[str, Any]) -> None:
    blob["segments"][1]["entries"][0]["fields"]["card"] = {"name": "Pikachu"}


MALFORMED: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("missing summary field", _drop_summary_field, "summary.gameId"),
    ("unknown kind", _unknown_kind, "segments.1.entries.0.kind"),
    ("wrong turnNumber type", _wrong_turn_number_type, "segments.1.turnNumber"),
    ("unknown top-level key", _unknown_top_level_key, "bogus"),
    ("unknown sub-entry key", _unknown_sub_entry_key, "segments.1.entries.0.subs.0.subs"),
    ("nested field value", _nested_field_value, "segments.1.entries.0.fields.card"),
]


@pytest.mark.parametrize(
    ("mutate", "path"),
    [(mutate, path) for _, mutate, path in MALFORMED],
    ids=[name for name, _, _ in MALFORMED],
)
def test_malformed_blob_fails_both_validators(
    mutate: Callable[[dict[str, Any]], None], path: str
) -> None:
    blob = _v2_blob()
    mutate(blob)
    with pytest.raises(jsonschema.ValidationError):
        _validate_schema(blob)
    with pytest.raises(ContractError) as excinfo:
        parse_blob(blob)
    assert path in excinfo.value.summary()
    assert excinfo.value.schema_version_seen == CONTRACT_VERSION


def test_unsupported_schema_version_names_the_key() -> None:
    blob = {**_v2_blob(), "schemaVersion": 3}
    with pytest.raises(ContractError) as excinfo:
        parse_blob(blob)
    assert excinfo.value.schema_version_seen == 3
    assert excinfo.value.summary().startswith("1 error(s): schemaVersion:")


def test_contract_error_summary_truncates() -> None:
    with pytest.raises(ContractError) as excinfo:
        parse_blob({"schemaVersion": 2})
    err = excinfo.value
    assert len(err.cause.errors()) == 5
    assert err.summary().startswith("5 error(s): summary: Field required; segments:")
    assert err.summary().endswith("(+2 more)")
    assert err.summary(limit=10).endswith("extras: Field required")
    assert str(err) == str(err.cause)


def test_non_object_body_is_a_contract_error() -> None:
    with pytest.raises(ContractError) as excinfo:
        parse_blob(["not", "a", "blob"])
    assert excinfo.value.schema_version_seen is None
    assert excinfo.value.summary().startswith("1 error(s): <root>:")


# ---- one per action kind ----


@pytest.mark.parametrize("kind", list(ActionKind), ids=[kind.value for kind in ActionKind])
def test_every_action_kind_validates(kind: ActionKind) -> None:
    entry = _entry(kind.value)
    entry["subs"] = [_sub_entry(kind.value)]
    _validate_schema(entry, _at(*ENTRY_PATH))
    parsed = Entry.model_validate(entry)
    assert parsed.kind is kind
    assert parsed.subs[0].kind is kind


# ---- real fixtures, once they exist ----

FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=[path.name for path in FIXTURE_FILES])
def test_fixture_blob_validates(path: Path) -> None:
    data = json.loads(path.read_text())
    parsed = parse_blob(data)
    if isinstance(parsed, ParsedBlobV2):
        _validate_schema(data)
