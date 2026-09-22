"""Pydantic mirror of the parsed-blob contract.

The wire format is contract/parsed-blob.schema.json (JSON Schema draft-07, title
ParsedBlobV2), exported from the producer's Zod schema. Every class below tracks one
object in that file: same keys, same optionality, same enums. The file sets
`additionalProperties: false` on every object, so every model forbids unknown keys;
the only open records are the ones the file declares as records (`fields`,
`statsByPlayer`, `extras`, `elo.modeElos`). tests/test_contract.py walks both sides and
fails on drift.

Names are snake_case here and camelCase on the wire (`alias_generator=to_camel`).
Dump with `by_alias=True` to get the wire names back. Validation is strict: JSON types
are not coerced, so "3" is not an integer, matching jsonschema.

Entry is one class rather than a discriminated union over ActionKind. The producer does
not type `fields` per kind: the schema declares it as a flat record of
string | number | boolean for every kind, so there is no per-kind shape to discriminate
into. The per-kind field names in docs/schema.md section 4 stay documentation until the
producer types them.

Optional keys are `X | None = None`. An explicit null on such a key passes here but not
the schema, which only allows the key to be absent; that gap is not worth a sentinel type.

ParsedBlobV1 is the pre-contract shape: the v2 blob minus `schemaVersion` and `summary`
(docs/schema.md section 1). It has no schema file upstream.
"""

from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

CONTRACT_VERSION: Final = 2


class ActionKind(StrEnum):
    """Every `kind` an entry or sub-entry can carry; the order follows the schema."""

    OPENING_HAND = "opening_hand"
    MULLIGAN = "mulligan"
    MULLIGAN_DRAW = "mulligan_draw"
    COIN_CHOICE = "coin_choice"
    COIN_TOSS = "coin_toss"
    GO_FIRST = "go_first"
    DRAW = "draw"
    DRAW_NAMED = "draw_named"
    PLAY_POKEMON = "play_pokemon"
    PLAY_STADIUM = "play_stadium"
    PLAY_CARD = "play_card"
    ATTACH = "attach"
    EVOLVE = "evolve"
    ATTACK = "attack"
    USE = "use"
    KNOCKOUT = "knockout"
    PRIZE = "prize"
    RETREAT = "retreat"
    PROMOTE = "promote"
    SWITCH = "switch"
    END_TURN = "end_turn"
    TIMEOUT = "timeout"
    CONCEDE = "concede"
    WIN = "win"
    DISCARD = "discard"
    DISCARD_NAMED = "discard_named"
    DISCARD_FROM = "discard_from"
    SHUFFLE = "shuffle"
    SHUFFLE_INTO = "shuffle_into"
    TO_HAND = "to_hand"
    ACTIVATED = "activated"
    CONDITION_DAMAGE = "condition_damage"
    STATUS = "status"
    STATUS_END = "status_end"
    PUT_COUNTERS = "put_counters"
    MOVE_COUNTERS = "move_counters"
    MOVE_TO_HAND = "move_to_hand"
    PUT_DECK = "put_deck"
    PREVENTED = "prevented"
    TOOK_DAMAGE = "took_damage"
    TERA = "tera"
    COIN_FLIP = "coin_flip"
    CHOICE = "choice"
    BREAKDOWN = "breakdown"
    DRAWN_CARDS = "drawn_cards"
    REVEALED = "revealed"
    OTHER = "other"


class SegmentKind(StrEnum):
    SETUP = "setup"
    TURN = "turn"
    CHECKUP = "checkup"
    OTHER = "other"


class DecklistSource(StrEnum):
    DEBUG = "debug"
    PASTE = "paste"
    API = "api"
    INFERRED = "inferred"


ExportVariant = Literal["stock", "debug", "manual"]
PlayedAtSource = Literal["log", "upload", "user"]
MySideSource = Literal["debug", "concede", "named_draws", "handle", "user"]
GameResult = Literal["win", "loss", "tie", "unknown"]
EndReason = Literal["concede", "opponent_concede", "prizes", "other", "unknown"]
UploadSource = Literal[
    "web", "overlay", "overlay-mac", "overlay-windows", "ios-shortcut", "manual", "unknown"
]
ArchetypeSource = Literal["auto", "user"]
ArchetypeResolution = Literal["exact", "alias", "created"]
SeasonSource = Literal["log", "user"]

FieldValue = str | int | float | bool


class ContractModel(BaseModel):
    """Shared config: camelCase on the wire, unknown keys rejected, no type coercion.

    Strict mode takes only enum instances from Python input, and blobs arrive as decoded
    JSON with plain strings, so every StrEnum-typed field opts back into value matching
    with `Field(strict=False)`. Literal-typed fields already accept their strings.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        strict=True,
    )


class SideStats(ContractModel):
    cards_drawn: int
    energy_attached: int
    damage_dealt: int
    knockouts: int
    prizes_taken: int
    mulligans: int
    turns_taken: int
    pokemon_played: list[str]
    cards_played: list[str]
    evolutions: list[str]
    attacks: list[str]


class CardRef(ContractModel):
    card_id: str | None = None
    base_card_id: str | None = None
    name: str | None = None
    set: str | None = None
    number: str | None = None
    count: int = Field(ge=0)


class Decklist(ContractModel):
    cards: list[CardRef]
    card_count: int = Field(ge=0)
    complete: bool
    source: DecklistSource = Field(strict=False)


class DeckMeta(ContractModel):
    tcgl_deck_id: str | None = None
    deck_name: str | None = None
    deck_definition_id: str | None = None
    size: int | None = None
    sleeve: str | None = None
    coin: str | None = None
    deck_box: str | None = None


class RoleStats(ContractModel):
    """`summary.stats`: the two sides re-keyed by role; either may be missing."""

    me: SideStats | None = None
    opponent: SideStats | None = None


class ObservedCards(ContractModel):
    me: list[CardRef] | None = None
    opponent: list[CardRef] | None = None


class CompetitiveElo(ContractModel):
    season_id: str
    season_name: str | None = None
    mode: str | None = None
    previous_elo: int
    new_elo: int
    delta: int
    modes: list[str] | None = None
    mode_elos: dict[str, int] | None = None
    default_elo: int | None = None


class GameSummary(ContractModel):
    game_id: str
    user_id: str
    uploaded_at: str
    played_at: str
    played_at_source: PlayedAtSource | None = None
    export_variant: ExportVariant
    parser_version: int
    unparsed_count: int
    players: list[str]
    my_side: int | None
    my_side_source: MySideSource | None = None
    opponent_name: str | None = None
    result: GameResult
    end_reason: EndReason
    winner: str | None = None
    went_first: bool | None = None
    won_coin_toss: bool | None = None
    turn_count: int
    stats: RoleStats
    match_id: str | None = None
    upload_source: UploadSource | None = None
    upload_token_id: str | None = None
    upload_client: str | None = Field(default=None, max_length=60)
    deck_id: str | None = None
    deck_version: int | None = None
    deck_name: str | None = None
    my_deck_meta: DeckMeta | None = None
    opponent_deck_meta: DeckMeta | None = None
    my_decklist_source: DecklistSource | None = Field(default=None, strict=False)
    opponent_decklist_source: DecklistSource | None = Field(default=None, strict=False)
    has_full_decklists: bool
    observed_cards: ObservedCards | None = None
    inferred_prizes: list[CardRef] | None = None
    prize_cards_taken: list[CardRef] | None = None
    opponent_archetype: str | None = None
    my_archetype_id: str | None = None
    my_archetype: str | None = None
    my_archetype_source: ArchetypeSource | None = None
    my_archetype_resolution: ArchetypeResolution | None = None
    opponent_archetype_id: str | None = None
    opponent_archetype_source: ArchetypeSource | None = None
    opponent_archetype_resolution: ArchetypeResolution | None = None
    tournament_id: str | None = None
    tournament_name: str | None = None
    round: str | None = None
    notes: str | None = None
    season_id: str | None = None
    season_name: str | None = None
    season_source: SeasonSource | None = None
    elo: CompetitiveElo | None = None
    excluded_from_stats: bool | None = None


class Action(ContractModel):
    """What an entry and a sub-entry share; not a wire object on its own."""

    line: int
    text: str
    kind: ActionKind = Field(strict=False)
    actor: str | None = None
    fields: dict[str, FieldValue]


class SubEntry(Action):
    details: list[str]


class Entry(Action):
    subs: list[SubEntry]


class Segment(ContractModel):
    kind: SegmentKind = Field(strict=False)
    turn_number: int | None = None
    player: str | None = None
    title: str
    entries: list[Entry]


class ParsedBlobV1(ContractModel):
    segments: list[Segment]
    stats_by_player: dict[str, SideStats]
    unparsed_lines: list[str]
    extras: dict[str, list[str]]
    my_decklist: Decklist | None = None
    opponent_decklist: Decklist | None = None


class ParsedBlobV2(ContractModel):
    schema_version: Literal[2]
    summary: GameSummary
    segments: list[Segment]
    stats_by_player: dict[str, SideStats]
    unparsed_lines: list[str]
    extras: dict[str, list[str]]
    my_decklist: Decklist | None = None
    opponent_decklist: Decklist | None = None


class ContractError(ValueError):
    """A blob that does not satisfy the contract; wraps the pydantic ValidationError.

    `schema_version_seen` is whatever the body carried under `schemaVersion` (None when the
    key is absent), so a quarantine record can be written without re-reading the body.
    """

    def __init__(self, cause: ValidationError, schema_version_seen: Any = None) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.schema_version_seen = schema_version_seen

    def summary(self, limit: int = 3) -> str:
        """One line: error count, then the first `limit` locations with their messages."""
        errors = self.cause.errors()
        parts = []
        for error in errors[:limit]:
            loc = ".".join(str(part) for part in error["loc"]) or "<root>"
            parts.append(f"{loc}: {error['msg']}")
        more = len(errors) - limit
        tail = f" (+{more} more)" if more > 0 else ""
        return f"{len(errors)} error(s): " + "; ".join(parts) + tail


def parse_blob(data: Any) -> ParsedBlobV1 | ParsedBlobV2:
    """Validate a decoded blob as v1 or v2, dispatching on the presence of `schemaVersion`.

    A `schemaVersion` other than 2 is a v2 parse that fails on that key, so the error names
    the version rather than complaining about a missing `summary`.
    """
    version: Any = None
    is_v1 = True
    if isinstance(data, dict):
        is_v1 = "schemaVersion" not in data
        version = data.get("schemaVersion")
    try:
        if is_v1:
            return ParsedBlobV1.model_validate(data)
        return ParsedBlobV2.model_validate(data)
    except ValidationError as exc:
        raise ContractError(exc, schema_version_seen=version) from exc
