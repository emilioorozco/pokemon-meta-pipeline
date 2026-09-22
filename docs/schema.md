# Schema: the data contract and the bronze tables

Field names, types and meanings below are taken from the upstream source
(`play-rough-analytics/packages/shared/src/battle-log/parser.ts`,
`.../battle-log/card-ref.ts`, `.../contracts/index.ts`,
`play-rough-analytics/apps/api/src/games/service.ts`). Types are given in JSON
terms. "Nullable" covers both `null` and absent unless stated. Rows marked
"planned (v2)" do not exist upstream yet.

## 1. Blob: top level

Object at `parsed/{userId}/{gameId}.json`.

| Field | Type | Nullable | Meaning | Source |
|---|---|---|---|---|
| `schemaVersion` | integer (`2`) | absent in v1 | Contract version. Absent means v1. | planned (v2) |
| `summary` | object, see section 2 | absent in v1 | The per-game summary row, identical to `gameSummarySchema`. | planned (v2) |
| `segments` | `Segment[]` | no | The battle log split into setup, turn and checkup blocks. | `ParsedBattleLog.segments` |
| `statsByPlayer` | `{ [handle: string]: SideStats }` | no | Per-side counters keyed by the in-game handle as printed. | `ParsedBattleLog.statsByPlayer` |
| `unparsedLines` | `string[]` | no | Lines (entries or sub-entries) no pattern matched, verbatim. | `ParsedBattleLog.unparsedLines` |
| `extras` | `{ [tag: string]: string[] }` | no (may be `{}`) | Unknown `[Tag] ...` lines from the preamble and the trailer, minus `[CompetitiveElo]`. Preamble lines that are not tagged go under `unparsed`. | `parsePreamble().extras` plus trailer loop in `parseExportText` |
| `myDecklist` | `Decklist` | yes | Owner's full list when a debug preamble or a paste supplied it. | preamble `local`, or `attachDecklist` |
| `opponentDecklist` | `Decklist` | yes | Opponent's full list, same sources. | preamble `opponent`, or `attachDecklist` |

## 2. Summary (`gameSummarySchema`)

Today this row lives in DynamoDB; in v2 it is embedded as `summary`. Identity,
time and outcome fields first; the long tail is grouped.

| Field | Type | Nullable | Meaning | Source |
|---|---|---|---|---|
| `gameId` | string (16 hex) | no | First 16 hex chars of SHA-256 over the normalized battle log. Manual games use a generated id. | `gameIdFor` |
| `userId` | string | no | Uploading account. Same value as in the S3 key. | auth session |
| `uploadedAt` | ISO 8601 string | no | When the upload happened. | server clock |
| `playedAt` | ISO 8601 string | no | When the game was played. Defaults to `uploadedAt` when the uploader sends none. | request or server clock |
| `exportVariant` | `"stock" \| "debug" \| "manual"` | no | Text shape of the export. See discovery for the `debug` caveat. | `splitExport().variant`, or `"manual"` |
| `parserVersion` | integer | no | Parser that produced the blob. `0` for manual games. | `PARSER_VERSION` |
| `unparsedCount` | integer | no | `unparsedLines.length` at parse time. | parser |
| `players` | `string[]` | no | Handles in order of first appearance; index is the seat. Two entries for parsed games. | `discoverPlayers` |
| `mySide` | integer or `null` | `null` allowed | Seat index of the owner, `null` when undetermined. | `resolveSide` |
| `mySideSource` | `"debug" \| "concede" \| "named_draws" \| "handle" \| "user"` | yes | How the seat was determined. | `resolveSide` |
| `opponentName` | string | yes | `players[1 - mySide]`. Absent when `mySide` is `null`. | `buildSummaryFields` |
| `result` | `"win" \| "loss" \| "tie" \| "unknown"` | no | From the owner's point of view. `tie` only for manual games. `unknown` when `mySide` is `null` or no winner was found. | `buildSummaryFields` |
| `endReason` | `"concede" \| "opponent_concede" \| "prizes" \| "other" \| "unknown"` | no | How the game ended. | `deriveOutcome` |
| `winner` | string | yes | Winning handle when a win or concede line was found. | `deriveOutcome` |
| `wentFirst` | boolean | yes | Owner went first. From the `go_first` action. | `deriveOutcome` |
| `wonCoinToss` | boolean | yes | Owner won the opening toss. | `deriveOutcome` |
| `turnCount` | integer | no | Number of `turn` segments. | parser |
| `stats` | `{ me?: SideStats, opponent?: SideStats }` | no (may be `{}`) | The owner's and opponent's counters, re-keyed by role. | `statsByPlayer` |
| `excludedFromStats` | boolean | yes | Owner chose to keep the game out of analytics. Filter it everywhere. | `setGameExcluded` |
| `hasFullDecklists` | boolean | yes | Both decklists present and complete. | `myDecklist` / `opponentDecklist` |
| `observedCards` | `{ me?: CardRef[], opponent?: CardRef[] }` | yes | Cards each side made public ("cards seen"). Counts capped at 4 except basic energy. | `observedCards()` |
| `opponentArchetype` | string | yes | Derived label, `"A / B"` of the two most prominent Pokemon, or the archetype row's name after linking. | `archetypeFromDecklist` or `archetypeFromStats` |
| `opponentArchetypeId` | string | yes | Shared archetype row behind the name. | `linkOpponentArchetype` |
| `opponentArchetypeSource` | `"auto" \| "user"` | yes | `user` pins the value against re-parse. | service |
| `opponentArchetypeResolution` | `"exact" \| "alias" \| "created"` | yes | How the auto-link matched the name; `alias` is worth a spot-check. | `ensureArchetypeByName` |
| `myArchetypeId`, `myArchetype` | string | yes | Manual games only: the owner's archetype. Uploaded games carry the owner's deck through `deckId` instead. | manual service |
| `seasonId`, `seasonName`, `seasonSource` | string, string, `"log" \| "user"` | yes | Season from the `[CompetitiveElo]` trailer or set by the owner. | `findCompetitiveElo`, `ensureSeason` |
| `elo` | `CompetitiveElo` minus `raw` | yes | `seasonId`, `seasonName?`, `mode?`, `previousElo`, `newElo`, `delta`, `modes?`, `modeElos?`, `defaultElo?`. Debug client only. | `competitive-elo.ts` |
| Deck link | `deckId?`, `deckVersion?`, `deckName?`, `myDeckMeta?`, `opponentDeckMeta?`, `myDecklistSource?`, `opponentDecklistSource?` | yes | Owner's deck record and version, and the preamble's deck metadata (`tcglDeckId`, `deckName`, `deckDefinitionId`, `size`, `sleeve`, `coin`, `deckBox`). | `linkDeck`, preamble |
| Prizes | `inferredPrizes?`, `prizeCardsTaken?` | `CardRef[]`, yes | Debug preamble `[DeckSearch]` and `[PrizeTaken]` lines. | preamble |
| Upload channel | `matchId?`, `uploadSource?`, `uploadTokenId?`, `uploadClient?` | yes | Where the upload came from (`web`, `overlay`, `overlay-mac`, `overlay-windows`, `ios-shortcut`, `manual`, `unknown`). | `stampUploadOrigin` |
| Event context | `tournamentId?`, `tournamentName?`, `round?`, `notes?` | yes | Manual-game and tournament bookkeeping. | manual service |

## 3. Segment

| Field | Type | Nullable | Meaning | Source |
|---|---|---|---|---|
| `kind` | `"setup" \| "turn" \| "checkup" \| "other"` | no | `Setup` header, `<handle>'s Turn` header, `Pokémon Checkup` header, or lines before any header. | `parseBattleLog` |
| `turnNumber` | integer | only on `turn` | 1-based running count of turn segments in the log (not the client's `Turn #` number). | parser |
| `player` | string (handle) | only on `turn` | Whose turn it is. | turn header |
| `title` | string | no | The header line as printed, including the handle for turns. Empty for `other`. | header line |
| `entries` | `Entry[]` | no | Top-level lines in this block, in order. | parser |

## 4. Entry and sub-entry

An `Entry` is an `Action` with `subs: SubEntry[]`; a `SubEntry` is an `Action`
with `details: string[]`. Sub-entries are the `- ` lines under an entry;
details are the indented bullet lines under a sub-entry (a detail directly
under an entry creates a synthetic sub-entry with empty `text` and kind
`other`).

| Field | Type | Nullable | Meaning | Source |
|---|---|---|---|---|
| `line` | integer | no | 1-based line number in the battle-log text. | parser |
| `text` | string | no | The line as printed (leading `- ` removed for subs), still containing handles and any `(clientId)` prefixes. | parser |
| `kind` | `ActionKind` | no | One of the 47 kinds below. | pattern table |
| `actor` | string (handle) | yes | Player the line is attributed to: the leading handle for actor-first kinds, the winner for `concede` and `win`, the owner for `discard_from`, `to_hand`, `condition_damage`. | `classify` |
| `fields` | `{ [key]: string \| number \| boolean }` | no (may be `{}`) | Pattern captures. Name-bearing keys (`card`, `pokemon`, `to`, `from`, `target`, `previous`) have the `(clientId)` prefix stripped. | pattern `map` |
| `subs` | `SubEntry[]` | entries only | Nested lines. | parser |
| `details` | `string[]` | sub-entries only | Bullet lines under the sub. | parser |

### Action kinds (47)

Grouped for reading; the enum is flat. Field names in parentheses are the
`fields` keys that kind emits.

- Setup and turn order: `opening_hand` (n), `mulligan`, `mulligan_draw` (n),
  `coin_choice` (choice), `coin_toss` (won), `go_first` (first), `revealed`
  (mulligan).
- Drawing and hand: `draw` (n, toBench?), `draw_named` (card, n),
  `drawn_cards` (n), `to_hand` (card, owner), `move_to_hand` (card).
- Playing cards: `play_pokemon` (card, to), `play_stadium` (card), `play_card`
  (card), `attach` (card, target, energy), `evolve` (from, to), `tera`
  (pokemon), `activated` (card), `use` (pokemon, move), `choice` (choice).
- Combat: `attack` (pokemon, move, targetOwner, target, damage, weakness),
  `took_damage` (pokemon, damage), `prevented` (pokemon), `knockout`
  (pokemon), `prize` (n), `breakdown`.
- Board position: `retreat` (pokemon), `promote` (pokemon), `switch` (pokemon,
  previous).
- Special conditions and counters: `status` (pokemon, condition), `status_end`
  (pokemon, condition), `condition_damage` (n, owner, pokemon, condition),
  `put_counters` (n, owner, pokemon), `move_counters` (n).
- Deck and discard: `discard` (n), `discard_named` (card), `discard_from` (n
  or card, owner, pokemon), `shuffle`, `shuffle_into` (n or card), `put_deck`
  (n).
- Flow and outcome: `end_turn`, `timeout`, `concede` (who, winner or loser),
  `win` (reason, winner), `coin_flip` (result).
- Fallback: `other` (no fields; the line is also appended to `unparsedLines`).

## 5. Side stats (`SideStats`, `sideStatsSchema`)

Same shape under `statsByPlayer[handle]` and under `summary.stats.me` /
`summary.stats.opponent`.

| Field | Type | Meaning | Source |
|---|---|---|---|
| `cardsDrawn` | integer | Sum of `n` over `opening_hand`, `draw`, `draw_named`, `mulligan_draw`. | `deriveStats` |
| `energyAttached` | integer | Count of `attach` with `energy = true`. | `deriveStats` |
| `damageDealt` | integer | Sum of `attack.damage`. | `deriveStats` |
| `knockouts` | integer | Knockouts credited to this side (the other side's Pokemon was knocked out). | `deriveStats` |
| `prizesTaken` | integer | Sum of `prize.n`. | `deriveStats` |
| `mulligans` | integer | Count of `mulligan`. | `deriveStats` |
| `turnsTaken` | integer | Count of `turn` segments for this player. | `deriveStats` |
| `pokemonPlayed` | `string[]` | Card names from `play_pokemon`, with repeats. | `deriveStats` |
| `cardsPlayed` | `string[]` | Card names from `play_card` and `play_stadium`, with repeats. | `deriveStats` |
| `evolutions` | `string[]` | `evolve.to` names, with repeats. | `deriveStats` |
| `attacks` | `string[]` | Distinct attack names used. | `deriveStats` |

## 6. Decklist and card reference

`Decklist` (`decklistSchema`):

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `cards` | `CardRef[]` | no | The list. |
| `cardCount` | integer | no | Sum of `count`. |
| `complete` | boolean | no | `cardCount >= 60` (or the preamble's `size`). |
| `source` | `"debug" \| "paste" \| "api" \| "inferred"` | no | Where the list came from. `inferred` is used for deck versions built from `observedCards`. |

`CardRef` (`cardRefSchema`, `card-ref.ts`):

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `cardId` | string | yes | Client card id, for example `sv6_25`. Present for preamble and paste-with-set lists. |
| `baseCardId` | string | yes | `cardId` without a print suffix (`_ph`, `_sph`, `_sf`). |
| `name` | string | yes | Printed name. Present for battle-log-derived references. |
| `set`, `number` | string | yes | Set code and collector number from a pasted list. |
| `count` | integer >= 0 | no | Copies. |

At least one of `cardId` or `name` is present in practice; the schema does not
enforce it. Comparison key: lowercased `name` when known, else `baseCardId`.

Card catalog (`catalog/cards.json`): `{ [cardId]: { name, set, number, type?, hp?, reg? } }`.

## 7. Bronze table

Parquet under `data/lake/bronze/play_date=YYYY-MM-DD/part-0.parquet`, Hive
partitioning, one row per game. A run deletes and rewrites whole every partition
it touches, so re-ingesting the same games replaces them instead of appending
duplicates; the file is staged under a temporary name in the partition
directory and moved into place, so a reader never sees a half-written part.
Compression is zstd. Every string that can carry a handle passes through the
anonymizer before the write, and the batch is refused if a raw handle survives
the rewrite, so no handle is stored in bronze.

Bronze keeps the blob's shape rather than flattening it: `summary` is a struct,
`segments` a list of structs, the decklists structs. The seat and event grains
(section 8 names columns at those grains) are projections of these columns and
are produced in silver, where the joins that need them already live. Only v2
blobs are written; a v1 blob has no play date, so it is quarantined for an
upstream re-parse rather than landed, and there is no S3-last-modified fallback
in the table.

Every v2 game lands, whatever it carries. A game a modified client exported with
both complete decklists is written like any other, both decklist columns
included: `summary.hasFullDecklists` is informational, not a filter
([data-handling.md](data-handling.md)).

The Parquet schema is pinned from the contract models, not inferred from the
batch being written. Inference would type `summary.elo` as a struct in a batch
that has one and as null in a batch that does not, and a reader spanning both
partitions would see two incompatible schemas.

### Lineage columns

| Column | Type | Nullable | From |
|---|---|---|---|
| `game_id` | string | no | `summary.gameId`, lifted so a filter needs no struct access |
| `user_id` | string | no | `summary.userId` |
| `play_date` (partition) | string in the file, `DATE` from the Hive path | no | `summary.playedAt`, first 10 characters |
| `play_date_source` | string | no | `"summary"`; the only value while v1 blobs are excluded |
| `played_at` | timestamp (us, UTC) | no | `summary.playedAt` |
| `contract_version` | int32 | no | `2` |
| `source_key` | string | no | the S3 key the blob was read from |
| `source_version_id` | string | yes | S3 object version |
| `source_last_modified` | timestamp (us, UTC) | yes | S3 object last-modified |
| `ingested_at` | timestamp (us, UTC) | no | run time, the same value for the whole run |

### Blob columns

| Column | Type | Nullable | From |
|---|---|---|---|
| `summary` | struct, one field per section 2 row | no | `summary` |
| `segments` | list<struct> mirroring section 3, entries and sub-entries nested inside | no | `segments` |
| `stats_by_player` | list<struct<handle, stats>> | no | `statsByPlayer` |
| `unparsed_lines` | list<string> | no | `unparsedLines` |
| `extras` | list<struct<tag, lines>> | no | `extras` |
| `my_decklist`, `opponent_decklist` | struct per section 6 | yes | blob; null when the blob carried no such list |

Two conversions the Parquet types force:

- `fields` (section 4) is an open record: its values are
  `string | number | boolean` and its keys differ per `kind`, so there is no
  stable struct for it and Parquet has no heterogeneous map. Every entry and
  sub-entry carries `fields_json` instead, the record as compact JSON text with
  keys sorted. Readers use a JSON function on it; silver promotes the numeric
  fields it cares about (`n`, `damage`) to real columns.
- `statsByPlayer` and `extras` are records keyed by handle and by tag. A
  Parquet map with a struct value reads back awkwardly in DuckDB, so both
  become lists of structs and key order follows the blob.
  `summary.elo.modeElos` stays a map: its values are plain integers.

A smoke query reads the output back after every run:
`SELECT play_date, count(*) FROM read_parquet('bronze/**/*.parquet', hive_partitioning=true) GROUP BY 1`.

### Quarantine record

Two files per rejected object under `data/lake/quarantine/<reason>/`: the body
exactly as it was read, and a sidecar beside it. The source key becomes the
filename with its slashes as `__`, so one reason directory lists every object
that failed that way and a re-run overwrites its own records instead of piling
up copies. Nothing from a quarantined blob is written to bronze.

```
quarantine/contract_violation/parsed__user-1__game-5.json        body as received
quarantine/contract_violation/parsed__user-1__game-5.meta.json   sidecar
```

Sidecar fields:

| Field | Type | Meaning |
|---|---|---|
| `source_key` | string | Which object failed. |
| `reason` | string | One of `invalid_json`, `contract_violation`, `v1_blob`, `handle_leak_check_failed`, `write_failed`. |
| `detail` | string | Validator paths and messages, or the masked leak paths. |
| `contract_version_seen` | int or null | `schemaVersion` if the body parsed. |
| `quarantined_at` | timestamp | Run time. |

The split is a privacy boundary, not a layout preference. The body is kept
unanonymized, because a blob that failed validation cannot be trusted to
anonymize correctly, so it stays in the gitignored `data/` directory and is
never published. The sidecar is the part that gets grepped, pasted into an
issue and read across runs, so it carries no handle: `detail` holds paths and
validator messages only, and a leaking dict key is reported as `<key>`.

## 8. Silver tables

Parquet under `data/lake/silver/<table>/play_date=YYYY-MM-DD/`, written by
`python -m pipeline.silver` (PySpark). Every table is partitioned by
`play_date`, written with dynamic partition overwrite so a rerun replaces only
the days it touches, and projected onto a schema pinned in
`pipeline.silver.SILVER_SCHEMAS` rather than inferred from the batch, for the
same reason bronze pins its own. Silver filters nothing: excluded games and
manual games are present and flagged, and gold decides what to drop.

### 8.1 `games`

One row per game.

| Column | Type | Nullable | From |
|---|---|---|---|
| `game_id` | string | no | `summary.gameId` |
| `user_id` | string | no | `summary.userId` |
| `play_date` (partition) | date | no | bronze partition |
| `played_at` | timestamp | no | `summary.playedAt` |
| `played_at_source` | string | yes | `summary.playedAtSource` |
| `export_variant` | string | no | `stock`, `debug` or `manual` |
| `upload_source` | string | yes | `summary.uploadSource` |
| `turn_count` | int | no | `summary.turnCount` |
| `end_reason` | string | no | `summary.endReason` |
| `result` | string | no | the uploader's result, as the summary records it |
| `winner_seat` | int | yes | position of `summary.winner` in `summary.players` |
| `went_first_seat` | int | yes | `summary.wentFirst` read against `mySide` |
| `coin_toss_winner_seat` | int | yes | `summary.wonCoinToss` read against `mySide` |
| `first_player` | int | yes | seat of the player of the first `turn` segment |
| `excluded_from_stats` | boolean | no | `summary.excludedFromStats`, null read as false |
| `has_full_decklists` | boolean | yes | `summary.hasFullDecklists` |
| `my_side` | int | yes | `summary.mySide` |
| `season_id`, `season_name` | string | yes | `summary` |
| `parser_version`, `unparsed_count`, `contract_version` | int | no | `summary`, bronze |
| `source_key` | string | no | bronze |
| `ingested_at` | timestamp | no | bronze |

`went_first_seat` and `first_player` answer the same question from two sources:
the summary's `wentFirst` flag read against `mySide`, and the player named in
the header of the log's first turn. On the current corpus they agree on all 124
games that have a log. Both are kept because that agreement is a cheap ongoing
check on the producer's seat resolution, and it is the kind of thing that breaks
quietly: the legacy stage's `firstPlayer = -1` sentinel gave both seats
`went_first = false` until a check like this one caught it.

### 8.2 `game_sides`

Two rows per game, seat 0 and seat 1. The grain gold's fact table is built on.

| Column | Type | Nullable | From |
|---|---|---|---|
| `game_id`, `play_date` | string, date | no | `games` |
| `seat` | int | no | 0 or 1 |
| `is_uploader` | boolean | no | `seat == summary.mySide` |
| `player_token` | string | yes | `summary.players[seat]`, NULL for a stranger |
| `is_member` | boolean | no | the token holds an uploader seat somewhere in bronze |
| `archetype_id` | string | yes | `myArchetypeId` or `opponentArchetypeId` |
| `archetype_name` | string | yes | the canonical name for that id, see 8.5 |
| `archetype_name_raw` | string | yes | the label this row arrived with |
| `archetype_source` | string | yes | `auto`, `user`, `manual` or null |
| `result_for_seat` | string | no | `win`, `loss`, `tie` or `unknown` |
| `went_first` | boolean | yes | `went_first_seat == seat` |
| `stats_*` | int or list of string | yes | one column per `SideStats` field (section 5), matched to the seat by handle; null for a manual game |
| `decklist_source`, `decklist_complete`, `decklist_card_count` | string, boolean, int | yes | the seat's decklist (section 6), null when it shared none |
| `deck_name` | string | yes | `summary.deckName`, else `myDeckMeta.deckName`; uploader seat only, null on the other |
| `deck_id` | string | yes | `summary.deckId`; uploader seat only, null on the other |

The uploader's archetype comes from `myArchetype` and from nothing else; the
opponent's comes from `opponentArchetype`. A manual game with no archetype row
for the opponent falls back to the name the uploader typed, which bronze has
already replaced with a token, so the stranger rule in 8.5 applies to it like
any other token.

`deck_name` and `deck_id` are the uploader's own deck record, kept for
player-level views and never used as an archetype label. A deck name is a
nickname typed into the game client: on the current corpus it is the client's
default, `New Deck 54`, on 61 of 128 games and a joke or a shorthand on most of
the rest, so reading one as an archetype would fill the matchup mart with labels
that name no deck. The consequence is visible in the data rather than hidden:
an uploaded game carries an uploader archetype only when the application derived
one or the user set one, so most uploader seats have `archetype_name` and
`archetype_source` null and the marts, which drop seats with no archetype, leave
them out.

### 8.3 `turns`

One row per `turn` segment.

| Column | Type | Nullable | From |
|---|---|---|---|
| `game_id`, `play_date` | string, date | no | `games` |
| `turn_number` | int | yes | `segment.turnNumber` |
| `seat` | int | yes | seat of `segment.player`, null when the header named nobody |
| `n_entries` | int | no | every action line in the segment, entries and sub-entries |
| `n_draw` | int | no | `opening_hand`, `draw`, `draw_named`, `mulligan_draw` |
| `n_attach` | int | no | `attach` |
| `n_attack` | int | no | `attack` |
| `n_play_pokemon` | int | no | `play_pokemon` |
| `n_play_trainer` | int | no | `play_card`, `play_stadium` |
| `n_evolve` | int | no | `evolve` |
| `n_retreat` | int | no | `retreat` |
| `n_knockout` | int | no | `knockout` |
| `n_prize_taken` | int | no | `prize` |
| `concession` | boolean | no | a `concede` line in this segment |

The buckets follow the producer's own `deriveStats` definitions (section 5), so
a counter summed over a game matches the side counter it came from. `drawn_cards`
is deliberately in no bucket: it is the sub-entry that lists what a `draw` drew,
so counting it would count the same draw twice. A kind in no bucket still counts
toward `n_entries`. A manual game has no segments and therefore no turn rows.

### 8.4 `cards_seen`

One row per (game, seat, card) from `summary.observedCards`, deduplicated.

| Column | Type | Nullable | From |
|---|---|---|---|
| `game_id`, `play_date` | string, date | no | `games` |
| `seat` | int | no | `mySide` for `observedCards.me`, the other seat for `.opponent` |
| `card_id` | string | no | the reference's `cardId`, else `baseCardId`, else its lowercased `name` |
| `base_card_id`, `card_name`, `set_code`, `number` | string | yes | the reference as it arrived |
| `count_seen` | int | no | `CardRef.count` |
| `in_decklist` | boolean | yes | the card is in that seat's decklist; null when the seat shared none |
| `catalog_name`, `catalog_set`, `catalog_type`, `catalog_hp`, `catalog_reg` | string, string, string, int, string | yes | `catalog/cards.json`, joined LEFT on `card_id` |

Why `card_id` is not simply `cardId`: `observedCards` never carries one, in a
stock or a debug export, because it is derived from the battle log and the log
prints names. A decklist reference is the mirror image, card ids and no names.
The coalesce above is the identity silver can actually resolve, it is never
null, which is what makes the grain a grain, and it matches a catalog key
whenever the export gave a real client card id. For `in_decklist` the catalog
bridges the two key spaces: a decklist entry contributes its card id and, when
the catalog knows that id, the lowercased catalog name. Without the catalog
there is no bridge, so an observed card keyed by name never matches a decklist
keyed by id and `in_decklist` reads false for every row of a seat that shared a
list; fetch the catalog before reading that column. A game with no `mySide`
produces no rows, because neither list can be attached to a seat.

A cards-seen row is a lower bound on a decklist, never a decklist. Section 3 of
[data-handling.md](data-handling.md) says what may be published from it.

### 8.5 Alias resolution and the stranger rule

`archetype_name` is resolved through an alias map built from bronze itself: for
each `archetype_id`, the canonical name is the one the most recently ingested
game gives it, because a rename upstream reaches a game only when that game is
rewritten and older rows keep the old label. `archetype_name_raw` keeps what the
row arrived with.

`player_token` is NULL unless the token holds an uploader seat somewhere in
bronze. Opponents who never uploaded a game never saw the in-app notice, so
their tokens are not carried into silver even in pseudonymous form;
`is_member` records the distinction, and a manual game's typed-in opponent name
goes through the same rule.

## 9. Gold tables

One DuckDB file, `$PIPELINE_DATA_DIR/warehouse/meta.duckdb`, built by
`python -m pipeline.gold` from the dbt project in `dbt/`. The silver Parquet
files are read in place through `read_parquet(...)` sources, so nothing is
loaded and the warehouse can be deleted and rebuilt at any time. Staging models
(`stg_games`, `stg_game_sides`, `stg_turns`, `stg_cards_seen`) are views over
those sources and carry no columns of their own beyond two surrogate keys; the
tables below are the ones a reader queries. Every column is described in
`dbt/models/marts/schema.yml`, which is also what `dbt docs generate` renders.

### 9.1 Grain and keys

| Table | Grain | Key | Foreign keys |
|---|---|---|---|
| `fct_game_side` | one row per (game, seat), two per game | `game_side_key` = `game_id` + `-` + `seat` | `player_key`, `archetype_key`, `opponent_archetype_key`, `season_key`, `format_key`, `date_key` |
| `dim_player` | one row per member token | `player_key` | none |
| `dim_archetype` | one row per archetype | `archetype_key` | none |
| `dim_season` | one row per season, plus `unknown` | `season_key` | none |
| `dim_format` | one row per export variant, plus `unknown` | `format_key` | none |
| `dim_card` | one row per card observed | `card_key` | none |
| `dim_date` | one row per play date with a game | `date_key` (the date) | none |
| `mart_matchups` | one row per ordered archetype pair | `matchup_key` | `archetype_key`, `opponent_archetype_key` |
| `mart_archetype_weekly` | one row per (archetype, ISO week) | `archetype_week_key` | `archetype_key` |
| `mart_cards_seen` | one row per (archetype, card) | `archetype_card_key` | `archetype_key`, `card_key` |
| `mart_player_summary` | one row per member token | `player_key` | `player_key` |

Surrogate keys are the natural key, not a hash: `archetype_key` is the shared
archetype row's identifier when a game carries one and the canonical name
lowercased and prefixed `name:` when it does not, `player_key` is silver's
keyed token, `card_key` is silver's `card_id`, and `date_key` is the date
itself. Nothing is gained by hashing a key that is already short, stable and
readable, and a readable key makes a failing test legible.

### 9.2 What the fact carries

Identity and keys (`game_side_key`, `game_id`, `seat`, `is_uploader`, and the
six foreign keys above), the outcome (`result_for_seat`, `is_win`, `is_loss`,
`is_tie`, `went_first`), the measures (`turn_count`, `prizes_taken`,
`knockouts`, `cards_drawn`, `energy_attached`, `damage_dealt`, `mulligans`,
`turns_taken`, `decklist_complete`, `decklist_card_count`), and the flags the
marts filter on (`has_full_decklists`, `export_variant`,
`excluded_from_stats`).

`player_key` is NULL for a stranger, which is the stranger rule (8.5) arriving
in gold unchanged: there is no token to key on, so there is no row in
`dim_player` and no way to follow that person across games. `archetype_key` and
`opponent_archetype_key` are NULL when the game named no archetype for that
seat; the marts drop those rows rather than bucketing them as "unknown
archetype", because an unnamed deck is not a deck. `season_key` and
`format_key` are never NULL: they point at the synthetic `unknown` row of their
dimension instead, so an inner join loses nothing.

Nothing is filtered out of the fact. `excluded_from_stats` rows are present and
flagged, and every mart drops them, for the same reason silver keeps what
bronze landed: a layer that has already dropped a row cannot count it.

### 9.3 What the marts measure, and what they do not

`win_rate`, everywhere it appears, is wins over wins plus losses. Ties and
unresolved results are left out of the denominator rather than counted as half
a win: a tie only happens on a hand-logged game and `unknown` means the seat
could not be resolved, so neither is evidence. The rate is NULL when nothing
was decided.

`mart_matchups` is symmetric because of the grain, not because of a union. One
game contributes one fact row per seat, each carrying its own archetype and the
other seat's, so it is counted once as (A, B) and once as (B, A) and either
lookup finds a row. A mirror, (A, A), is the exception worth knowing: both
seats produce the same pair, so the row counts each mirror game twice and its
wins equal its losses by definition.

`mart_archetype_weekly` holds two counts that must not be confused. `games`
counts seat rows, every appearance of the archetype on either side of the
table. `week_games` counts games once each, from the uploader seats, because
exactly one seat per game is the uploader. `share_of_week` is the first over
the second, so it reads as the share of the week's games the archetype was one
of the two decks in, and the column sums to roughly two across a week.

`mart_cards_seen` is the one place where the limit in 8.4 becomes a number, so
it is labelled twice. `seen_rate` is the share of games in which the card was
observed being played or revealed, and it is not a deck inclusion rate: a stock
export only reveals played cards. `inclusion_rate` sits beside it, over the
seats that shared a full decklist in game, and it is still bounded by
observation because silver exposes no row per decklist card. Both are lower
bounds, `inclusion_rate` the tighter one, and nothing averages them together.

## 10. Quirks and the column that carries each

| Quirk | Where it lands |
|---|---|
| Cards seen are a lower bound on the deck | `cards_seen.in_decklist`, null unless that seat shared a list; marts label "cards seen rate" |
| `(clientId) Name` prefixes | stripped in `fields`, still present in `text` |
| Unparsed lines | `games.unparsed_count`, bronze `unparsed_lines`, entries of kind `other` |
| `mySide` may be null | `games.my_side` null, no seat is the uploader, `games.result` `unknown`, no `cards_seen` rows |
| Excluded games | `games.excluded_from_stats`, kept in silver and filtered in every gold model |
| Handle changes | identity is `user_id`; the token is per handle, so one account can map to several tokens |
| Manual games | landed and given two `game_sides` rows, with null counters, no turns and no cards seen |
| `observedCards` has no card ids | `cards_seen.card_id` falls back to the lowercased name (section 8.4) |
| A decklist has no card names | `in_decklist` matches through the catalog (section 8.4) |
| `playedAt` defaults to upload time | `play_date_source` for v1; v2 rows carry `played_at` and `uploaded_at` so the gap is measurable |
