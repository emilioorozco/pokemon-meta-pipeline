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
blobs are written; a v1 blob is upgraded to v2 or quarantined before it reaches
the writer, so there is no S3-last-modified fallback in the table.

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
| `my_decklist`, `opponent_decklist` | struct per section 6 | yes | blob |

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

One JSON line per rejected object under
`data/lake/quarantine/bronze/run_date=YYYY-MM-DD/`. Nothing from a quarantined
blob is written to the three tables.

| Field | Type | Meaning |
|---|---|---|
| `source_key`, `source_version_id` | string | Which object failed. |
| `user_id`, `game_id` | string | Parsed from the key, so a record exists even when the body is unreadable. |
| `schema_version_seen` | int or null | `schemaVersion` if the body parsed. |
| `reason_code` | string | One of `json_parse_error`, `schema_invalid`, `players_lt_2`, `played_at_missing`, `handle_leak_check_failed`, `unsupported_schema_version`. |
| `reason_detail` | string | Validator message or the failing path. |
| `body_sha256` | string | For deduplicating repeat failures across runs. |
| `observed_at` | timestamp | Run time. |

## 8. Quirks and the column that carries each

| Quirk | Where it lands |
|---|---|
| Cards seen are a lower bound on the deck | `game_seat.observed_cards` versus `decklist_cards`; marts label "cards seen rate" |
| `(clientId) Name` prefixes | stripped in `fields`, still present in `text` |
| Unparsed lines | `game.unparsed_count`, `game.unparsed_lines`, `game_event.kind = 'other'` |
| `mySide` may be null | `game.my_side` null, `game_seat.is_owner` null, `game.result` `unknown` |
| Excluded games | `game.excluded_from_stats`, filtered in every silver and gold model |
| Handle changes | identity is `user_id`; `player_hash` is per handle, so one account can map to several hashes |
| Manual games | not present in `parsed/`, therefore not in bronze today |
| `playedAt` defaults to upload time | `play_date_source` for v1; v2 rows carry `played_at` and `uploaded_at` so the gap is measurable |
