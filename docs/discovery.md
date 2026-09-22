# Discovery: what the analytics source actually contains

Profiled 2026-09-21, before writing any ingest code against this source. Rule:
never design a schema before looking at real records and the code that writes
them. Everything below comes from reading the upstream producer's source (paths
given as `play-rough-analytics/<path>`) and its test fixtures. Player handles
are never quoted; where an example needs one, `PlayerA` and `PlayerB` are used.

## The upstream producer in one paragraph

Play Rough Analytics is a web application (TypeScript monorepo, AWS Lambda API,
React front end) where players upload battle-log exports from the Pokemon
Trading Card Game (TCG) Live client. On upload the API splits the export, parses
the battle log into structured segments and entries, derives a per-game summary
row, and persists three things: the original text, the parsed blob, and the
summary. It is the system of record for the games; this pipeline is a downstream
consumer.

## Where the data lives

One private, versioned Amazon Simple Storage Service (S3) bucket per deployed
environment (`dev` and `prod`; construct `RawLogsBucket` in
`play-rough-analytics/infra/lib/constructs/raw-logs-bucket.ts`, name exported as
`RawBucketName`). Key layout, from
`play-rough-analytics/apps/api/src/store/types.ts` (`blobKeys`):

| Prefix | Content | Written by |
|---|---|---|
| `raw/{userId}/{gameId}.txt` | The uploaded export text, verbatim (`text/plain`) | upload |
| `parsed/{userId}/{gameId}.json` | The parsed blob (`application/json`), see below | upload, decklist attach, admin re-parse |
| `decklists/{userId}/{gameId}/{me,opponent}.txt` | A decklist the owner pasted after the fact | decklist attach |
| `catalog/cards.json` | Slim card catalog keyed by client card id | site deploy |

The per-game summary row is not in S3 today. It lives in DynamoDB (one table
per environment) and is what the API returns as `GameSummary`.

`gameId` for an uploaded game is the first 16 hex characters of the SHA-256 of
the normalized battle-log text (`gameIdFor` in
`play-rough-analytics/apps/api/src/games/service.ts`), so re-uploading the same
log is a no-op and the id is stable across environments. `userId` is the
account id from the authentication provider, not a player handle.

## Blob v1: the shape observed today

`ParsedBlob` (`play-rough-analytics/apps/api/src/games/service.ts`, built by
`parseExportText`):

```
{
  segments:         Segment[]                  // setup / turn / checkup blocks with parsed entries
  statsByPlayer:    { [handle]: SideStats }     // per-side counters, keyed by in-game handle
  unparsedLines:    string[]                    // lines no pattern matched
  extras:           { [tag]: string[] }         // unknown "[Tag] ..." lines from preamble and trailer
  myDecklist?:      Decklist                    // only when a debug preamble or a paste supplied it
  opponentDecklist?: Decklist
}
```

The segment and entry shapes come from
`play-rough-analytics/packages/shared/src/battle-log/parser.ts`; the field
tables are in [schema.md](schema.md). The blob has no version marker, no game
id, no timestamp, no player list and no result: everything a consumer needs to
place a blob in time or attribute it to a seat is in the DynamoDB summary.

Three producers rewrite an existing blob in place: the upload itself, the
"attach decklist" action (adds `myDecklist` or `opponentDecklist` with
`source: "paste"`, `service.ts` `attachDecklist`), and the admin re-parse
(`play-rough-analytics/apps/api/src/games/reparse.ts`), which re-runs the
current parser over `raw/` and overwrites `parsed/` while carrying pasted
decklists forward. Blobs are therefore mutable, and the bucket's object
versioning is the only history.

## Export variants

`exportVariant` on the summary is `stock`, `debug` or `manual`
(`exportVariantSchema` in `play-rough-analytics/packages/shared/src/contracts/index.ts`).

- `stock`: the unmodified client's clipboard export. Only the battle log. The
  parser discovers player handles from turn headers, and the only decklist
  information is what each side revealed by playing, attaching, evolving or
  discarding cards ("cards seen").
- `debug`: an export from a modified client. `split.ts` classifies an export as
  `debug` when it finds the `=== Battle Log ===` marker or any trailing
  `[Tag] ...` lines. A `=== Match Start ===` preamble (parsed by `preamble.ts`)
  can carry both full decklists as client card ids, deck metadata (deck name,
  size, sleeve, coin), inferred prizes and prizes taken. A `[CompetitiveElo]`
  trailer line carries the season and the rating change. Note the asymmetry: a
  stock-looking log with only a trailer is still `debug`, so `debug` does not
  imply full decklists; the summary's `hasFullDecklists` does.
- `manual`: a game the owner typed in (result, two archetypes, optional
  tournament and round). Created in
  `play-rough-analytics/apps/api/src/manual/service.ts` with `parserVersion: 0`,
  `turnCount: 0`, `stats: {}` and no S3 objects at all, neither `raw/` nor
  `parsed/`. Manual games can also end in a `tie`.

## What is missing in v1, and why v2 exists

A pipeline that reads only `parsed/` gets segments and counters but cannot
answer "when was this played, by whom, who won, was it excluded" without a
second read against DynamoDB. That couples the pipeline to the application's
operational store and to its item shape. The project's decision is to make the
blob self-describing instead:

- Contract v2 (in progress upstream): the blob gains `schemaVersion: 2` and an
  embedded `summary` equal to `gameSummarySchema`. The API validates with Zod on
  write and exports the same schema as JSON Schema so this repository can
  contract-test its reader against it without porting any TypeScript.
- Readers accept v1 (no `schemaVersion`, no `summary`) and v2. A v1 blob has no
  `playedAt`, so its partition date falls back to the S3 object's last-modified
  time and the row is flagged (`play_date_source = "s3_last_modified"`). The
  intended cleanup is an upstream re-parse that rewrites every blob as v2; the
  fallback exists so a bronze build never blocks on that.
- Manual games had no blob under v1. Since contract v2 the application writes
  a summary-only blob for them (`segments: []`, `statsByPlayer: {}`, the row as
  `summary`) on create and on every edit, so the pipeline ingests every game.
  Marts must still treat them as result-only records: no turns, no cards seen.

## Counts

Volume today is small and stated plainly: on the order of a hundred-plus games
from a single-digit number of players, one bucket per environment. Nothing
about the design depends on that number. The scale story is date partitioning
plus a Spark job that runs identically on a laptop and on a cluster; the counts
will be re-recorded once the bronze build runs against the bucket.

## Quirks recorded from the code

Each of these has a consequence in [schema.md](schema.md).

1. Cards seen are not cards in the deck. For stock exports `observedCards`
   (`observedCards()` in `parser.ts`) lists only cards a side made public,
   with counts capped at four for non-basic-energy cards. Card inclusion rates
   over stock games are lower bounds.
2. Some clients print `(clientId) Name` in front of a card name. The parser
   strips it (`cleanCardName` in
   `play-rough-analytics/packages/shared/src/battle-log/card-ref.ts`) in the
   name-bearing `fields` of an entry, but the raw entry `text` still contains it.
3. `unparsedLines` is populated whenever a line matches no pattern; the summary
   tracks `unparsedCount` and `parserVersion`. A parser upgrade plus re-parse
   changes both for old games.
4. `mySide` may be `null` when neither the preamble, a concede line, named
   draws nor a configured handle identified the local player. Such a game has
   no `result`, `opponentName`, `stats` or `observedCards`.
5. `excludedFromStats` keeps a game visible in the app but out of its analytics
   and deck records. Every mart must filter it.
6. Handles change. The same account can appear under different in-game
   handles across games (`tcglHandles` on the profile is a list). Player
   identity for analytics is the account (`userId`), not the handle.
7. Manual games have no segments, a summary-only blob, `parserVersion: 0` and
   may be ties.
8. `playedAt` defaults to `uploadedAt` when the uploader sends no timestamp,
   so `play_date` is an upload date for part of the corpus. The debug client
   and the phone shortcut send one; browser uploads may not.
9. Handles appear in free text everywhere: segment titles (`PlayerA's Turn`),
   entry `text`, `actor`, several `fields` (`targetOwner`, `owner`, `winner`,
   `loser`), `statsByPlayer` keys, `unparsedLines`, `extras`, preamble deck
   headers, and in the summary's `players`, `winner`, `opponentName`.
   Anonymization must be a rewrite over all string values.
10. `extras` collects every unknown `[Tag]` line except `[CompetitiveElo]`,
    which is parsed into the summary's `elo` instead. A v1 blob carries no
    rating information at all.

## Decisions this drives

1. Ingest source is the parsed JSON blobs under `parsed/`, not the raw text.
   The application is the upstream producer and its Zod schema is the data
   contract. `raw/` stays as the replay source: an upstream re-parse plus a
   bronze rebuild is the full-recompute path. No TypeScript parser is ported.
2. Contract v2 as above; v1 documented as current, v2 as target.
3. Bronze is the validated blob flattened into `game`, `game_seat` and
   `game_event`, Hive-partitioned by `play_date` from `summary.playedAt`,
   written by idempotent partition rewrite, with failures quarantined with a
   reason code.
4. Privacy: handles are HMAC-anonymized at bronze with a per-environment key;
   the key never enters the lake. Only `stock`-variant games without full
   decklists appear in anything public. Games that carry a full decklist
   (`hasFullDecklists`, or any `myDecklist` / `opponentDecklist` in the blob)
   never go into test fixtures.
5. Natural key: `(user_id, game_id)`. Seat index 0 or 1 (position in
   `players`) distinguishes the two rows per game; `my_side` marks the owner's
   seat.
6. The card dimension comes from `catalog/cards.json` (client card id to
   `{name, set, number, type?, hp?, reg?}`, see
   `play-rough-analytics/packages/shared/src/catalog/index.ts`). Name-only
   references from stock logs join through a lowercased name index, the same
   rule the application uses.
7. Archetype is a derived dimension. The application's rule (top two Pokemon
   by play and evolution counts, bonus for `ex` and `Mega`) produces
   `opponentArchetype`; admins merge and rename rows, leaving alias tombstones
   (`mergedInto` on `archetypeSchema`). Silver resolves names through those
   aliases so a merge upstream is reflected downstream.
