# Stages

Each stage is one task in the orchestrated DAG (directed acyclic graph) and one
command that can be run by hand. Inputs and outputs are files or tables; no
stage reaches back into a previous stage's internals.

```
S3 parsed/{userId}/{gameId}.json  (contract v1 today, v2 target)
        |
        v
[1 bronze]  validate, anonymize, flatten -> game / game_seat / game_event Parquet
        |                                    (+ quarantine)
        v
[2 silver]  PySpark: typed, cards exploded, catalog + archetype aliases joined, features
        |
        v
[3 gold]    dbt on DuckDB: fact_game_seat + dims, marts
        |
        +--> [4 model]   MLflow-tracked win-probability model
        |         |
        |         v
        |    [5 serve]   FastAPI: predictions over the gold tables
        |
        +--> [6 agent]   LangChain agent over the marts and card text
        |
        v
[7 orchestrate]  Airflow DAG for 1-3, model retrain on schedule
        |
        v
[8 publish]  results written back to the application (S3 prefix it reads)
```

Status legend: done, in progress, planned.

## 1. Bronze ingest (in progress)

Input: every object under `parsed/` in the environment's bucket (name from
the `RawBucketName` stack output), plus the anonymization key from the
environment. Reads are by key listing plus per-object version id; a run can be
limited to a date range or a list of keys.

Steps per object:

1. Parse JSON. Failure: quarantine `json_parse_error`.
2. Detect version: `schemaVersion` present and equal to 2, or absent (v1).
   Anything else: quarantine `unsupported_schema_version`.
3. Validate against the exported JSON Schema for that version. Failure:
   quarantine `schema_invalid` with the failing path.
4. Resolve `play_date`: `summary.playedAt` for v2; S3 last-modified for v1,
   flagged in `play_date_source`.
5. Collect handles: `summary.players` (v2) or `statsByPlayer` keys (v1), plus
   `Segment.player`. Fewer than two: quarantine `players_lt_2`.
6. Anonymize: replace each handle with its HMAC token in every string value of
   the blob (titles, `text`, `actor`, `fields`, `statsByPlayer` keys,
   `unparsedLines`, `extras`, summary fields). Then scan the rewritten blob for
   any remaining raw handle; a hit quarantines `handle_leak_check_failed`
   rather than writing.
7. Flatten into `game`, `game_seat`, `game_event` rows as defined in
   [schema.md](schema.md).

Output: the three bronze tables, partitioned by `play_date`, each touched
partition deleted and rewritten in full (idempotent); quarantine records as
JSON lines. Quality checks that fail the task: exactly two seat rows per game,
at most one `is_winner` per game, no raw handle in any string column.

Why this shape: the blob is already parsed, so bronze is a contract check and
a flatten, not a parser. Keeping bronze close to the source names means a
contract change upstream shows up as a validation failure here, not as a
silently wrong column.

At 1000x: the per-object Python loop becomes the bottleneck. The listing
becomes an S3 inventory report, the loop becomes a Spark job reading the same
JSON, and the logic (validate, anonymize, flatten, partitioned write) does not
change.

## 2. Silver (planned, PySpark)

Input: bronze tables, `catalog/cards.json`, the archetype table and its alias
tombstones exported from the application.

Output tables:

- `silver.game_seat`: typed, `excluded_from_stats` rows dropped, `is_owner`
  and `is_winner` resolved, archetype name resolved through aliases to a
  canonical `archetype_key`, season attached.
- `silver.seat_card`: one row per (game, seat, card) from `observed_cards`
  and, where present, `decklist_cards`, with a `card_source` of `observed` or
  `decklist`, joined to the catalog by `card_id` or lowercased name.
- `silver.game_event`: typed events with `turn_number`, `actor_seat`, and the
  numeric `fields` (`n`, `damage`) promoted to columns.
- Derived features per seat: prizes taken by turn, first knockout turn, energy
  attached by turn 3, distinct attackers, went first.

Why Spark: the joins and explodes are where data grows (events are the largest
table by far, cards per seat multiply rows). The same job runs on a laptop
in local mode and on a cluster unchanged.

## 3. Gold (planned, dbt on DuckDB)

Input: silver Parquet, read in place by DuckDB.

Star schema at game-by-seat grain:

- `fact_game_seat`: one row per (game, seat): `game_key`, `seat`, `player_key`,
  `archetype_key`, `opponent_archetype_key` (denormalized so a matchup is one
  group-by), `season_key`, `play_date`, `is_winner`, `went_first`,
  `turn_count`, the seven counters, `export_variant`, `has_full_decklists`.
- `dim_player`: `player_key` from `user_id`; the handle hashes it has appeared
  under; first and last seen dates. No handle text.
- `dim_archetype`: canonical name, alias names it absorbed, flagship Pokemon.
- `dim_season`: `season_id`, name, format when known, first and last game.
- `dim_card`: from the catalog: id, base id, name, set, number, type, HP,
  regulation mark, category (pokemon, trainer, energy).
- `bridge_seat_card`: (game, seat, card, count, card_source) for card-level
  marts.

Marts:

- `mart_archetype_winrates`: games, wins, win rate, share of games, by
  archetype and season, with a minimum-games floor.
- `mart_matchup_matrix`: archetype by archetype: games, win rate, symmetric by
  construction (wr(A, B) + wr(B, A) = 1 is a test).
- `mart_cards_seen`: for each card and archetype: share of seats where the card
  was seen, average copies seen, split by `card_source` so decklist-backed
  numbers are never mixed with observed lower bounds.

dbt tests: unique and not-null keys, relationships to dimensions, accepted
values on enums, two rows per game, no excluded games, win rates in [0, 1].

Only `stock`-variant games without full decklists feed anything that is
published outside the environment; the filter is a dbt variable, not a manual
step.

## 4. Win-probability model (planned, MLflow)

Input: `fact_game_seat` plus the silver per-seat features. Target:
`is_winner`. Baseline: logistic regression on archetype pair, went first and
season; then gradient boosting with the turn-indexed features. Every run is
tracked in MLflow (parameters, metrics, the exact gold snapshot date), and the
promoted model is registered with the training data's `play_date` range.
Honest framing: with a hundred-plus games the model is a pipeline exercise;
its reported metric is cross-validated and the sample size is printed next to
it.

## 5. Serving (planned, FastAPI)

A small FastAPI service that loads the registered model and answers
`POST /predict` with archetype pair and optional early-game features, and
`GET /matchups/{archetype}` from the gold marts. It reads DuckDB read-only and
never touches bronze.

## 6. Agent (planned, LangChain)

A LangChain agent with two tools: SQL over the gold marts (read-only DuckDB
connection, schema-limited to `gold`) and retrieval over card text from the
catalog. It answers questions such as "which archetype has the best record
against X this season" by writing and running the mart query and citing the
row counts.

## 7. Orchestration (planned, Airflow)

Airflow standalone (single local process, SQLite metadata database) running
`bronze >> silver >> gold >> quality_gate` daily, with a weekly `retrain`
task downstream of `gold`. Each task is the stage's command-line entry point,
so any other scheduler could call the same commands. Idempotent partitions make
retries safe.

## 8. Publish back (planned)

Gold marts and the model's per-archetype summaries are written as JSON under a
prefix the application reads (in the same environment's bucket), so the web
application can show community-level matchup and win-rate views without
querying the warehouse. Only the public-safe subset (section 3) is published.

## Open items

- Manual games: `exportVariant: "manual"` games have no blob in `parsed/`. The
  pipeline cannot see them until upstream either writes a summary-only v2 blob
  for them or the project decides they stay out of scope. Until then, marts
  state "uploaded games only".
- v1 backfill: an upstream admin re-parse rewrites every blob as v2 and
  removes the `s3_last_modified` fallback from the data. Bronze is designed to
  run before and after that without code changes.
- Archetype and alias export: silver needs the application's archetype rows
  (with `mergedInto` tombstones). The export format (a JSON object under the
  bucket, or an API call) is not decided.

## History

The first version of this pipeline (2026-08-31) ingested a public corpus of
about 1,240 AI-versus-AI replays from a finished competition. Stage 1 of that
version was built and run: it extracted per-game facts, both 60-card decklists
and per-turn engine events into three bronze Parquet tables partitioned by
`play_date`, enriched with timestamps and ratings recovered from the
competition's episode-metadata endpoint, and quarantined four corrupt games.
Two lessons carried over unchanged: land the whole upstream response before
parsing it (the original corpus pull had discarded timestamps it already had),
and quality checks catch the pipeline author's own bugs at least as often as the
source's (a `firstPlayer = -1` sentinel produced `went_first = false` for both
seats until an "exactly one seat went first" check exposed it). The code for
that stage is deprecated and kept under `pipeline/legacy/kaggle/`; see
[adr/0001-deprecate-kaggle-source.md](adr/0001-deprecate-kaggle-source.md).
