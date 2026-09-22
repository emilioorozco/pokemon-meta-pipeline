# Stages

Each stage is one task in the orchestrated DAG (directed acyclic graph) and one
command that can be run by hand. Inputs and outputs are files or tables; no
stage reaches back into a previous stage's internals.

```
S3 parsed/{userId}/{gameId}.json  (contract v1 today, v2 target)
        |
        v
[1 bronze]  validate, anonymize, land one nested row per game as Parquet
        |                                    (+ quarantine)
        v
[2 silver]  PySpark: typed, cards exploded, catalog + archetype aliases joined, features
        |
        v
[3 gold]    dbt on DuckDB: fct_game_side + dims, marts
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
environment. Reads are by paginated key listing plus the version id the read
itself reports; a run can be limited to the first N objects.

Command: `python -m pipeline.backfill` (`--limit N`, `--dry-run`).

The same command also reads a local directory of blobs with `--source-dir PATH` (with `--bronze-dir` and `--quarantine-dir` to send the output elsewhere), which needs no bucket and no credentials and is how the committed fixtures are ingested for demos ([demo.md](demo.md)); everything after the read is identical.

Steps per object:

1. Parse JSON. Failure: quarantine `invalid_json`.
2. Validate against the contract models, which dispatch on `schemaVersion`: a
   version other than 2 fails as a v2 blob and names that key. Failure:
   quarantine `contract_violation` with the failing paths.
3. A blob with no `schemaVersion` is v1, the pre-contract shape. It has no
   `summary` and therefore no play date, and the only substitute is the
   object's S3 last-modified time, which is the upload time and not the play
   time. Rather than guess a partition, quarantine `v1_blob` with a hint to
   re-parse it upstream to v2. Such a blob is not anonymized and not written.
4. Every valid v2 game lands, blob untouched, including the ones a modified
   client exported with both complete decklists. An opponent's list is in the
   blob only because the opponent shared it in-game
   ([data-handling.md](data-handling.md)), so `hasFullDecklists` is
   informational: the run counts those games and routes nothing on the flag.
5. `play_date` is `summary.playedAt`, so `play_date_source` is always
   `summary` while only v2 blobs are written.
6. Collect handles: `summary.players`, `winner`, `opponentName`,
   `statsByPlayer` keys and `Segment.player`.
7. Anonymize: replace each handle with its HMAC token in every string value of
   the blob (titles, `text`, `actor`, `fields`, `statsByPlayer` keys,
   `unparsedLines`, `extras`, summary fields), then re-validate the rewrite
   against the contract, which proves it changed strings and not shape.
   Failure: quarantine `contract_violation` with an `after anonymization:`
   prefix.
8. Land one row per game with the blob kept nested (summary struct, segments
   list, decklists) as defined in [schema.md](schema.md); seat and event grains
   are produced in silver.

The write happens once, at the end of the run, because a partition write
replaces the whole day. Before it, the batch is scanned for any handle that
survived the rewrite; a hit is re-checked per game, and the games that leak are
quarantined `handle_leak_check_failed` while the rest are written. One
unrewritable handle costs its own game, not the run.

Output: bronze partitioned by `play_date`, each touched partition deleted and
rewritten in full (idempotent); quarantined objects kept as received with a
handle-free sidecar. The run prints read, landed, quarantined by reason, how
many landed games carry full decklists, and the rows per partition. Quality
checks that fail the task: exactly two seat rows per game, at most one
`is_winner` per game, no raw handle in any string column.

Why this shape: the blob is already parsed, so bronze is a contract check and
a flatten, not a parser. Keeping bronze close to the source names means a
contract change upstream shows up as a validation failure here, not as a
silently wrong column.

At 1000x: the per-object Python loop becomes the bottleneck. The listing
becomes an S3 inventory report, the loop becomes a Spark job reading the same
JSON, and the logic (validate, anonymize, flatten, partitioned write) does not
change.

## 2. Silver (in progress, PySpark)

Input: the bronze Parquet tables and, optionally, the card catalog
(`data/catalog/cards.json`, fetched by `scripts/fetch_catalog.py`). No archetype
export is needed: the alias map is built from bronze itself, see below.

Command: `python -m pipeline.silver` (`--bronze-dir`, `--silver-dir`,
`--catalog`, `--master`). It needs a Java Virtual Machine (JVM); everything else
is the `spark` extra.

Bronze is one nested row per game. Silver is the grain change, four tables under
`data/lake/silver/<table>/play_date=YYYY-MM-DD/`:

- `games`: one row per game. Identity and lineage (`game_id`, `user_id`,
  `play_date`, `played_at`, `source_key`, `ingested_at`, `contract_version`),
  the export's own description of itself (`export_variant`, `upload_source`,
  `parser_version`, `unparsed_count`, `played_at_source`, `has_full_decklists`,
  `excluded_from_stats`, `season_id`, `season_name`), and the outcome with every
  handle already resolved to a seat: `result` (the uploader's), `winner_seat`,
  `went_first_seat`, `coin_toss_winner_seat`, `first_player`, `my_side`,
  `turn_count`, `end_reason`.
- `game_sides`: two rows per game, seat 0 and seat 1. `is_uploader`,
  `player_token`, `is_member`, the archetype columns (`archetype_id`,
  `archetype_name`, `archetype_name_raw`, `archetype_source`),
  `result_for_seat`, `went_first`, one `stats_*` column per `SideStats` counter,
  the seat's decklist facts (`decklist_source`, `decklist_complete`,
  `decklist_card_count`) and, on the uploader seat only, that player's own deck
  record (`deck_name`, `deck_id`). This is the grain the gold fact table is
  built on.
- `turns`: one row per turn segment. `turn_number`, `seat`, `n_entries` and nine
  counters by action kind (`n_draw`, `n_attach`, `n_attack`, `n_play_pokemon`,
  `n_play_trainer`, `n_evolve`, `n_retreat`, `n_knockout`, `n_prize_taken`),
  plus `concession`. Every action line in the segment is counted, the top-level
  entries and the sub-entries under them alike, because the draw a Professor's
  Research causes is printed as a sub-entry of the line that played it. A kind
  with no counter still lands in `n_entries`. No `fields_json` is parsed here.
- `cards_seen`: one row per (game, seat, card) from `summary.observedCards`,
  left joined to the card catalog.

Three rules are worth stating on their own.

**Archetype aliases.** An archetype is renamed upstream by editing one shared
row, and the rename reaches a game only the next time that game is written, so
older bronze rows keep the old label forever. Silver therefore builds the alias
map out of bronze: for each `archetype_id`, the canonical name is the one the
most recently ingested game gives it, and every other game carrying that id
inherits it. The label the row actually arrived with is kept as
`archetype_name_raw`, so the rename is visible rather than erased. Ties inside a
run break on play date and then game id, so the map is the same on every rerun.
Deck names are the uploader's own nicknames and are never used as archetype
labels; uploaded games get an uploader archetype only when the application
derived or the user set one.

**Strangers.** `member_tokens` is the set of player tokens that hold an uploader
seat somewhere in bronze. Every other token belongs to somebody who was matched
against a member and never uploaded anything, so they never saw the in-app
notice and never consented to anything. Those tokens are written as NULL in
`game_sides.player_token` and anywhere else a token could reach a column, and
`is_member` records which is which. See [data-handling.md](data-handling.md).

**Reconciliation.** After the write the run asserts `games_in == games_out`,
`game_sides == 2 * games` and no duplicate `(game_id, seat, card_id)` in
`cards_seen`, prints each check and the per-table row counts, and exits non-zero
on a failure. The tables are on disk either way: a run that dropped half the
games is worse silent than loud.

Two shapes of the real data decide how cards are keyed, and both are the
opposite of what the field names suggest. `observedCards` never carries a
`cardId`, in a stock or a debug export, because it is derived from the battle
log and the log prints names. A decklist reference is the mirror image: card ids
and no names. So `cards_seen.card_id` is the identity silver can actually
resolve, the reference's `cardId` when it has one, else its `baseCardId`, else
its lowercased name, which keeps the grain non-null and lets the catalog join
hit whenever the key is a real client card id. For `in_decklist` the catalog is
the bridge between the two key spaces: a decklist entry contributes its card id
and, when the catalog knows that id, the lowercased catalog name. With no
catalog the run still works, with null catalog columns and id-to-id matching,
but `in_decklist` then reads false everywhere and is not worth querying until
the catalog has been fetched.

Why Spark: the joins and explodes are where data grows (cards per seat and turns
per game multiply rows). The same job runs on a laptop in local mode and on a
cluster unchanged.

At 1000x, what changes is configuration, not code: the master URL (`--master`
or `PRA_SPARK_MASTER`), the input and output paths becoming `s3a://`, and
`spark.sql.shuffle.partitions`, which is 8 here because a laptop run with the
default 200 spends more time on empty tasks than on work.

## 3. Gold (in progress, dbt on DuckDB)

Input: the silver Parquet tables, read in place. There is no load step: the dbt
sources are `read_parquet(...)` expressions pointed at
`$PIPELINE_DATA_DIR/lake/silver/<table>/**/*.parquet`, so a silver rerun is
visible to the next `dbt run` with nothing copied.

Command: `python -m pipeline.gold` (`--target`, `--data-dir`). It runs
`dbt deps` when the project has packages, then `dbt run`, then `dbt test`, all
with `--project-dir dbt --profiles-dir dbt`, so orchestration has one command
per stage. The same three commands can be run by hand:

```bash
uv run python -m pipeline.gold                                  # build and test
uv run dbt run   --project-dir dbt --profiles-dir dbt           # build only
uv run dbt test  --project-dir dbt --profiles-dir dbt           # test only
uv run dbt docs generate --project-dir dbt --profiles-dir dbt   # lineage + catalog
```

The profile is committed at `dbt/profiles.yml` rather than left in `~/.dbt`, so
a fresh clone builds with no setup. It writes one DuckDB file,
`$PIPELINE_DATA_DIR/warehouse/meta.duckdb`, which holds no state worth keeping:
deleting it and rerunning produces the same warehouse.

### Staging

Four views, one per silver table (`stg_games`, `stg_game_sides`, `stg_turns`,
`stg_cards_seen`): renames and casts, nothing else. Views rather than tables
because materializing a rename layer would copy the lake into DuckDB for no
gain. Two surrogate keys are derived here rather than in each model that wants
them, so the dimension and the fact cannot drift apart: `player_key` (silver's
token, already NULL for a stranger) and `archetype_key`.

### Star schema

`fct_game_side` is the fact, at the (game, seat) grain, two rows per game. That
grain is what makes every mart a group-by: a win rate is an average of
`is_win`, a matchup is a group-by on two columns, a player's record is a
group-by on `player_key`. The other seat's archetype is denormalized onto the
row as `opponent_archetype_key`, so a matchup query never self-joins. Nothing
is filtered out of the fact, `excluded_from_stats` included: the marts drop
those rows, and a fact that had already dropped them could not answer how many
there were.

Six dimensions:

- `dim_player`: one row per member token, with first and last seen and the
  seats they hold. Strangers are absent by construction, not by a filter here,
  because silver already wrote their token as NULL
  ([data-handling.md](data-handling.md)).
- `dim_archetype`: one row per archetype, keyed by the shared archetype row's
  id when a game carries one and by the canonical name otherwise (prefixed
  `name:`, so the two key spaces cannot collide). `aliases` keeps every raw
  label the archetype has arrived under, so a rename stays visible.
- `dim_season`: one row per season plus a synthetic `unknown`, because the
  season trailer only exists in a debug export and most games have none.
  Pointing them at `unknown` keeps them joinable.
- `dim_format`: a placeholder, and the model says so. Nothing upstream records
  the ruleset a game was played under, so the key is the export variant, which
  is not a format at all. It is kept rather than dropped so the fact has a
  stable `format_key` slot: adding a column to a fact later is a smaller change
  than adding a dimension to a star schema.
- `dim_card`: one row per card observed, with the catalog columns silver had
  already joined on. Built from `cards_seen` rather than from the catalog file,
  so gold does not fail when the optional catalog was never fetched.
- `dim_date`: one row per play date, with International Organization for
  Standardization (ISO) year, week, week start and month. Not a gap-free
  calendar: nothing yet needs to show an empty day.

### Marts

- `mart_matchups`: archetype A against archetype B, one row per ordered pair.
  `games`, `wins`, `losses`, `ties`, `win_rate` (wins over wins plus losses,
  null when nothing was decided) and `min_games_met`, the `min_games` project
  variable that defaults to 5. Symmetric by construction rather than by a
  union: a game puts one row in the fact per seat and each carries both
  archetypes, so the same game lands once as (A, B) and once as (B, A) and the
  application can look up either direction. A mirror row counts each mirror
  game twice, once per seat.
- `mart_archetype_weekly`: one row per archetype per ISO week. `games` counts
  seat rows, which is the right numerator for a win rate because a win belongs
  to a seat. `week_games` counts games once each, taken from the uploader seats
  because exactly one seat per game is the uploader. `share_of_week` divides
  the two, so it reads as the share of the week's games the archetype was one
  of the two decks in, and sums to roughly two across a week rather than one.
- `mart_cards_seen`: one row per (archetype, card). `seen_rate` is the share of
  games in which the card was observed being played or revealed. It is not a
  deck inclusion rate: stock exports only reveal played cards. `inclusion_rate`
  sits next to it, computed over the seats that shared a full decklist in game,
  and it is still bounded by observation because silver carries no row per
  decklist card. Both are lower bounds; the model description says which is
  which and the two are never averaged together.
- `mart_player_summary`: one row per member, their record and the archetype
  they play most. Members only, again by construction.

### Tests

105 of them today, run by `dbt test` and therefore by `python -m pipeline.gold`.
`unique` and `not_null` on every primary key, the fact's `game_side_key`, each
dimension's key and each mart's grain key; `relationships` from every foreign
key on the fact to its dimension, with the `player_key` one scoped to the
non-null rows because a stranger has no key; `accepted_values` on `seat`
(0 and 1, the seat numbering silver takes from the contract's `players`
array), on `result_for_seat` and on `export_variant`. Two singular tests carry
the invariants a generic test cannot state: `assert_two_sides_per_game`, which
repeats silver's reconciliation on the other side of the join, and
`assert_matchups_symmetric`, which checks that A vs B and B vs A exist as a
pair, agree on games, and mirror wins against losses.

`tests/test_gold.py` (marker `dbt`, skipped by the default `pytest` run) builds
silver from the committed fixtures, runs the whole thing through `run_gold`,
and asserts the numbers dbt cannot: two fact rows per fixture game, the matchup
symmetry as an independent query, `seen_rate` inside its bounds, `dim_player`
exactly as large as the set of uploader tokens silver wrote, and the model's
feature table built to its grain with no game straddling the train and holdout
boundary.

The models under `dbt/models/ml/` are built by the same `dbt run`, because they
are dbt models like any other; stage 4 describes them.

At 1000x: the models are ordinary SQL, so the change is the adapter. DuckDB
over Parquet becomes a warehouse the same models compile against, the marts
become incremental on `play_date`, and the fact stops being rebuilt in full.
The grain and the tests do not change.

## 4. Model (in progress, LightGBM under MLflow)

Input: `features_turn`, read out of the same DuckDB warehouse gold wrote.
Command: `python -m pipeline.train` (`--experiment`, `--tracking-uri`,
`--params key=value ...`, `--warehouse`). Output: two MLflow runs. Nothing is
written back to the warehouse.

### The features

Three dbt models under `dbt/models/ml/`, tagged `ml` and built by the ordinary
`dbt run`, so the training data is versioned SQL with tests on it rather than a
notebook cell:

- `ml_labeled_side` (ephemeral): the seats the model may learn from. It drops
  seats with no archetype on either side, results that are not a win or a loss,
  hand-logged games with no turns, games the uploader excluded from statistics,
  and games whose first player could not be resolved.
- `ml_split_cutoff`: one row, one date, the boundary between training and
  holdout. Computed as a percentile over the distinct play dates weighted by
  games, so that roughly the last quarter of games falls on or after it; the
  `holdout_start` variable pins it instead when a run has to be reproduced.
- `features_turn`: one row per (game, seat, turn number), holding the state of
  the game **at the start of that turn** and the label of how it ended. Both
  seats get a row at every turn number, not only at their own turns, and every
  counter is summed over turns strictly before this one, which is what keeps
  the label out of the features. Two singular dbt tests enforce exactly those
  two properties.

Bench size, hand size and energy in play are not in the table and are not
approximated: the log counters count actions, and nothing counts a Pokemon
leaving play, so a cumulative `n_play_pokemon` is "Pokemon played so far" and
not a bench. [features.md](features.md) is the column-by-column document, with
the source and the reason for each one.

### The training run

`python -m pipeline.train` trains a LightGBM binary classifier on
`split = 'train'`, evaluates it on `split = 'holdout'`, and logs into one
MLflow run: every hyperparameter, log loss and area under the curve on both
halves, a calibration plot, a feature-importance plot and comma separated file,
`features.json` with the ordered feature list and dtypes, the archetype code
map a serving layer will need, the row counts and date ranges of both halves as
parameters, the git commit as a tag, and the model itself through
`mlflow.lightgbm.log_model` with a signature and an input example.

The feature list is narrower than the table. `seat` is an array index,
`prizes_remaining_*` are exactly six minus columns already present, and
`is_uploader` is a fact about who kept the log rather than about the game: on
this corpus the uploader won every eligible holdout game, so a model trained
with it scores near one and has learned nothing. The list of withheld columns
is logged as a parameter of the run.

### The baseline

A second run, in the same experiment, tagged `baseline=true`, scored on the
same holdout: predict the archetype pair's historical win rate from the
training split, falling back to that archetype's overall rate and then to the
global rate. It is the matchup mart used as a model, which is what the pipeline
can already serve with no model at all, and it is the number the gradient
boosted model has to beat to be worth its dependency. `beats_baseline` is
logged as a 0 or 1 metric on both runs, the command prints the comparison in
words, and it exits 0 either way: a model that loses to a group-by is a
result, not a crash.

### Tracking

`MLFLOW_TRACKING_URI` when it is set, otherwise a plain directory of runs at
`data/mlruns`, so a fresh clone trains with no server. MLflow 3 keeps that
directory store behind `MLFLOW_ALLOW_FILE_STORE`, which the command sets for
itself; reading the same runs with `mlflow ui --backend-store-uri data/mlruns`
means exporting it by hand. `compose.yaml` has an
`mlflow` service (SQLite backend, artifacts on a mounted volume, port 5000) for
when the user interface is wanted:

```bash
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m pipeline.train
```

### Tests

`tests/test_train.py` (marker `ml`, skipped by the default run, and the only
slow suite that needs no Java) builds a synthetic `features_turn` straight into
a temporary DuckDB file, with a signal planted in `prize_diff`, and runs the
whole command against it into a temporary tracking directory. It asserts the
run exists with the required parameters, metrics and artifacts, that the logged
model loads and returns probabilities in [0, 1], that the baseline run exists
and loses to the planted signal, and that the last training day is before the
first holdout day.

### The honest claim

With 128 games landed and 28 of them carrying an archetype on both seats,
`features_turn` is 596 rows. That is enough to demonstrate the loop end to end,
feature table to tracked experiment to comparable baseline, and it is not
enough to claim a good win-probability model. The archetype keys are close to
player identifiers at this size, so both the model and the baseline score
higher on the holdout than either deserves. The numbers worth reading are the
row counts, the date ranges and the gap between the two runs; the absolute area
under the curve is not.

At 1000x: nothing about the shape changes. The feature table becomes
incremental on `play_date`, the split cutoff becomes a rolling window rather
than a percentile of everything, and the tracking store becomes the compose
service or a hosted one. Cross-validation over time folds becomes worth its
runtime, which at 28 games it is not.

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

- Manual games: `exportVariant: "manual"` games now arrive as summary-only v2
  blobs, so bronze lands them and silver gives them a `games` row and two
  `game_sides` rows with null counters, no turns and no cards seen. Any mart
  that counts turns or cards has to exclude them explicitly.
- v1 backfill: v1 blobs are quarantined, not landed, because they carry no play
  date. An upstream admin re-parse rewrites them as v2 and the next run picks
  them up with no code change here.
- Uploader archetypes: the application derives the opponent archetype from the
  log but not the uploader's own. Until it does, most uploader seats have no
  archetype and are excluded from the marts (upstream ticket).
- Archetype tombstones: the alias map is derived from bronze, which handles a
  rename but not a merge of two archetype ids into one. A `mergedInto` export
  from the application is still the only way to collapse those.

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
