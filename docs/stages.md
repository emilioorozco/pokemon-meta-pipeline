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
[7 orchestrate]  Airflow DAG over every stage above, daily, plus a quality gate
        |        (`python -m pipeline.run_all` is the same list with no scheduler)
        |
        v
[8 publish]  marts written back into the application's DynamoDB table
```

Status legend: done, in progress, planned.

## Where the lake lives

Every stage takes one root and derives everything else under it:
`PIPELINE_DATA_DIR`, or the `--data-dir`, `--bronze-dir`, `--silver-dir`,
`--out-dir`, `--warehouse` and `--card-index` flags that override parts of it.
That root can be a directory or an `s3://bucket/prefix`, and the layout under it
is the same either way:

```
<root>/
  lake/bronze/play_date=YYYY-MM-DD/part-0.parquet
  lake/silver/<table>/play_date=YYYY-MM-DD/*.parquet
  lake/quarantine/<reason>/<flattened key>.json + .meta.json
  lake/run_metrics/<run id>-<stage>.parquet
  catalog/cards.json, catalog/card_text.jsonl, catalog/card_index/
  warehouse/meta.duckdb, warehouse/marts/<mart>.parquet
  mlruns/                        MLflow's runs, params, metrics, tags, registry
  mlruns-artifacts/<experiment>/ the files those runs logged
  drift/drift_report.md, drift/drift_summary.json
```

Only the root moves. A laptop run and a container run differ by one environment
variable, which is the whole point: the nightly job is the command a reviewer
already ran, pointed somewhere else.

`pipeline/storage.py` is the one module that knows the difference. Its docstring
argues the design; the four things worth knowing before running this are below.

### Credentials, and one client

Credentials come from boto3's default chain, the same one the ingest already
uses: environment, shared profile, container or instance role. Region comes from
`AWS_REGION`. An endpoint override (`AWS_ENDPOINT_URL_S3`, else
`AWS_ENDPOINT_URL`) is honoured for a local S3 stand-in such as moto, LocalStack
or MinIO, and is what the end-to-end test of the S3 path uses. No key is ever
written into a profile, a DAG or this repository.

The two engines are the exceptions, because neither speaks through boto3. Spark
reads and writes through the Hadoop `s3a://` connector, with a credential
provider that walks the same places; the connector is a jar, fetched from Maven
at session start unless `PRA_SPARK_PACKAGES` names one the image already ships.
DuckDB reads through `httpfs` with a `credential_chain` secret, which is its own
walk of the same places.

### There is no rename, so a partition replace is two steps

A bronze partition is replaced whole, which is what makes a re-run idempotent.
On a disk that is a temporary file moved into place with `os.replace`. Object
storage has no rename, so the replace writes the new object first and deletes
whatever the new set did not name second. A reader that lists the prefix between
the two steps can see both sets.

The order is deliberate. Delete-then-write has a window in which the partition
is empty, and an empty partition reads as "this day has no games", which is a
wrong answer; write-then-delete has a window in which a stale file is still
there, which is a duplicated one. In practice the window does not open at all
for bronze, because the file name is fixed (`part-0.parquet`) and a rewrite of
the same partition overwrites that key. It opens only when the set of file names
changes, which is a layout change. Closing it properly means a manifest, which
means a table format (Apache Iceberg, Delta Lake); that is the next design step,
not this one, and the same paragraph in `pipeline/bronze.py` says so.

### DuckDB builds on local disk

A DuckDB database is a file the engine seeks around in, not a stream it appends
to, and there is no such thing over object storage. So `python -m pipeline.gold`
always builds `meta.duckdb` on the task's local disk, whatever the root is, and
then uploads it under `warehouse/` afterwards. That is honest about what the
file is: it holds no state worth keeping, it is rebuilt from the lake on every
run, and the copy in the bucket is a convenience for the next reader and not a
database anyone writes to in place.

dbt reads silver in place through `httpfs`, so there is still no load step. The
profile has two targets for it, `dev` and `s3`, identical except that `s3` loads
the extensions and creates the credential secret; the stage picks one from the
shape of the root.

The marts are also written out as Parquet, one file per materialized table,
under `warehouse/marts/<mart>.parquet`. The next design step reads the marts
without DuckDB at all, and a Parquet file is what every engine can open; a
reader that only wants the matchup matrix should not have to download a
warehouse to get it.

Everything downstream that opens the warehouse (`publish`, `quality_gate`,
`train`, `drift`, the agent's `query_marts`, `eval`) takes a `--warehouse` that
can be `s3://.../meta.duckdb`. It is downloaded once per process and opened
read-only, with `httpfs` loaded: the `ops` models are views over the run-metrics
Parquet, so reading the published warehouse means reading the lake too.

### MLflow is a synced file store with its artifacts left on S3

With an `s3://` root, `mlruns/` is downloaded to a temporary directory at the
start of `train`, `promote`, `drift` and `eval`, used as the local file store
for the length of the command, and uploaded back at the end. `publish` and the
serving application read it the same way and do not upload, so a reader is never
briefly a writer. Nothing is deleted on the way up, and a command that raises
uploads nothing.

**Only the metadata is synced.** The runs, parameters, metrics, tags and the
registry are small and are what the sync carries; the artifacts stay on S3 under
`mlruns-artifacts/<experiment>/` and are read and written in place. The reason
is that MLflow's file store records absolute paths. An experiment created inside
one command's temporary directory writes that directory into its
`artifact_location`, every run under it inherits it as its `artifact_uri`, and
every registered version points at the same place, so the next command reads a
store whose files are all in a directory that no longer exists: `serve` cannot
load `models:/win-probability@production`, `promote` cannot open a candidate's
artifacts, `drift` cannot read the reference it wrote yesterday. Creating each
experiment with an `s3://` artifact location instead makes every one of those
URIs absolute somewhere that outlives the command and is the same from any
machine. `pipeline.storage.experiment_id` is the one place that does it, and
`train`, `drift` and `eval` all go through it.

The artifact prefix is a sibling of `mlruns/` and not a directory inside it,
because the sync copies the whole store down and back on every command: under
it, logging one metric would mean downloading every model ever trained.

A local root is unchanged. No artifact location is named, MLflow puts the files
beside the runs, and `mlflow ui --backend-store-uri data/mlruns` finds them.

**The single writer is an assumption, not a guarantee.** Two commands syncing
the same prefix at once will each upload their own view, and the later one wins
for any file they both touched. Today there is one writer: the nightly job, one
stage at a time, under one run identifier. The reason for doing it this way is
that it is free and reversible, needs no service standing up, and disappears the
moment `MLFLOW_TRACKING_URI` points at a server.

A tracking server is the design the day there is a second writer, and
`docs/orchestration-on-aws.md` already recommends it and says why: the file
store's own documentation warns it is unsafe under concurrent writers, and the
registry it protects is what decides which model answers `/predict`.

**One-time operator step.** An experiment's artifact location is written when
the experiment is created and cannot be corrected afterwards, so experiments
created on a lake before this change keep pointing at a temporary directory that
is gone. Delete the `mlruns/` prefix under the lake root once, before the first
run with this behaviour, and let `train` recreate it:

```
aws s3 rm "$PIPELINE_DATA_DIR/mlruns" --recursive
```

Nothing downstream depends on what is there: the warehouse and the lake are
rebuilt from bronze, and a registered version is reproduced by the next training
run. Do this only against a lake whose registry you are willing to lose.

### What the tests cover, and what the live run has to

The moto-backed tests cover the storage helper, the bronze partition replace and
single-game merge, the backfill from fixtures, quarantine, run metrics and the
gold build with its publish back, all against an `s3://` root. `run_all` and the
dbt build run against moto's threaded server rather than its in-process fake,
because a subprocess and DuckDB's own HTTP client cannot see a patched botocore.

Silver on `s3a://` is the one path with no test: the shared SparkSession in the
suite is started once for the whole run, and the connector is chosen when the
Java Virtual Machine starts, so a second session with different jars is not
possible in the same process. It is verified by running the pipeline end to end
against a local S3 server, and by the live run against a real bucket.

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

### 1b. Event path (in progress)

Command: `python -m pipeline.consume` (`--once`, `--max-messages N`,
`--wait-seconds N`, `--bronze-dir`, `--quarantine-dir`), or the `consumer`
service in `compose.yaml`, which is the same command in a container with the
lake mounted from the host. This is the development shape of the stage and
nothing operational depends on it: the deployed shape is the Lambda in 1c
below, which runs the same routine with no machine of its own, and the queue
of a deployed environment already has that function on it. Reach for the
command to point the ingest at a scratch lake while changing it, not to drain
a real queue. Either way the bucket notification on `parsed/` for
`s3:ObjectCreated:*` and `s3:ObjectRemoved:*`, the `parsed-games` queue and its
dead-letter queue at three deliveries are defined in the application's stack.
The command-line consumer reads `PRA_QUEUE_URL`, which nothing else does.

It logs and reports itself like every other stage (the Ops section below), with
one difference that follows from never finishing: the unit it records is the
receive batch, not the run. A batch that came back with messages writes one
`run_metrics` row counting what that batch received, landed and quarantined,
and an idle long poll writes nothing.

What a message turns into:

- A record is acted on when it names the configured bucket, its key is under
  the prefix and ends in `.json`, and its `eventName` starts with
  `ObjectCreated` or `ObjectRemoved`. Keys arrive URL-encoded and are decoded
  first.
- A created object goes through the backfill's own `process_object`: the same
  decode, contract check, anonymization, re-validation and quarantine reasons.
  There is no second implementation of any of it.
- A removed object is taken out of bronze. The blob is gone, so its play date
  cannot be read off it; the partition holding it is found by scanning the
  `source_key` column of the partition files. A key bronze never landed is a
  no-op, counted as ignored.
- Anything else is counted as ignored and acknowledged: the S3 test event sent
  when a notification is configured, a key outside the prefix, another event
  type.

When a message is deleted, which is a deliberate deviation from "leave it on
any failure": the message is deleted when every record in it was handled, and a
blob quarantined for a reason that belongs to the blob (`invalid_json`,
`contract_violation`, `v1_blob`, `handle_leak_check_failed`) counts as handled.
A bad blob fails identically on every redelivery, so keeping the message would
replay one failure three times and then bury the evidence in the dead-letter
queue; the quarantine pair on disk is the record instead. The message is left,
and left only, for what a retry can fix: an S3 error, a write failure, a body
that is not an S3 event at all. Those are redelivered and land in the
dead-letter queue after three attempts. The backfill's `write_failed`
quarantine has no counterpart here, because a queue already has the retry that
a batch run does not.

How one game is written: bronze partitions by play date and the backfill
replaces a partition whole, which for a single game would delete the rest of
its day. So an event writes with an upsert: read the partition, drop any row
with the same `game_id`, append the new row, write the partition back through
the same atomic replace. A delete is the same rewrite without the row, and a
partition left with no rows is removed. Applying the same message twice
therefore changes nothing, which is what an at-least-once queue requires.

At scale both the rewrite and the scan are wrong: a partition of a million rows
cannot be rewritten per event, and the delete lookup cannot walk every file.
The rewrite becomes an append-only file per event plus a compaction step, or a
table format (Apache Iceberg, Delta Lake) that does row-level upserts and
deletes; the lookup becomes a `game_id -> play_date` index written beside the
partitions. Neither changes the contract, the routing or the quarantine.

### 1c. The event path as a Lambda (in progress)

The same ingest with the polling taken out. `pipeline.lambda_consumer.handler`
is a Lambda handler behind an SQS **event source mapping**: the mapping is the
thing that long-polls the queue, batches up to ten messages and invokes the
function with them, and on a clean return it deletes the messages the function
did not report back. So the function holds no receive loop, no
`delete_message` and no visibility timeout, and `PRA_QUEUE_URL` is not in its
environment at all: the mapping owns the queue, and the code owns what a
message means. The queue, the bucket notification, the function, its role and
the mapping are defined in the application's CDK stack, beside the producer
that fills the queue; this repository owns the image the function runs.

Per message it calls `MessageHandler.apply` from `pipeline.consume`, which is
the routine the command-line consumer calls, so every routing and quarantine
decision above is the same decision here and there is no second implementation
to keep in step.

**Partial batch responses** are how a failure stays the failing message's
alone. The handler returns
`{"batchItemFailures": [{"itemIdentifier": "<messageId>"}, ...]}`, and SQS
deletes the rest of the batch and redelivers only the named ones. Without it, a
Lambda that raises fails its whole batch: nine good messages become visible
again with the one poison message and ride along with it to the dead-letter
queue after three deliveries, and the receive count that the redrive policy
counts to three stops meaning anything. The mapping has to be created with
`ReportBatchItemFailures` for the field to be read; a mapping without it
ignores the return value silently. Which records are named is unchanged from
the section above: a blob that fails the contract is quarantined and
acknowledged, and an S3 error, a storage error or a body that is not an S3
event is named so the queue can try again.

**Reserved concurrency is 1**, and that is a correctness setting rather than a
throughput one. A bronze partition is written by reading the day's Parquet
file, dropping the rows this message replaces and writing the file back. Two
invocations doing that to the same day at once would each build a file from
what it read before the other wrote, and the second write would silently drop
the first one's game. One invocation at a time makes that impossible. The
ceiling it sets is a batch of ten games every second or so, which is orders of
magnitude above what this application produces; lifting it later means a table
format with row-level writes (Apache Iceberg, Delta Lake), not more writers
over the same rewrite.

**The key** comes from Secrets Manager, not from a function environment
variable: an environment variable is readable by anyone who can call
`GetFunction`, and this key is the one thing standing between the lake and a
reversible handle. `HANDLE_HMAC_KEY_SECRET_ARN` names the secret, the value is
read once per execution environment and kept for the life of the container, and
a rotation arrives as a new container, which a deployment or an idle timeout
produces on its own. `HANDLE_HMAC_KEY` in the environment still wins, so the
handler can be invoked on a laptop with no AWS call at all ([demo.md](demo.md)).
The secret is put there once by hand from the 1Password value; nothing in either
repository writes it.

**The image** is `Dockerfile.lambda`, built on `public.ecr.aws/lambda/python`
for x86_64. Its dependency set is `uv export` of the project's runtime
dependencies from `uv.lock`, minus DuckDB, which is boto3, pydantic, pyarrow
and python-dotenv with their transitives: no Spark, no MLflow, no LangChain, no
torch. That set is what the handler's import graph actually needs, and
`bronze.read_smoke` imports DuckDB inside the function rather than at the top
of the module so that the one command-line use of it does not follow the
consumer into the image. A test asserts the claim by importing the handler in a
subprocess and checking `sys.modules`. A container image rather than a zip
because pyarrow alone is most of the 250 MB unzipped limit; the cost is a few
seconds of cold start, which a queue consumer does not notice because nothing
is waiting on the reply.

`.github/workflows/consumer-image.yml` builds and pushes it on every push to
`main` that touches `pipeline/`, `contract/` or the Dockerfile, tags it
`:latest` and `:<commit>`, then runs `aws lambda update-function-code` on the
commit tag and waits for the update to finish. It does that once per
environment: the job is a matrix over `dev` and `prod`, `fail-fast` off so a
broken dev deploy does not hold prod back, and a `workflow_dispatch` input
narrows it to one when only one needs rebuilding. It assumes its role by
OpenID Connect, so no access key is stored; the role, the region, the registry
repository and the function name are variables on the GitHub Environment of
the same name, set by hand, which is also what puts the stage in the token's
subject that the role's trust policy is bound to. The workflow skips an
environment with a notice when its variables are unset, so a fork does not see
a red build for not having an account.

How to watch it. The logs are JSON lines in the function's CloudWatch log
group, carrying the same `run_id`, `stage` and event fields every other stage
logs, with the run identifier being the invocation's own request id, so one
invocation is one filter in Logs Insights:

```
fields @timestamp, msg, run_id, rows_out, rows_quarantined
| filter stage = "consume"
| sort @timestamp desc
```

The numbers are also a table. Every invocation writes one `run_metrics` row to
the lake, so the `ops` dbt models and the health mart count the function's runs
beside every other stage's with nothing to configure; `extra_json` on those
rows carries `ignored`, `deleted` and `failed`. Beyond that it is the queue's
own metrics: `ApproximateNumberOfMessagesVisible` near zero on the queue, and
anything at all on the dead-letter queue is the alarm worth having, because by
construction only a message three retries could not fix reaches it.

## 2. Silver (in progress, PySpark)

Input: the bronze Parquet tables and, optionally, the card catalog
(`data/catalog/cards.json`, fetched by `scripts/fetch_catalog.py`). No archetype
export is needed: the alias map is built from bronze itself, see below.

Command: `python -m pipeline.silver` (`--bronze-dir`, `--silver-dir`,
`--catalog`, `--master`). It needs a Java Virtual Machine (JVM); everything else
is the `spark` extra.

Bronze is one nested row per upload. Silver is the grain change, four tables
under `data/lake/silver/<table>/play_date=YYYY-MM-DD/`, and the first thing the
stage does is collapse bronze to one row per `game_id`, keeping the earliest
`source_last_modified` and breaking ties on `source_key` ascending, so a match
both players uploaded becomes one game with `upload_count = 2` rather than two
of everything downstream:

- `games`: one row per game. Identity and lineage (`game_id`, `user_id`,
  `play_date`, `played_at`, `source_key`, `upload_count`, `ingested_at`,
  `contract_version`), the export's own description of itself
  (`export_variant`, `upload_source`, `parser_version`, `unparsed_count`,
  `played_at_source`, `has_full_decklists`, `excluded_from_stats`, `season_id`,
  `season_name`), and the outcome with every
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
  with no counter still lands in `n_entries`.

  Four of those kinds are counted a second time, by the seat the log credits
  each line to rather than by the seat whose turn it was: `n_attack_self` and
  `n_attack_opp`, `n_energy_attach_self` and `n_energy_attach_opp`,
  `n_prize_self` and `n_prize_opp`, `n_knockout_self` and `n_knockout_opp`.
  The two attributions are not the same number. A Pokemon can go down on its
  own owner's turn from a card effect, so the knockout and the prizes taken
  for it belong to the other seat; `n_attach` counts tools as well as energy;
  and `n_prize_taken` counts prize lines, while a line can take two or three
  cards. These eight are the only place the stage looks inside `fields_json`,
  for the `energy` flag of an attachment and the `n` of a prize, and
  `mart_archetype_pace` is what reads them.
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

Two commands go with editing the models rather than with building:

```bash
uv run python scripts/generate_warehouse_tables.py              # after a model changes
uv run python scripts/generate_marts_schema.py                  # after a marts description changes
```

The first writes `pipeline/warehouse_tables.py`, the list of relation names
the agent's SQL validator uses to tell a table the model invented from a real
one it may not read. The second writes `pipeline/marts_schema.py`, the
descriptions the agent's prompt renders its table listing from.

Both are committed because the serving image ships without the dbt project,
so reading it at run time answered nothing there: an empty relation list made
the validator call real tables invented (`docs/sql-gate.md`), and an empty
schema sent the model a prompt with no table listing in it at all
(`docs/agent-service.md`). A test re-runs each parse and fails when the
committed file has drifted, so forgetting either command is a red test rather
than a wrong answer in production.

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
- `mart_archetype_turn_order`: going first against going second, one row per
  (archetype, opponent archetype, side of the table), plus an all-opponents
  row per archetype and side under the literal opponent key `all`. `games`,
  `wins`, `losses`, `decided_games`, `win_rate` over the decided games, and
  `ci_low` and `ci_high`, the Wilson 95% interval around that rate. The split
  is the one thing no public source records: a site that publishes matchup
  tables does not know who opened, and every seat row here does. Both sides
  are always present, so a pairing played from one seat only carries a row of
  zeros on the other, because "0 games going first" is an answer and a
  missing row is not. A seat whose opening side the log never recorded is in
  neither half, the way `mart_player_summary` counts it. Wilson rather than
  the rate plus or minus two standard errors, because the corpus is small and
  the normal approximation is wrong exactly there: at four games and four
  wins it gives an interval of zero width around 100%. Two sides whose
  intervals overlap are two sides this corpus cannot tell apart, and on a
  corpus this size that is nearly all of them.
- `mart_archetype_weekly`: one row per archetype per ISO week. `games` counts
  seat rows, which is the right numerator for a win rate because a win belongs
  to a seat. `week_games` counts games once each, taken from the uploader seats
  because exactly one seat per game is the uploader. `share_of_week` divides
  the two, so it reads as the share of the week's games the archetype was one
  of the two decks in, and sums to roughly two across a week rather than one.
- `mart_archetype_pace`: one row per archetype, ten numbers for how fast the
  deck plays: the turn of its first attack, the share of its turns with no
  attack, energy attached per turn, prizes taken by the end of turns 4, 6, 8
  and 10, the turn of its first prize, the turn of its first knockout and the
  turn a concession ended the game on, each averaged over the seats that
  played it, with `games` and `min_games_met` beside them. The ten are the ten
  the application already computes for a member's own game when it is
  uploaded, written a second time in SQL so a member's number and the
  community's number are the same measurement and can be put side by side.
  `int_game_side_pace` is the ephemeral model that holds the definitions one
  seat at a time; the two writings are held together by an equality test over
  the ten fixture games rather than by a comment.
- `mart_cards_seen`: one row per (archetype, card). `seen_rate` is the share of
  games in which the card was observed being played or revealed. It is not a
  deck inclusion rate: stock exports only reveal played cards. `inclusion_rate`
  sits next to it, computed over the seats that shared a full decklist in game,
  and it is still bounded by observation because silver carries no row per
  decklist card. Both are lower bounds; the model description says which is
  which and the two are never averaged together.
- `mart_player_summary`: one row per member, their record and the archetype
  they play most. Members only, again by construction. The record is also
  split by which side opened the game, `games_first`, `wins_first`,
  `games_second` and `wins_second`, because "does going first matter for me"
  is a question about one member and this is the only mart keyed by one.

### Tests

140 of them today, run by `dbt test` and therefore by `python -m pipeline.gold`.
`unique` and `not_null` on every primary key, the fact's `game_side_key`, each
dimension's key and each mart's grain key; `relationships` from every foreign
key on the fact to its dimension, with the `player_key` one scoped to the
non-null rows because a stranger has no key; `accepted_values` on `seat`
(0 and 1, the seat numbering silver takes from the contract's `players`
array), on `result_for_seat` and on `export_variant`. Four singular tests
carry the invariants a generic test cannot state: `assert_two_sides_per_game`,
which repeats silver's reconciliation on the other side of the join;
`assert_matchups_symmetric`, which checks that A vs B and B vs A exist as a
pair, agree on games, and mirror wins against losses;
`assert_turn_order_sides_account_for_every_seat`, which counts the two sides
of each pairing out of the fact rather than out of another mart and insists
both sides have a row; and `assert_turn_order_interval_contains_the_rate`,
which holds every rate and every bound inside 0 and 1 and the interval around
its own rate.

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
Commands: `python -m pipeline.train` (`--experiment`, `--tracking-uri`,
`--params key=value ...`, `--warehouse`), `python -m pipeline.promote`
(`--candidate`, `--metric`, `--tracking-uri`), `python -m pipeline.serve`
(`--host`, `--port`, `--tracking-uri`, `--alias`) and `python -m pipeline.drift`
(`--reference`, `--window-days`, `--as-of`, `--psi-threshold`,
`--tracking-uri`, `--warehouse`, `--out-dir`). Output: two MLflow runs and one
new registered model version per training run, an alias move per accepted
promotion, an HTTP service, and a drift report with its own MLflow run.
Nothing is written back to the warehouse.

The four are deliberately four commands and not one. Training is allowed to
produce a worse model, promotion is the only thing that decides what is served,
serving holds no opinion at all (it loads whatever the alias points at), and
the drift report decides nothing whatsoever: it prints a verdict and leaves the
next move to a person. A single `train-and-deploy` command would make every
scheduled retrain a deployment.

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

`MLFLOW_TRACKING_URI` when it is set, otherwise a plain directory of runs and
registered versions at `data/mlruns`, so a fresh clone trains, registers,
promotes and serves with no server at all. MLflow 3 keeps that directory store
behind `MLFLOW_ALLOW_FILE_STORE`, which each of the three commands sets for
itself; reading the same runs with `mlflow ui --backend-store-uri data/mlruns`
means exporting it by hand. `compose.yaml` has an
`mlflow` service (SQLite backend, artifacts on a mounted volume, port 5000) for
when the user interface is wanted, or when the registry has to be shared by
more than one machine:

```bash
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m pipeline.train
```

### The registry and the promotion step

Every training run registers its model as a new version of the `win-probability`
registered model, tagged with `holdout_logloss`, `holdout_auc`,
`beats_baseline`, `train_from`, `train_to` and `holdout_to`. Both ends of the
training window and not only the last day, because the drift report selects
rows by them and a window with one end is a filter that quietly reaches back to
the first game ever played. The baseline run is not
registered: it is a yardstick, and a registry entry nothing can serve is a
loaded gun. Registering unconditionally is the point, because "which models
have we trained" and "which model are we serving" are different questions and
the second one has its own command.

`python -m pipeline.promote` answers the second. It reads the candidate's
holdout numbers (from the version tags, falling back to the source run's
metrics) and compares them with whatever currently holds the `production`
alias:

- `beats_baseline` must be 1, or it is refused. A model can improve on the
  model before it and still lose to the archetype win-rate group-by, and
  shipping that is the same numbers with a LightGBM dependency in front.
- Then the primary metric, `--metric logloss` by default: lower is better, and
  the candidate has to be at least as good. On an exact tie the other metric
  breaks it (`auc`, higher is better), and a tie on both promotes, because the
  newer version was trained on more recent games.
- The first ever promotion has nothing to compare against, so beating the
  baseline is the whole test.

Accepted, the candidate takes the `production` alias. Refused, it takes
`staging` and the reason is written onto the version as a `promotion_decision`
tag and printed. Either way the command prints one line and exits 0, because a
refusal is the gate working; exit 2 is for a candidate that does not exist.

Aliases, not stages. MLflow 3 deprecates the old `Staging` and `Production`
stages, and an alias is the better shape anyway: it is a pointer, so
`models:/win-probability@production` is a stable address, a rollback is moving
the pointer back, and the version's own history stays a record rather than
something that gets edited.

```
train      -> version 4 registered, tagged, serving nothing
promote    -> rejected: version 4 has holdout logloss 0.3682 against 0.1987 at
              version 3. win-probability version 4 now holds @staging.
```

### Serving

`python -m pipeline.serve` runs a FastAPI application (uvicorn, port 8000) that
loads `models:/win-probability@production` and the archetype code map from that
version's run, and answers:

- `POST /predict`: one board state at the start of a turn, one win probability,
  with the model name, version and alias that produced it. `prize_diff` is
  computed when it is left out. An archetype the model never trained on is
  encoded as the missing category the trainer uses for exactly that case and
  named back in `unknown_archetypes`, rather than rejected: the metagame moves
  every set release, and a 422 would make the ordinary case an error.
- `GET /health`: liveness and `model_loaded`.
- `GET /model`: which version is loaded, when, and its holdout numbers.
- `POST /reload`: re-read the alias, so a promotion reaches the service without
  a restart.

It loads by alias rather than by version, which is what makes the promotion
step a deployment: `promote` moves the pointer, `/reload` picks it up, and no
image is rebuilt. It also means an empty registry is a state and not a crash.
A service that exited because nothing is promoted yet would tell an
orchestrator the image is broken; instead it starts, `/health` reports
`model_loaded: false`, and `/predict` answers 503 with the two commands that
fix it.

The request and response models are Pydantic with a description on every field,
so `/docs` is the schema rather than a separate document, and the feature list
itself lives in `pipeline/ml_features.py`, imported by both the trainer and the
service. A test asserts the request fields are exactly that list.

```bash
uv run python -m pipeline.serve &
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "turn_number": 8, "went_first": true,
  "archetype_key": "name:charizard-ex", "opponent_archetype_key": "name:gardevoir-ex",
  "prizes_taken_self": 3, "prizes_taken_opp": 1,
  "knockouts_self": 3, "knockouts_opp": 1, "cards_drawn_self": 26,
  "energy_attached_self": 5, "pokemon_played_self": 6, "trainers_played_self": 15,
  "evolutions_self": 2, "attacks_self": 4, "turns_played_self": 4 }'
```

```json
{"win_probability": 0.4469, "model_name": "win-probability",
 "model_version": "3", "model_alias": "production",
 "features_used": ["turn_number", "..."], "unknown_archetypes": []}
```

`compose.yaml` has a `predict` service (built from `Dockerfile.serve`, port
8000, `MLFLOW_TRACKING_URI` pointed at the `mlflow` service) for running it
next to the tracking server. The image carries no model, on purpose.

Deployed, the same application runs as a Lambda container function behind a
function URL: `Dockerfile.agent` builds it with the `agent` extra as well, so
`/ask` answers there, and `pipeline/lambda_serve.py` is the handler.
[agent-service.md](agent-service.md) is what runs where, the environment the
function is given, the rule that decides when a warm container re-reads the
warehouse, and the measured cost of a cold start.

It is also the one stage that is traced and scraped rather than writing a
`run_metrics` row: `GET /metrics` is a Prometheus exposition, every request is a
span, and `docker compose --profile observability up -d predict grafana` puts
Jaeger and a Grafana dashboard next to it. See the Ops section below.

### Drift

A model is a claim about a distribution, and the claim expires quietly: the
service keeps answering, the holdout score in MLflow keeps saying what it said
the day it was trained, and nothing in either notices that the metagame moved.
`python -m pipeline.drift` is the noticing. It compares a reference window
against the most recent window of `features_turn` and writes `drift_report.md`
and `drift_summary.json`, logged as artifacts of a run in the
`win-probability-drift` experiment with the windows as parameters and
`max_psi`, `archetype_mix_psi`, `drifted`, and both label rates as metrics.

What is compared:

- **The reference window** is `split = 'train'` by default, which is what the
  last training run learned from. `--reference 4` takes a registered version's
  `train_from` and `train_to` tags instead, so a model still serving from three
  retrains ago can be compared against today without anyone writing the dates
  down.
- **The current window** is the last `--window-days` days (30 by default) up to
  `--as-of`, which defaults to the latest play date in the table.
- **Every numeric feature** gets a population stability index over ten quantile
  bins fitted on the reference, with each bin share floored at 1e-6 so an empty
  bin cannot make the logarithm infinite. Quantile bins rather than uniform
  ones because most of these features are small integer counters, and a uniform
  split of `prizes_taken_self` would be eight empty bins reporting on the
  binning.
- **The two archetype columns** are pooled into one metagame mix, because a
  deck appearing on either side of the table is the same event. The mix gets a
  chi-square test of independence (`scipy.stats.chi2_contingency`, which
  arrives with scikit-learn), a PSI over the shares with each archetype as a
  bin, and the per-archetype share change. Archetypes under two per cent of
  *both* windows fold into `other`; one under two per cent in the reference and
  over it now is exactly the arrival worth seeing, so it keeps its row.
- **The label rate** in both windows, labelled a hint rather than a metric. A
  moving win rate points at concept drift, the relationship between features
  and label changing rather than the features moving, which no PSI can see.

The usual reading of a PSI is printed in the report: below 0.1 stable, 0.1 to
0.2 moderate, 0.2 or more significant. `drifted` is true when any numeric PSI
or the mix PSI reaches `--psi-threshold` (0.2 by default), and the command
exits 0 either way; exit 3 is a current window under twenty rows, where the
honest answer is that there was nothing to look at.

The archetype mix is the trigger to expect in practice. Counters such as
`prize_diff` and `cards_drawn_self` are properties of how the game is played
and move slowly; the deck distribution moves on the day a set releases, and it
moves the feature the model leans on hardest. That is also why serving encodes
an unseen archetype as the missing category rather than refusing it: the report
is how anyone finds out it happened.

When the flag fires, investigate before retraining. On this corpus the numbers
are noisy, a window of a few dozen games can cross the threshold on nothing but
which decks were queued that week, and the overlap line at the top of the
report says how many current rows are also reference rows (a thirty-day window
over a ten-day corpus contains the training window whole, and every PSI is then
near zero by construction). If the shift is real, `python -m pipeline.train`
produces a candidate on the newer data and `python -m pipeline.promote` decides
whether it is actually better. The drift command never does either: an
automatic retrain on a threshold crossing is how a noisy week becomes a
deployment.

```
drift flagged: max feature PSI 0.2841 (cards_drawn_self), archetype mix PSI
0.5512, threshold 0.20, 212 current rows against 448 reference rows
```

### Tests

`tests/test_train.py` and `tests/test_promote.py` (marker `ml`, skipped by the
default run, and the only slow suites that need no Java) build a synthetic
`features_turn` straight into a temporary DuckDB file, with a signal planted in
`prize_diff`, and run the real commands against it into a temporary tracking
directory. The training tests assert the run exists with the required
parameters, metrics and artifacts, that the logged model loads and returns
probabilities in [0, 1], that the baseline run exists and loses to the planted
signal, and that the last training day is before the first holdout day. The
promotion tests train three versions in a row, a short run, a single boosting
round on two leaves, and a long run, and assert that the first takes
`production`, that the deliberately broken second is refused with its reason on
the version and `production` unmoved, that the third moves the alias, and that
a candidate that does not exist exits 2. The rule itself is also tested
directly on hand-built versions, for the ties and the undefined metrics three
training runs cannot be made to produce on demand.

`tests/test_drift.py` (marker `ml`, and it trains nothing) builds two corpora
from the same synthetic table: one whose recent window is a copy of the
training rows, and one where that copy has the prize race three prizes further
along and two archetypes replaced by a deck that did not exist before. The
first has to report a PSI of zero and no drift, which is an assertion rather
than a hope because a distribution compared with a copy of itself has exactly
that; the second has to fire the flag with `prize_diff` at the top of the table
and the new deck at the top of the mix. The rest covers the markdown carrying
its tables and its verdict line, the run carrying its artifacts and metrics,
and a window of under twenty rows exiting 3 with the reason.

`tests/test_serve.py` is in the **fast** suite, not the `ml` one. The loader is
injected, so the endpoints are driven with a stub model through Starlette's
test client: no tracking server, no registry, no LightGBM. It covers the four
endpoints, a probability in [0, 1] carrying its version, an unknown archetype
reported rather than refused, a malformed body as a 422, and `/reload` swapping
the loaded version. What needs a real registry is tested against one in the
promotion suite; what needs neither should not cost a model train.

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

## 5. Serving (in progress, FastAPI)

`POST /predict` is live; it is documented in stage 4, next to the registry and
the alias it loads by, because the three commands are one loop and splitting
them across two sections would hide that. `python -m pipeline.serve` is the
command and `compose.yaml`'s `predict` service is the container.

Still planned: `GET /matchups/{archetype}` straight from the gold marts, over a
read-only DuckDB connection, so the application can ask for a matchup table
without the model being involved at all. It would never touch bronze.

## 6. Agent (in progress, LangChain)

`python -m pipeline.agent "..."` and `POST /ask` are live. A LangChain
tool-calling agent with two tools: SQL over the gold marts, over a read-only
DuckDB connection with an allowlist of tables, and retrieval over the text of
printed cards. It answers a question such as "how does Dragapult ex do against
Gholdengo ex" by writing the mart query, running it, and reporting the number
next to the sample size it came from.

### The tools

`query_marts(sql)` runs one read-only SELECT and returns the rows as a markdown
table with a row count. Every statement goes through `validate_sql` first,
which is a pure function of the statement and the allowlist and is therefore
unit tested with no database behind it. It refuses, naming the rule:

| rule | what it refuses |
| --- | --- |
| one statement | a `;` anywhere but at the end |
| read only | anything not starting with `SELECT` or `WITH` |
| no side effects | `ATTACH`, `DETACH`, `COPY`, `INSTALL`, `LOAD`, `PRAGMA`, `SET`, `CREATE`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and the rest of the statement keywords |
| no file access | `read_parquet`, `read_csv`, `read_json`, `glob` and the other table functions that leave the warehouse |
| allowlist | any table other than the nine below |

The allowlist is `mart_matchups`, `mart_archetype_turn_order`,
`mart_archetype_weekly`, `mart_archetype_pace`, `mart_cards_seen`,
`mart_player_summary`, `dim_archetype`, `dim_card` and `dim_date`. Two absences
are deliberate. `fct_game_side` is off it because the marts aggregate it
correctly and a model writing its own group-by over a two-rows-per-game fact is
where double counting starts. `dim_player`, the member roster, is off it
because the question it answers is "who is here"; so is everything in silver
and staging, so no raw log line is reachable either. `mart_player_summary` is
on it and is the one player-keyed table the agent can read: an aggregate over
members, keyed by the same irreversible token, with no handle in it, and rule 5
of the prompt is what stops the agent presenting a token as a person. A
boundary that is a list of table names is one a reviewer can check
(docs/data-handling.md).

Two more limits sit under the allowlist. The connection is opened `read_only`,
which is the second lock and not the first: `read_only` would still allow
`read_csv('/etc/passwd')`, and the allowlist is what does not. And a query with
no `LIMIT` gets `LIMIT 50` appended, a query with a larger one has it cut to
200, and DuckDB is given a five-second statement timeout, because a mart query
over this corpus is milliseconds and anything slower is a mistake.

A refusal and a failed query both come back to the model as text with a row
count of zero, never as an exception. The model that wrote a bad query is the
only thing that can write a better one, so it has to read why it was refused.

`lookup_cards(query, k)` searches the text of printed cards by meaning and
returns the top k with their abilities, attacks and rules. It is a card
reference and not game data, and the prompt says so: nothing it returns is
evidence about how often anything is played.

### The prompt

`pipeline/prompts.py`. The schema half is rendered at import time out of
`dbt/models/marts/schema.yml`, the same file dbt builds and tests the models
from, cut to one sentence per table and per column so it fits a budget of about
two thousand tokens. A column renamed in the model is renamed in the prompt on
the next import, and a column that never existed cannot be described in it at
all. Only the allowed tables are rendered, so a table the tool would refuse is
never advertised.

The rules beside it are hand written and are the part that matters:

1. cite the sample size, the `games` count, in the same sentence as the number;
2. say in words when `min_games_met` is false, rather than reporting the rate
   alone;
3. `seen_rate` is the share of games in which a card was **observed**, not a
   deck inclusion rate, and must be labelled as an observation and a lower
   bound whenever it is reported;
4. never guess a number: an empty result or a refusal is reported as one;
5. player identity is not available, and a question about a person is answered
   by saying the pipeline anonymizes players before anything is written.

### Provider and model

Anthropic through `langchain-anthropic`, default model
`claude-haiku-4-5-20251001`, overridden with `PRA_AGENT_MODEL`. The key is
`ANTHROPIC_API_KEY` and only these two entry points read it; every other stage
runs without it. The size of model is the job: write one SELECT over seven
tables and read a dozen rows back.

The model is injected, exactly as the serving stage injects its model loader,
so the tests pass a scripted chat model that returns pre-written `AIMessage`s
with real `tool_calls` on them. The loop, the tool, the SQL validation and the
DuckDB query are all the real ones under it; the only thing the fake replaces
is the decision about which SQL to write, which is the part that costs a key
and is not deterministic. No test in this repository needs a provider.

### The card text and its terms

`scripts/fetch_card_text.py` downloads printed card text from
[TCGdex](https://tcgdex.net), a free, open, community-maintained card database
with a public REST API, and writes `data/catalog/card_text.jsonl`, one JSON
object per card: name, set, number, types, hit points, stage, abilities,
attacks, rules text, retreat cost, regulation mark and a `source_url`. Only
the Standard format is fetched: printings whose regulation mark is in
`STANDARD_REGULATION_MARKS` (`pipeline/config.py`, bumped at each rotation),
a few thousand cards rather than the 25,000 printings the catalog lists;
`--reg all` or `--reg G,H` changes that. Within the format, the
local catalog decides what is fetched: its set codes are resolved against
TCGdex's set list, and a catalog entry finds its card by (set, collector
number) first, by name among the listed sets second, and by an exact-name query
against the whole database third. A card that matches nothing is counted and
skipped rather than guessed at, because a wrong card's text in a retriever is
worse than a missing one.

The two sources spell a set differently, and the spelling is the whole match:
the client writes `SV6`, `SV8-5` and `MEBSP` where TCGdex writes `sv06`,
`sv08.5` and `mep`. A code becomes a candidate identifier through a normalizer
(the series number padded to two digits, a `-5` tail written as the `.5` of a
special set) plus a small table for the codes no rule reaches, including the
client's `RSV10-5` and `ZSV10-5`, which are the two halves of one special set
that TCGdex serves as White Flare and Black Bolt. Candidates are checked
against the live set list before they are used, and a code that resolves to
nothing is logged at WARNING, because it is a whole set of the current format
about to be missing. An entry whose set never resolved is counted as unmatched
and broken down by set code in the run's `extra`, rather than being dropped in
silence: the name query that would otherwise answer for it searches every set
TCGdex has and returns the oldest printing of that name, which is a card from
another era or another game.

The card names and rules text are Pokemon Trading Card Game content owned by
Nintendo, Creatures Inc. and GAME FREAK, and this project is not affiliated
with any of them. What the script writes is a local working copy for a local
index: the dump is gitignored, never committed and never republished, and the
tool quotes a card next to an attribution back to its `source_url`. The client
is polite about it: four requests in flight, a jittered backoff on a 429 or a
5xx, one listing request per set rather than one per card, and a `User-Agent`
naming the project.

### Embeddings and the index

`python -m pipeline.card_index build` writes `data/catalog/card_index/`: a
`cards.parquet` of distinct cards with their printings, a `vectors.parquet` of
passages with their vectors, and a `meta.json` naming the model, its pooling
and the index format version that built it, so an index from an older layout is
rebuilt rather than misread and an index paired with the wrong query embedder
is refused rather than searched.

A card is indexed as several passages rather than one document: an identity
line (name, stage, types, hit points) and one passage per ability, attack and
rule, each prefixed with the card's name so it is self-describing. Passages are
scored and the scores are aggregated to the card by their maximum, so a card
comes back once, with the passage that matched it. One blob per card was the
first design, and on the real corpus it ranked Dragapult ex around sixtieth
for a verbatim quote of its own attack: the name, the stage, the hit points and
the other attack diluted the text against short single-effect trainer cards.
Reprints are collapsed the same way: 2,264 Standard printings are 1,552
distinct cards, and before the collapse three copies of one Supporter could
fill a top five.

The ranking is hybrid. A small embedder is weakest on exact game vocabulary,
where "Benched", "damage counters" and "Prize cards" are the whole meaning of a
line, so the same passages are scored with BM25 over a hand-rolled index and
the two rankings are fused by reciprocal rank (k=60). RRF rather than a
weighted sum because a cosine and a BM25 score are not on the same scale. On
the real corpus a verbatim quote of Phantom Dive ranks Dragapult ex first, "put
damage counters on the bench" third; the two-word "bench damage" still ranks it
outside the top fifteen, behind cards whose entire text is a shorter sentence
about bench damage, which is what a two-word query deserves and why the agent
is told to name the card or archetype it is asking about. Model load is about
seven seconds once per process, and a search is under twenty milliseconds.

The vectors come from `BAAI/bge-small-en-v1.5`, 384 dimensions, running locally
on a CPU. Local rather than an embedding API because card text is short,
domain-specific and never changes, so the index is built once and read many
times and an API would add a key, a bill and a network hop to something that is
already fast. `all-MiniLM-L6-v2` is the alternative and is selected with
`--embedder`. A query is embedded with bge's instruction prefix and a card is
not, which is how the model was trained and is worth real accuracy: without it,
"put damage counters on the bench" does not rank the card that does exactly
that first, and with it, it does.

**Two embedders, one model.** `pipeline.query_embedder` holds both and
`docs/agent-service.md` has the numbers. Building an index uses
`SentenceTransformerEmbedder`: torch, transformers, the reference
implementation, and the definition of what the vectors mean. Answering a
question uses `OnnxEmbedder`: the same network exported to one ONNX graph, run
by ONNX Runtime, tokenized by the `tokenizers` library out of the model's own
`tokenizer.json`, with CLS pooling and L2 normalization applied in this
repository because they are not in the graph.

The reason is a cold start. On the deployed function the Python import of torch
and transformers was 105 s of every cold `GET /warm`, which is thousands of
small files paged in over image layers Lambda fetches on first touch; the
weights themselves loaded in under three seconds afterwards. So the serving
image has neither framework in it, the `serve` extra is the list that says so,
and `tests/test_serve_imports.py` fails if one comes back. The build side keeps
them, because the nightly runs on a runner with no cold start to pay and
nothing is served from it.

The two agree: `tests/test_query_embedder.py -m ml` embeds five fixture
passages and five questions with both and the smallest cosine between a pair is
0.9999999, with identical top-5 retrieval order over the fixture index. The
things that could make them disagree are all outside the graph, which is why
`meta.json` records the model *and* the pooling and `CardIndex.load` refuses a
pair that does not match rather than searching one space with the other's
vectors.

The ONNX files are three: `model.onnx`, `tokenizer.json` and an
`embedder.json` naming the model, its pooling and its width.
`scripts/export_query_embedder.py` writes them, by downloading the
`onnx/model.onnx` that `BAAI/bge-small-en-v1.5` publishes in its own repository
(133,093,490 bytes, one file, no external data) rather than converting
anything; `--export` converts with `optimum`, for a model that ships no graph.
`Dockerfile.agent` runs the script in a build stage and copies the directory
into an image that has no Hugging Face client in it at all. Locally,
`PRA_QUERY_EMBEDDER_DIR` or the default `.models/query-embedder` is where
`pipeline.config` looks.

Storage is a Parquet of vectors and a numpy dot product, not DuckDB's `vss`
extension. A few thousand passages at 384 float32 is a few megabytes and one
matrix-vector product per query: exact rather than approximate, with no recall
parameter to tune. `vss` would add an extension to
install at build time, an HNSW index whose persistence in a file-backed
database is still behind an experimental flag, and a second copy of the card
text inside the warehouse the SQL tool is deliberately restricted from reading.
At millions of rows the trade goes the other way, and the storage is one file
and one loader.

**On Linux, torch comes from the PyTorch CPU index, not from PyPI.**
`pyproject.toml` declares that index as `explicit`, so nothing resolves from it
unless asked, and `[tool.uv.sources]` asks for `torch` under
`sys_platform == 'linux'` and nothing else. The reason is that the PyPI Linux
wheel depends on the whole CUDA runtime (`cuda-toolkit`, `nvidia-cudnn-cu13`,
`nvidia-nccl-cu13`, `triton`), about 2.5 GB that `libtorch_global_deps.so`
links against and `import torch` therefore preloads, and the embedder runs on a
CPU everywhere this project runs: a laptop, a GitHub runner and a Lambda
function, none of which has a GPU. `docs/agent-service.md` has what it was
costing the deployed function. macOS and Windows keep the PyPI wheel, which is
already CPU-only there, so `uv sync` on a development machine installs exactly
what it did before; `uv.lock` carries both and the only visible difference is
that a Linux install reports its version as `2.14.1+cpu`. This now applies to
the `agent` extra alone, which is the build side and the laptop, and no longer
to anything deployed: the serving image installs `serve`, which resolves no
torch at all, so its export needs neither `--emit-index-url` nor
`--index-strategy unsafe-best-match`.

`HashingEmbedder` is the third implementation, and it is the one that stays in
`pipeline.card_index` because it is made of that module's tokenizer. It needs
no download at all: it hashes word tokens, with a five-character stem, into 256
buckets. It is not a semantic model and does not pretend to be one. It exists
so that the build, the Parquet round trip, the search, the tool and its
instrumentation all run in the fast test suite with nothing fetched from
anywhere.

### How to run it

The export is once per machine, and it is what `query`, `pipeline.agent` and
`POST /ask` read: searching needs the ONNX files, building does not.

```bash
uv run python scripts/fetch_card_text.py            # corpus from TCGdex
uv run python scripts/export_query_embedder.py      # the ONNX graph and tokenizer
uv run python -m pipeline.card_index build          # embed it, the Standard format
uv run python -m pipeline.card_index query "put damage counters on the bench" -k 5
op run --env-file=.env.op -- uv run python -m pipeline.agent \
  "how does Dragapult ex do against Gholdengo ex"
op run --env-file=.env.op -- uv run python -m pipeline.agent --repl
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "which archetype has the best record this month"}'
```

`POST /ask` returns `{answer, tool_calls, model, usage, evidence, gate_summary,
latency_ms, run_id}`. `tool_calls` is every tool the run made with the query it
was given and the number of rows that came back; `evidence` is the same run
written out to be read, every statement in full with its first ten rows, the
cards it matched, and the gate's verdict on each query, which is what the
application's "what I looked up" panel is built from. `python -m pipeline.agent
--evidence` prints the same object. The whole contract is in
docs/agent-service.md. The agent is built on the first question rather than at
startup: a service with no provider key serves `/predict` and answers `/ask`
with a 503 that says what is missing.

`build_card_index` is a task in the DAG and a stage in `python -m
pipeline.run_all`. It is skipped, with the reason logged and in the summary
table, when `data/catalog/card_text.jsonl` has not been fetched, the same way a
silver run without a card catalog is a real run with null catalog columns.

### What is logged and traced

Every tool call is a span, `agent.tool.query_marts` or
`agent.tool.lookup_cards`, carrying the length of the input and the number of
rows it returned, inside an `agent.answer` span carrying the model, the number
of tool calls and the token usage the provider reported. The SQL is on the span
as a length rather than as text: a query here is short and harmless, but a span
attribute is the wrong place to start putting model output.

Each call also increments `agent_tool_calls_total{tool=...}`, the Prometheus
counter `pipeline/telemetry.py` declared while the serving instrumentation was
being built, so the panel and the metric name were decided before the agent
existed. A JSON log line per tool call carries the tool, the row count and the
input length, and one per answer carries the model, the tool-call count and the
usage. Nothing logs the question, the answer or the SQL.

### The SQL gate

Behind the denylist, and off unless `PRA_SQL_GATE=jev`, a second opinion from
TypeSafe's Jev: one typed Choice question per statement, "is this SQL a
read-only SELECT over the marts schema that answers the user's question?",
answered with a confidence, refused below a threshold and refused on any
error. It exists for the injection the denylist cannot see, a legal `SELECT`
that reads what the question never asked for. The provider is OpenRouter's
Decisions endpoint with TypeSafe's direct API as a drop-in swap, the decision
is on the tool-call span and in a `gate` label on `agent_tool_calls_total`, and
the golden set shows gate hits and cost per run. Written up in
[sql-gate.md](sql-gate.md).

### The golden question set

`evals/golden.yaml` and `python -m pipeline.eval`, written up in full in
[evals.md](evals.md). Ten questions the fixture marts really answer, each with
the tools its answer has to call, the facts it has to contain and the claims it
must not make; the command runs them through the real agent, prints a table and
exits non-zero when anything failed.

It grades facts rather than wording. A `require` entry is a number, an
archetype, a card or a sample size, matched as a substring or as a regular
expression, because two correct answers to the same question will not share a
sentence and a set that insists on one gets ignored. `expect_tools` is a set
and an extra call is reported rather than failed, for the same reason: calling
the tool is a fact, the order is not. The `forbid` half is where the rules
above are actually enforced. `[0-9a-f]{16}`, the shape of the player token, is
forbidden on every question; two questions ask directly for a deck inclusion
rate the stock-only corpus cannot give, and forbid the claim rather than the
phrase, since the right answer is to say the number is an observation.

The prompt can be replaced wholesale by pointing
`PRA_AGENT_SYSTEM_PROMPT_FILE` at a file, which exists so the claim that the
seven rules matter can be run as an experiment:
`--prompt-override evals/broken_prompt.txt` swaps in the same job description
with the schema and the rules cut out, and the score falls. Each run is logged
to MLflow in the `agent-evals` experiment with the sha256 of the prompt that
was actually rendered, so a column renamed in `schema.yml` is visible as a
different prompt even though no Python changed.

The cadence is a trade. `.github/workflows/agent-eval.yml` scores the set
against the real provider weekly, on a push to main that touches the agent or
the questions, and on demand; it is not on every pull request, because a run is
tens of model calls and a paid check that flickers is a check people route
around. What runs on every pull request is the same set with the recorded turns
in `evals/transcript.yaml` replayed through the real graph, the real SQL gate
and real DuckDB against the fixture marts: ten out of ten, no key, no bill, and
it fails the moment a mart is renamed or the gate starts refusing a query the
set depends on.

## 7. Orchestration (in progress, Airflow and a plain runner)

Two ways to run the whole pipeline, over one list of stages. The scheduler is
`orchestration/airflow/dags/play_rough_pipeline.py`, an Airflow DAG whose every
task is a `BashOperator` calling a stage's command-line entry point; the plain
runner is `python -m pipeline.run_all`, the same list as subprocesses with no
scheduler at all. Neither holds any pipeline logic: a task that called the
stages as Python functions would be a second way to invoke them, and the second
way is the one that drifts.

What actually runs every night is the plain runner on a schedule:
`.github/workflows/nightly.yml`, documented in [nightly.md](nightly.md).

### The task graph

```
backfill -> spark_silver -> dbt_run -> dbt_test -> build_features -> train
                                           |            -> promote -> drift -> quality_gate
                                           +-> build_card_index
```

| task | command | why it is its own node |
|---|---|---|
| `backfill` | `pipeline.backfill [--source-dir]` | the only task that talks to the network, and the only one with a retry |
| `spark_silver` | `pipeline.silver` | starts a JVM, gives it back |
| `dbt_run` | `pipeline.gold --steps run` | a failed build is a different thing from a failed test |
| `dbt_test` | `pipeline.gold --steps test` | 113 data tests; retryable on its own |
| `build_features` | `pipeline.gold --steps run --select tag:ml` | the model's input as a named node |
| `train` | `pipeline.train` | one MLflow run plus its baseline |
| `promote` | `pipeline.promote --candidate latest` | the gate that can refuse a worse model |
| `drift` | `pipeline.drift` | writes the report, exits 0 whether or not it flagged |
| `build_card_index` | `pipeline.card_index build` | stage 6's retriever index; parallel because nothing waits on it, and a logged no-op when the card-text corpus has not been fetched |
| `quality_gate` | `pipeline.quality_gate` | the only task allowed to fail a run that got this far |

`--steps` and `--stage-name` exist for this graph. dbt's build and dbt's tests
are two tasks, and three invocations of one command under one run identifier
would otherwise write three `run_metrics` rows to one file and keep the last, so
each names itself (`gold_run`, `gold_test`, `gold_features`). `run_all` keeps
them as one `gold` step, because a terminal has no graph to draw.

`build_features` is a rebuild: `dbt run` already built the feature table, since
it builds the whole project. Having it as a node makes a feature change one task
to rerun rather than a whole warehouse.

### With compose

```bash
docker compose build airflow
docker compose up -d airflow mlflow            # http://localhost:8080, admin/admin
docker compose exec airflow airflow dags trigger play_rough_pipeline \
  --conf '{"source_dir": "tests/fixtures", "ingest_mode": "backfill"}'
docker compose exec airflow airflow tasks states-for-dag-run play_rough_pipeline <run id>
docker compose down                            # -v also drops Airflow's database
```

`HANDLE_HMAC_KEY` has to be set even for the fixture run: the committed games
are already anonymized, but bronze anonymizes whatever it is given and refuses
to run without a key. Any string will do locally, and compose forwards it from
the shell or from `.env`.

`Dockerfile.airflow` builds the image from `apache/airflow:2.10.5-python3.12`
and puts the project's dependencies in a virtual environment of their own at
`/opt/pipeline-venv`, exported from `uv.lock` with hashes so the image gets the
versions this repository resolved. Two environments in one image is the point:
Airflow pins large parts of the same dependency tree the pipeline uses, and what
a scheduler needs to schedule is not what a stage needs to run. The repository
itself is bind mounted at `/opt/pipeline`, so the tasks run the working tree and
only a dependency change needs a rebuild.

The image also carries a headless JRE for Spark and `libgomp1` for LightGBM,
which are the two native dependencies a pure `pip install` does not bring.

The DAG is `@daily` with `catchup=False` and `max_active_runs=1`. It is
unpaused at creation (`AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=false`), which
is local convenience rather than a recommendation, and it has one consequence
worth knowing: the first `docker compose up` also schedules the most recent
completed daily interval, and that run takes the parameter defaults, which means
S3. Without a bucket configured it fails at `backfill` and the fixture run is
the manual one beside it. Pause the DAG, or set `PRA_BUCKET`, depending on which
of the two you meant.

### Without Airflow

```bash
python -m pipeline.run_all --source-dir tests/fixtures      # the whole thing, 20 seconds
python -m pipeline.run_all --stop-after gold                # bronze, silver, gold
python -m pipeline.run_all --skip train,promote,drift       # leave the model loop out
PRA_RUN_ID=nightly-1 python -m pipeline.run_all             # or let it generate one
```

It runs each stage as `sys.executable -m pipeline.<stage>`, stops at the first
non-zero exit and exits with that code, and closes with a table of what ran,
what was skipped and how long each took. The same summary goes into a
`run_metrics` row of its own, under the stage name `run_all`, with the per-stage
durations in `extra_json`.

Stages are skipped rather than dropped, each with its reason in that table and
in the logs:

- `backfill`, when `PRA_INGEST_MODE=consumer`: the consumer is already landing
  bronze, so a backfill would read the same bucket twice.
- `train`, `promote` and `drift`, when `features_turn` holds no rows.
- `build_card_index`, when the card-text corpus has not been fetched.
- `publish`, when `PRA_INSIGHTS_TABLE` is unset, which is the normal state of a
  clone.
- `publish`, when the run was fed from `--source-dir`. Those blobs came from a
  directory on whichever machine ran the command, not from the bucket, and the
  publish replaces the insights table whole rather than adding to it, so a
  fixture run reaching it would swap the real rows for fixture ones. Pass
  `--publish` alongside `--source-dir` for the rare run that means it.

### Parameters

| parameter | default | what it does |
|---|---|---|
| `source_dir` | empty | empty reads the S3 bucket; `tests/fixtures` runs the committed games with no AWS account |
| `ingest_mode` | `backfill` | `consumer` means the event-driven ingest is already landing bronze, so the first task is a logged no-op |

`run_all` takes the same two as `--source-dir` and the `PRA_INGEST_MODE`
environment variable, plus `--publish`, `--skip`, `--stop-after`, `--data-dir`,
`--run-id` and `--summary-path`. The last one writes the stage table to a local
file as well as to the log, which is how the scheduled run gets it back off a
runner whose lake root is an `s3://` prefix.

### The run identifier

Every task exports `PRA_RUN_ID={{ run_id }}`, Airflow's own identifier for the
DAG run, so one run of the graph is one string in the UI, in every log line and
in every `run_metrics` row. `run_all` does the same with `$PRA_RUN_ID` or a
fresh one. After the fixture run above:

```
manual__2026-09-22T18-25-47-00-00-bronze_backfill.parquet
manual__2026-09-22T18-25-47-00-00-silver.parquet
manual__2026-09-22T18-25-47-00-00-gold_run.parquet
manual__2026-09-22T18-25-47-00-00-gold_test.parquet
manual__2026-09-22T18-25-47-00-00-gold_features.parquet
manual__2026-09-22T18-25-47-00-00-train.parquet
manual__2026-09-22T18-25-47-00-00-promote.parquet
manual__2026-09-22T18-25-47-00-00-drift.parquet
manual__2026-09-22T18-25-47-00-00-quality_gate.parquet
```

### The quality gate

`python -m pipeline.quality_gate` is what turns a `run_metrics` row nobody
queried into a run that goes red. It reads `mart_pipeline_health` and refuses
when any stage's last run failed, or any stage's `quarantine_rate_over_threshold`
is true, which is the 5% over the last ten runs that
`dbt_project.yml`'s `quarantine_rate_alert` defines. Exit 1 is a verdict, exit 2
is "the gate could not be run at all" (no warehouse, no mart), and the two are
different things to be woken up by.

```
stage            status    quarantine  gate
bronze_backfill  ok             0.0%  ok
drift            ok             0.0%  ok
gold_features    ok             0.0%  ok
gold_run         ok             0.0%  ok
gold_test        ok             0.0%  ok
promote          ok             0.0%  ok
silver           ok             0.0%  ok
train            ok             0.0%  ok

gate passed: 8 stage(s) healthy.
```

It judges every stage that runs before it, not only the ones that just ran: a
stage that failed last night and was not rerun is still broken this morning.
Three rows are left out. Its own, because a gate that read its own refusal back
would stay red for ever; and `publish` and `run_all`, because they write their
rows after the gate has returned, so during a run their last row is always the
previous run's. The first scheduled runs showed why that matters: a publish
refused by a missing grant one night would otherwise have refused the next
night's gate before it had a chance.

### Nothing to do is not a failure

Three stages read `features_turn`, and a warehouse that built cleanly can hold
no rows in it: a first run, or a corpus with no game carrying an archetype on
both seats. `train` then logs `no training rows` and exits 0 without registering
a version, `promote` logs `nothing to promote` when the registry is empty, and
`drift` logs `nothing to compare`. `run_all` skips all three with the reason in
its summary. On the ten committed fixtures none of that fires: `features_turn`
holds 66 rows, the model trains, and `promote` rejects it for not beating the
win-rate baseline, which is the gate working.

### Still to come

Sensors rather than a clock: now that the consumer is live and landing bronze
continuously, a graph that starts when there are new partitions is a better
shape than one that starts at a fixed hour and finds nothing. The scheduled
workflow in [nightly.md](nightly.md) is the clock version, and it is what runs
today. AWS Step Functions remains the managed alternative (stage list,
stretch).

## 8. Publish (in progress)

Input: the marts in `$PIPELINE_DATA_DIR/warehouse/meta.duckdb`. Output: rows in
the application's DynamoDB table, named by `PRA_INSIGHTS_TABLE`
(`pra-<stage>-insights`, one per deployment stage). This is the stage that closes the
loop: everything before it lands in a DuckDB file on whichever machine ran the
pipeline, and the application cannot read that, so the last step copies the
public-safe marts into the store it already reads on every request.

Command: `python -m pipeline.publish` (`--warehouse`, `--table`, `--dry-run`,
`--tracking-uri`). Nobody runs it as routine: it is the last stage of the
nightly, after `quality_gate` rather than before it, because a run whose gate
refused must not put its numbers in front of the application's readers and a
failed stage stops what follows it. A publish by hand is a recovery, and the
way to do it is to rerun the whole nightly (`gh workflow run nightly.yml -f
stage=prod`) so the gate is in front of it as usual. The runner
and the DAG both skip it with a logged reason when `PRA_INSIGHTS_TABLE` is
unset, which is the normal state of a clone. The runner skips it a second way:
a run given `--source-dir` read its blobs from a directory rather than from the
bucket, and does not publish unless `--publish` is passed as well.

### The contract

One table, keys `pk` (S) and `sk` (S), four partition-key families over it.
Attribute names are camelCase, because the application is TypeScript and reads
these rows straight into its models; the warehouse's snake_case stops here.

| pk | sk | attributes |
| --- | --- | --- |
| `MATCHUP` | `<archetypeKey>#<opponentArchetypeKey>` | `archetypeKey`, `archetypeName`, `opponentArchetypeKey`, `opponentArchetypeName`, `games`, `wins`, `losses`, `ties`, `winRate`, `minGamesMet`, `runId`, `publishedAt` |
| `WEEKLY#<archetypeKey>` | `<isoYear>-W<isoWeek>` | `archetypeKey`, `archetypeName`, `isoYear`, `isoWeek`, `weekStart`, `games`, `wins`, `losses`, `ties`, `winRate`, `shareOfWeek`, `runId`, `publishedAt` |
| `ARCHETYPE` | `<archetypeKey>` | `archetypeKey`, `archetypeName`, `games`, `wins`, `losses`, `winRate`, `runId`, `publishedAt` |
| `META` | `LATEST` | `runId`, `publishedAt`, `modelName`, `modelVersion`, `modelAlias`, `gamesTotal`, `matchupRows`, `weeklyRows`, `archetypeRows`, `sourceCommit`, `minGames` |

Four things about the values are worth stating rather than discovering.

**Numbers are `Decimal`.** DynamoDB has one numeric type and boto3 refuses a
float outright rather than rounding one silently, so every count and every rate
is converted through `str()` on the way in.

**Rates are percentages, 0 to 100, rounded to two places.** `winRate` and
`shareOfWeek` are `55.56`, not `0.5556`. The marts hold fractions, because a
division produces one; the application's wire format is percentages, and this
boundary is where the scaling happens so that it happens once.

**A rate that does not exist is left out.** A matchup with no decided game has
no win rate, and the attribute is absent rather than null, so
`attribute_exists(winRate)` is a filter the application can use. The same goes
for `shareOfWeek`, for `sourceCommit` outside a checkout, and for `minGames`
when the dbt project cannot be read.

**`modelVersion` and `modelAlias` are always present, and `null` until a model
is promoted.** Every META row carries both keys, on every stage: `null` when no
model version holds the `production` alias yet, the version string and
`"production"` once one does. Unlike a missing rate, the two are part of the
row's shape from the first publish, so a reader can tell "no model promoted"
apart from "this field was never written" instead of treating an absent key as
either.

**The ISO week is zero padded.** `2026-W09`, not `2026-W9`, because without the
padding week 9 sorts after week 10 and a `between` over a range of weeks
silently returns the wrong set.

The archetype family is aggregated here, over the `mart_matchups` rows for the
archetype, rather than read from a mart of its own. Summing them counts every
seat the archetype held against a known opponent, mirrors included once per
seat, so an archetype total and the matchup cells under it agree.
`mart_archetype_weekly` would have given a different number (it keeps seats
whose opponent archetype is unknown) and `dim_archetype.games_played` a third
one (it counts before the `excluded_from_stats` filter). `gamesTotal` in `META`
is a fourth question, deliberately: distinct games in `fct_game_side` that are
not excluded, which is what "built from 128 games" means on a page.

### Refresh semantics

A publish is a refresh, not a merge. Every row is written with the new `runId`,
in `BatchWriteItem` chunks of 25 with the unprocessed items resent under a
back-off, then `META`, and only then are the rows whose `runId` is an older one
deleted. That ordering is the point: at no moment is the table missing a row it
had before, so a reader mid-publish sees the old row or the new one and never a
gap. Deleting first would have shown an empty matchup matrix for as long as the
write took.

The sweep queries each of `MATCHUP`, `ARCHETYPE` and `META`, and for the weekly
family it queries the archetype partitions it just published plus any others
found by a `Scan` filtered on `begins_with(pk, "WEEKLY#")`. The Scan is there
for exactly the case deletion exists for, an archetype that has dropped out of
the corpus and would otherwise keep its partition for ever.

**The scale caveat.** That Scan reads the whole table. It is bounded by the
table's size, which at a few thousand rows is nothing and at a few million
would be both slow and expensive. The replacement when that day comes is a
global secondary index on `runId`: query it for the previous run's rows and
delete those, instead of scanning for all of them and comparing.

### Permissions

The code names no profile and no role: it is boto3's default credential chain,
the same as every other stage that talks to AWS. What changed for this stage is
on the other side of that chain, in the application's account. The role this
pipeline assumes was read-only, and now carries write (`PutItem`, `DeleteItem`,
`BatchWriteItem`, `Query`, `Scan`) on this one table and on nothing else. The
S3 bucket it reads is still read-only, and no other table in the account is
reachable.

### How the application reads it

Every access pattern is a query on the key, and none of them is a scan:

- the matchup matrix, or one row of it: `Query pk = "MATCHUP"`, optionally with
  `begins_with(sk, "<archetypeKey>#")` for one archetype's row of the matrix;
- one archetype over time: `Query pk = "WEEKLY#<archetypeKey>"`, with
  `sk between "2026-W01" and "2026-W12"` for a range of weeks;
- the leaderboard: `Query pk = "ARCHETYPE"`;
- the freshness banner, and the model version behind any prediction shown
  beside these numbers: `GetItem pk = "META", sk = "LATEST"`.

`publishedAt` and `runId` are on every row, so a page can say when the numbers
are from, and a row that looks wrong can be traced back to the run that wrote
it, in this repository's `run_metrics` and in the Airflow UI, by one identifier.

### What a run looks like

```
publish to pra-<stage>-insights under run 5890ea06ffab4256 at 2026-09-22T20:52:54Z

kind       items
matchup      180
weekly       109
archetype     80
meta           1

read 289 mart row(s), wrote 370 item(s), deleted 370 stale row(s)
```

`--dry-run` builds every item, prints those counts, one sample item per kind and
the meta row, and writes nothing. It is the way to see what a publish would do
against a warehouse without an account that can write, and it is what the
`run_metrics` row of a dry run reports as `rows_out = 0`.

### Not published

Only the public-safe subset, which is the marts above. Nothing per player
reaches this stage: `mart_player_summary` is not read, no token, handle or user
id is in any item, and the only free text in the table is an archetype name.
See [data-handling.md](data-handling.md).

## Ops: logging, run metrics, traces and dashboards

Every stage writes the same two things: a structured log to standard error and
one `run_metrics` row to the lake. Both come from `pipeline/observability.py`,
which is the only module every other stage imports, and neither needs a server.

**The log.** One JSON object per line, on standard error:

```json
{"ts": "2026-09-22T17:41:02.134612+00:00", "level": "INFO", "logger": "pipeline.observability",
 "stage": "silver", "run_id": "smoke-1", "msg": "stage complete", "duration_s": 12.31,
 "rows_in": 128, "rows_out": 128, "rows_quarantined": 0, "status": "ok"}
```

`ts`, `level`, `logger`, `stage`, `run_id` and `msg` are always there; anything
passed as `extra=` is merged in beside them, and an exception adds `exc_type`,
`exc_message` and `stack`. Set `PRA_LOG_FORMAT=console` (or run in a terminal,
which is the default when standard error is a teletype) and the same records
render as one compact line each, `HH:MM:SS LEVEL [stage run_id] msg key=value`.
`PRA_LOG_FORMAT=json` forces the machine form. Standard library `logging` only:
a `logging.Filter` puts the run identifier and the stage on every record,
including the ones PySpark, MLflow and boto3 emit, so nothing has to be threaded
through a call.

Standard output is kept for the command's own result. A stage that used to print
a summary block now calls `emit_summary`, which logs the summary as one record
with its fields and writes the readable block to standard output, so
`python -m pipeline.backfill 2>/dev/null` is still a table a person reads and
`python -m pipeline.silver 2>&1 >/dev/null | jq` is still parseable.

**The run identifier.** `PRA_RUN_ID` when it is set, which is how the
orchestrator will give one whole DAG run a single identifier, and sixteen random
hex characters otherwise. It is on every log line and in every `run_metrics`
row, so "what did the 06:00 run do" is one filter on either.

**The row.** `stage_run` wraps the body of each stage, times it, and writes
exactly one Parquet file to `$PIPELINE_DATA_DIR/lake/run_metrics/`, named
`<run id>-<stage>.parquet`:

| column | meaning |
|---|---|
| `run_id`, `stage` | the grain |
| `started_at`, `finished_at`, `duration_s` | when and how long |
| `rows_in`, `rows_out`, `rows_quarantined` | what the stage read, wrote and refused |
| `status`, `error` | `ok` or `failed`, and the exception class and message |
| `extra_json` | whatever else the stage recorded, as JSON |
| `git_commit`, `hostname` | which code, which machine |

One file per run rather than one appended table, because two stages of the same
run finish at unpredictable times and a writer that rewrites a shared file loses
one of them. The schema is pinned (docs/schema.md section 11). A stage that
raises still writes its row, with `status = "failed"`, before the exception
continues; a stage whose bookkeeping fails logs a warning and reports its real
result, because a run that did its work must not be marked broken by its own
telemetry.

Stage names, one per command: `bronze_backfill`, `consume`, `silver`, `gold`,
`train`, `promote`, `drift`, `quality_gate` and `run_all`, plus
`refresh_fixtures` and `fetch_catalog` for the two maintenance scripts. Under
the Airflow DAG the gold step is three tasks and names itself `gold_run`,
`gold_test` and `gold_features`, for the reason section 7 gives.

Two commands do not fit one row per run, and each bends it a different way.
`consume` never finishes, so its unit is one receive batch (section 1b): a
batch that came back with messages is timed and written, an idle long poll
writes nothing, and the row a consumer that has been up for days holds is the
one for its most recent batch. `serve` is a process rather than a run at all,
so it configures logging at startup, logs one record per request (method, path,
status, duration in milliseconds, loaded model version) from a middleware, and
writes no `run_metrics` row.

**Reading it back.** Two dbt models under `models/ops/`, both views over the
Parquet so a stage that finished a second ago is in the next query:

- `run_metrics`: every row, one per stage per run. Guarded against an empty
  directory, so a fresh clone builds before it has ever run a stage.
- `mart_pipeline_health`: one row per stage, carrying the last run's status,
  duration and counts, and the trend over the last ten runs, including
  `quarantine_rate` and the `quarantine_rate_over_threshold` flag.

```sql
-- which stage is throwing rows away
select stage, last_status, last_duration_s, last_rows_in, last_rows_out,
       quarantine_rate, quarantine_rate_over_threshold
from mart_pipeline_health
order by quarantine_rate desc;
```

The quarantine rate is rows quarantined over rows read across the window, not
the mean of the per-run rates: a run that read three objects and rejected one is
33%, and averaging that against a run of ten thousand would let a tiny run shout
down a large one.

**The alert.** `quarantine_rate_over_threshold` is true when a stage has
quarantined more than 5% of what it read over its last ten runs
(`quarantine_rate_alert` in `dbt/dbt_project.yml`), and `python -m
pipeline.quality_gate` is what acts on it: the last task of the DAG and the last
step of `run_all`, it reads this column and `last_status` and exits non-zero, so
a failed stage's row becomes a red run rather than a row nobody queried. Nothing
pages yet, because nothing is on call; the run going red is where a notifier
would hang. See section 7.

### Traces and metrics on the serving API

Everything above is built for a batch stage, where the unit of work is a run
that starts, does something and ends. The serving process has no run to close,
so the same three questions need three different instruments, and it carries all
of them. `pipeline/telemetry.py` is the module; `pipeline/observability.py`
stays exactly as it is, for the stages.

| pillar | where it is | what it answers | what it cannot |
|---|---|---|---|
| logs | JSON lines on standard error, plus `run_metrics` for the stages | what happened, in order, with the detail attached: this request, this body, this exception | aggregate. Counting anything means reading every line |
| metrics | Prometheus, scraped off `GET /metrics` | how it is doing overall: rate, error ratio, the latency distribution | say anything about one request. The cost is fixed, and that is the trade |
| traces | OpenTelemetry over OTLP to a collector, then Jaeger | where one particular request spent its time, span by span | tell you it is happening at all. A trace is found because a metric or a log sent you looking |

The order matters as much as the table: a metric says the p95 moved, a trace
says the time is in the model rather than around it, and the log line for that
request says which model version and which archetypes. Each one hands off to the
next, which is why all three carry the model version.

**Traces.** The FastAPI instrumentation emits one span per request, and
`/predict` opens a child span, `predict.inference`, around the model call and
nothing else, with `model.version`, `model.alias` and
`features.unknown_archetypes` on it. A span over the whole handler would just
restate the HTTP span; the question worth a second span is whether the time is
the model or the code around it, and that needs the two side by side.
`/health` and `/metrics` are excluded, because a liveness probe and a scrape
every fifteen seconds would be almost the entire trace store and neither has
ever been worth reading.

Exporting is optional and silent about it. `OTEL_EXPORTER_OTLP_ENDPOINT` names
the collector; unset, or naming a host that does not resolve, the tracer
provider is a genuine no-op and the service starts exactly as it did before. The
resolve check is not decoration: left to the exporter, a service started without
the collector logs a retry warning per batch for as long as it runs, which is a
log full of the telemetry failing to leave.

**Metrics.** Six families, on a registry the application owns rather than the
process-global one, so two applications in one test process do not collide:

| metric | labels | why |
|---|---|---|
| `http_requests_total` | `method`, `route`, `status` | rate and error ratio, per endpoint |
| `http_request_duration_seconds` | `method`, `route` | the latency histogram a p95 is read from |
| `model_inference_duration_seconds` | `model_version` | the model call alone, so a slow promotion is visible as a slow promotion |
| `model_predictions_total` | `model_version`, `unknown_archetype` | how much of the traffic the model has never seen the decks for |
| `model_info` | `name`, `version`, `alias` | always 1; which version is answering, cleared and reset on `/reload` |
| `agent_tool_calls_total` | `tool` | how much of the agent's work is queries and how much is card lookups; one series per tool, and the tool names are a closed set |

The labels are bounded deliberately. `route` is the matched template, `/predict`
and not the request path, so a service that is scanned for `/wp-admin.php` gets
one `unmatched` series instead of one per probe; `unknown_archetype` is a yes or
no rather than the archetype name, which is unbounded by definition, because the
name is already on the span and in the request log where an unbounded value
costs nothing. The HTTP counter and histogram are observed by the request-log
middleware that was already there, rather than by a second middleware: two of
them would time two slightly different things and disagree about latency by
whatever sits between them.

**Running it.** Four containers behind a compose profile, so the default
`docker compose up -d mlflow predict` is still two:

```bash
docker compose --profile observability up -d --build predict grafana
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '...'
open http://localhost:16686    # Jaeger: the trace, with its inference span
open http://localhost:3000     # Grafana: Play Rough / Predict service
open http://localhost:9090     # Prometheus, for writing the query first
docker compose --profile observability down
```

The collector (`orchestration/observability/otel-collector.yaml`) receives OTLP
over HTTP on 4318 and fans out to a debug log and to Jaeger. It is a hop the
service does not strictly need, and it is there for what it buys: the service
knows one address and one protocol forever, and moving the traces to a hosted
backend is three lines of collector configuration rather than a redeploy of the
thing being traced. Prometheus scrapes `predict:8000/metrics` every fifteen
seconds; Grafana has one provisioned dashboard, `Predict service`, with four
panels, request rate by route, p95 latency by route, inference p95 by model
version and predictions by version. Both the datasource and the dashboard are
provisioned from files under `orchestration/observability/`, mounted read only,
so the panels are in the repository rather than in a volume nobody backs up.

**The demo stub.** `PRA_SERVE_STUB_MODEL=1` makes the service answer from a
hand-written logistic on the prize lead instead of the registry. It exists for
exactly one reason: the promotion gate refuses a candidate that does not beat
the archetype win-rate baseline, on this corpus nothing does, and so the
ordinary state of a local registry is that nothing holds the `production` alias
and `/predict` is a 503. Demonstrating a trace with an inference span in it then
means either a registry fixture nobody maintains or moving the alias by hand,
and moving the alias by hand is much worse than an obviously fake predictor. It
reports `stub` as its version and its alias, in the response body, in the log
and on every metric label, and every archetype comes back in
`unknown_archetypes`, because nothing trained it. It is never a model.

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
- Publish sweep: the stale-row sweep finds orphaned weekly partitions with a
  `Scan`, which is bounded by the table and fine at this size. A global
  secondary index on `runId` replaces it when the table stops being small.

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
