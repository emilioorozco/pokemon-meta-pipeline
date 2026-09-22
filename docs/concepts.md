# Data-engineering concepts used in this project

Beginner-level explanations, each tied to this project's data so it is concrete.
Skim now, come back when a term shows up in a stage doc.

## ETL / ELT: the shape of every pipeline

Extract data from a source, Transform it, Load it somewhere queryable (ETL).
Modern practice is usually ELT: load raw-ish data into the warehouse first, then
transform it inside the warehouse with SQL (that is dbt's whole job). We do a
hybrid: Python and Spark for the parts SQL is bad at (flattening nested JSON,
text rewrites for anonymization), then ELT-style SQL modeling in DuckDB for
everything after.

## Data lake, and the bronze / silver / gold layers

A data lake is just files in folders (here: Parquet under `data/lake/`),
organized in layers by how refined they are:

- Bronze: source data, faithfully captured, ugly allowed. Ours: one row per
  game, one row per game seat, one row per parsed log entry, all still carrying
  the upstream field names.
- Silver: cleaned, typed, validated, enriched. Ours: cards exploded to one row
  each and joined to the card catalog, archetype names resolved through the
  alias table, derived features attached.
- Gold: business-ready answers. Ours: the star schema and the win-rate,
  matchup and cards-seen marts.

The point of layers: when something looks wrong in gold, you can trace it back
through silver to bronze without re-reading the source blobs. Reprocessing is
cheap because each layer only depends on the one before it.

## Parquet and columnar storage

A CSV or JSON file stores data row by row: to compute the average of one column
you still read every byte of every row. Parquet stores data column by column,
with types and compression. "Average game length" reads only the `turn_count`
column. It is also self-describing (the schema travels with the file). This is
why every serious analytics system speaks Parquet.

## Partitioning

Splitting a dataset into subfolders by a key, for example
`lake/bronze/game/play_date=2026-09-14/`. Queries that filter on the key skip
whole folders ("partition pruning"). The classic key is date, and that is ours:
`play_date`, taken from the game's `playedAt` timestamp. Re-running a day
rewrites exactly that day's folder and nothing else.

## Data warehouse and DuckDB

A database optimized for analytics (big scans, aggregations) rather than
transactions (many small updates). Snowflake, BigQuery and Redshift are the
cloud ones. DuckDB gives you the same experience as one local file; it even
queries Parquet files in place. Our warehouse is `data/warehouse/meta.duckdb`.

## Star schema: facts and dimensions

The classic warehouse design (also called dimensional modeling, from Kimball):

- A fact table holds events or measurements at a fixed grain (what one row
  means). Ours: `fact_game_seat`, one row per (game, seat) with the outcome and
  the per-side counters. Facts are long and skinny: mostly foreign keys plus
  numbers.
- Dimension tables hold descriptions you slice by: `dim_player`,
  `dim_archetype`, `dim_season`, `dim_card`. Dimensions are short and wide.

Drawn with the fact in the middle and dimensions around it, it looks like a
star. Why bother? Any question becomes the same query shape: filter or group by
dimension attributes, aggregate fact measures. "Win rate by archetype" is a join
from the fact to `dim_archetype`, a group by, and an average.

## Grain

The single most important design decision for a fact table: what does one row
represent? Get it wrong and every count is subtly wrong. We chose one row per
(game, seat), so each game contributes exactly two rows and a win rate is just
`AVG(is_winner)`. A "matchup" view is derived by pairing the two rows of a game,
not stored at a different grain.

## Spark (PySpark)

A compute engine that splits a job across many cores or machines: you write
dataframe transformations ("explode each game's cards into one row per card,
then aggregate"), Spark builds a plan and distributes it. On today's data one
laptop is more than enough; we use Spark anyway for the pattern: the exact same
job would run on a cluster over terabytes by changing config, not code. Key
mental model: Spark is lazy. Transformations build a plan; nothing executes
until an action (write, count) forces it.

## dbt

"SQL models as software." Each table in the warehouse is one versioned `SELECT`
file; dbt figures out dependency order, runs them, and runs tests you declare
(unique, not-null, accepted values) after every build. It replaces the pile of
run-in-the-right-order SQL scripts every analytics team eventually regrets.

## Orchestration and DAGs (Airflow)

A pipeline is a DAG, a directed acyclic graph of tasks: ingest, then spark,
then dbt, then checks. An orchestrator runs the DAG on a schedule, retries what
fails, skips what is already done, and records history. Airflow is the most
common one; its unit is a Python file defining tasks and their `>>`
dependencies. It is the grown-up alternative to a bash script that runs five
things and dies silently on the third.

## Idempotency

A stage is idempotent if running it twice produces the same result as once,
usually by overwriting a partition rather than appending. Non-idempotent
pipelines double-count data every time someone re-runs a failed day. Every stage
we build overwrites its output partition; nothing appends.

## Data contract

An agreed, machine-checkable description of the data one system hands to
another. Here the upstream web application owns the contract as Zod schemas and
validates every blob it writes; this pipeline validates the same shape on read
(against a JSON Schema exported from those Zod schemas) and quarantines anything
that fails. The contract is versioned (`schemaVersion`) so readers can accept
old and new shapes side by side.

## Data quality checks

Assertions that run inside the pipeline: every game has exactly two seat rows,
exactly one seat per game is the winner (or none, for unknown results), card
names join to the card dimension, win rates are in [0, 1], excluded games never
reach a mart. Failing checks halt the DAG before bad data reaches gold. In our
stack: dbt tests plus a few Python-side checks at ingest.

## Anonymization (keyed hashing)

Player handles are personal data and appear in free text throughout the source.
Bronze replaces each handle with an HMAC (hash-based message authentication
code): a hash keyed by a secret, so the same handle always maps to the same
token within a run, tokens are stable across runs with the same key, and nobody
without the key can reverse or brute-force them. Because handles occur inside
sentences, this is a text rewrite pass over every string field, not a single
column replacement.

## History

These concepts were first written up for an earlier version of this pipeline
that ingested a public corpus of AI-versus-AI replays. The examples have been
reframed for the current source, the parsed battle-log blobs produced by the
analytics web application. The stage history is in [stages.md](stages.md).
