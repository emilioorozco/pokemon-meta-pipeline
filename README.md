# pokemon-meta-pipeline

![CI](https://github.com/emilioorozco/pokemon-meta-pipeline/actions/workflows/ci.yml/badge.svg)

The analytics data platform for Play Rough Analytics, a web application where
players upload Pokemon Trading Card Game (TCG) Live battle logs. The application
parses each log into a per-game JSON blob and stores it in Amazon Simple Storage
Service (S3). This pipeline reads those blobs, anonymizes player handles, and
builds a metagame warehouse: archetype win rates, matchup matrix, cards seen,
plus a win-probability model and an agent that answers questions over the marts.
Results are published back to the application.

## Stages

```
S3 parsed/{userId}/{gameId}.json
  -> bronze   validate, anonymize, flatten to Parquet (Python)
  -> silver   typed tables, cards exploded, archetypes resolved (PySpark)
  -> gold     star schema and marts (dbt on DuckDB)
  -> model    win-probability model tracked in MLflow, served by FastAPI
  -> agent    LangChain agent over the gold marts and card text
  -> airflow  DAG (directed acyclic graph) running bronze, silver, gold, retrain
  -> publish  marts written back to a prefix the application reads
```

Each stage is one command and one DAG task; inputs and outputs are files or
tables, and every partition write is idempotent.

Bronze backfill: `python -m pipeline.backfill`

Details and status per stage
are in [docs/stages.md](docs/stages.md); what the source contains and why the
schema looks the way it does is in [docs/discovery.md](docs/discovery.md).

## Source data

The source is the application's private S3 bucket. Nothing in it is committed
here. Configure access through environment variables (see `.env.example`):

```
PRA_BUCKET        S3 bucket that holds parsed/{userId}/{gameId}.json
PRA_PREFIX        key prefix to read, default parsed/
AWS_REGION        bucket region, default us-west-2
AWS_PROFILE       optional named AWS profile
HANDLE_HMAC_KEY   secret used to anonymize player handles; never commit it
PIPELINE_DATA_DIR where the lake and warehouse are written, default ./data
```

One backfill run reads every blob under the prefix, lands the valid ones in
bronze and quarantines the rest under `data/lake/quarantine/`:

```
op run --env-file=.env.op -- uv run python -m pipeline.backfill   # key from 1Password
uv run python -m pipeline.backfill                                # env already populated
```

`--dry-run` reads and validates without writing anything; `--limit N` stops
after N blobs.

Data handling, anonymization and deletion: see [docs/data-handling.md](docs/data-handling.md).

## Layout

```
pipeline/          Python package: bronze ingest, Spark jobs, shared config
pipeline/legacy/   deprecated first source (Kaggle corpus), kept for reference
dbt/               dbt project (DuckDB): staging, star schema, marts
dags/              Airflow DAG definitions
tests/             unit tests for transforms and data-quality checks
docs/              concepts, discovery, schema, stage plan, decision records
data/              local lake and warehouse output (gitignored)
```

## Status

Stage 0 done: repo, CI, docs. Bronze ingest from S3 in progress. Everything
else planned; see [docs/stages.md](docs/stages.md).

The original stage 1, built against a finished Kaggle competition's replay
corpus, is deprecated and lives under `pipeline/legacy/kaggle/`. It is not part
of any stage; see [docs/adr/0001-deprecate-kaggle-source.md](docs/adr/0001-deprecate-kaggle-source.md).
