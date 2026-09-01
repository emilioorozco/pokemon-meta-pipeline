# pokemon-meta-pipeline

A batch data-engineering pipeline that turns a corpus of Pokémon TCG AI-battle
replays (from the Kaggle "Pokémon TCG AI Battle" competition, now finished) into a
metagame analytics warehouse: archetype win rates, card inclusion rates, and a
matchup win-rate matrix.

## Architecture

```
source corpus (external, read-only)          this repo
┌──────────────────────────────┐   ┌─────────────────────────────────────────┐
│ replay JSONs (~1,240 games)  │   │ 1. ingest  → partitioned Parquet lake   │
│ EN_Card_Data.csv             │──▶│ 2. PySpark → game/deck/event tables     │
│ (path via SOURCE_DATA_DIR)   │   │ 3. dbt+DuckDB → star schema + marts     │
└──────────────────────────────┘   │ 4. Airflow → orchestrates 1–3           │
                                   └─────────────────────────────────────────┘
```

- **Lake**: partitioned Parquet under `data/lake/` (gitignored)
- **Warehouse**: DuckDB at `data/warehouse/meta.duckdb` (gitignored)
- **Cloud**: optional flag-gated S3 output paths — never a hard dependency

## Source data

The raw corpus is **not** in this repo. Point the pipeline at it:

```bash
cp .env.example .env   # then edit SOURCE_DATA_DIR
```

Expected layout under `SOURCE_DATA_DIR`:

```
data/replays/corpus/episode-*.json    # Kaggle episode replays, batch 1
data/replays/corpus2/episode-*.json   # batch 2
data/EN_Card_Data.csv                 # card reference data
```

## Layout

```
pipeline/   Python package: ingest + Spark jobs + shared config
dbt/        dbt project (DuckDB) — staging models, star schema, marts
dags/       Airflow DAG definitions
tests/      unit tests for transforms + data-quality checks
data/       local lake + warehouse output (gitignored)
```

## Status

Scaffold + discovery done. Stage 1 (ingest) in progress.
