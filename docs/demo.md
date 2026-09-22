# Five-minute demo

An anchor for a live walk-through. The order below goes from a clone to a
queryable lake to a deliberate failure to green CI, and every command proves
exactly one thing, so a step that misbehaves can be named, skipped and left
behind without losing the thread. Read the "Say" lines once before the session
and then work from the commands alone; they are meant to be run top to bottom,
not read out.

**Before you start**

- A fresh terminal in an empty scratch directory. Step 1 creates the repo and
  every later command runs from inside it.
- Nothing else running: no editor watcher, no other `uv` process, no VPN prompt
  waiting for a click.
- `uv` and `git` on the path, `gh` too if you plan to run step 7 in the
  terminal rather than in a browser tab.
- Start a timer. The whole thing is five minutes when nothing goes wrong and
  the point of the timer is to notice when something has.

## Today (about five minutes)

### 1. Clone and install

```
git clone https://github.com/emilioorozco/pokemon-meta-pipeline.git
cd pokemon-meta-pipeline
uv sync --group dev
```

**What it proves:** the project installs from a lockfile into an isolated
environment, with no global state.

**Say:** Dependencies are pinned in `uv.lock`, so this resolves to the same
versions CI uses. It builds a virtual environment in the project directory and
touches nothing else on the machine. Everything after this runs through
`uv run`, so there is no activate step and no ambient Python.

### 2. Run the tests

```
uv run pytest -q
```

**What it proves:** the behaviour claimed in the rest of the demo is already
asserted, before a single thing is run by hand.

**Say:** Two hundred plus tests, about ten seconds on a cold clone. The contract
models are compared field by field against the committed JSON Schema file, so a
drift between the two is a test failure rather than a surprise at ingest time.
The S3 reads run against moto, an in-process fake, which means the whole ingest
path is exercised with no AWS account and no network.

### 3. Run the bronze ingest over the committed fixtures

```
HANDLE_HMAC_KEY=$(openssl rand -hex 32) uv run python -m pipeline.backfill \
  --source-dir tests/fixtures \
  --bronze-dir /tmp/demo/bronze \
  --quarantine-dir /tmp/demo/quarantine
```

**What it proves:** the real ingest stage runs end to end on ten real games with
no bucket, no credentials and no network.

**Say:** `--source-dir` swaps the S3 listing for a directory walk and changes
nothing else: same validation against the contract, same anonymization, same
routing to bronze or quarantine. The key is thrown away when this shell exits,
which is the point: handles are replaced by keyed HMAC tokens, and without the
key the tokens join to nothing. The summary at the end is the whole audit of the
run: read, landed, quarantined by reason, and rows per partition.

### 4. Look at what landed

```
find /tmp/demo/bronze -maxdepth 1 | sort
```

**What it proves:** the output is a hive-partitioned Parquet dataset on disk,
keyed by the day the game was played.

**Say:** One directory per `play_date`, taken from the game's own timestamp and
not from when the file was uploaded. A run replaces each day it touches whole
rather than appending, so running the same games twice gives the same row count.
That is what makes a re-run after a key rotation or a bug fix safe.

### 5. Query it with DuckDB

```
uv run python - <<'PY'
import duckdb
games = "read_parquet('/tmp/demo/bronze/**/*.parquet', hive_partitioning=true)"
duckdb.sql(f"SELECT play_date, count(*) AS games FROM {games} GROUP BY 1 ORDER BY 1").show()
duckdb.sql(
    f"SELECT summary.opponent_archetype AS opponent, count(*) AS games "
    f"FROM {games} GROUP BY 1 ORDER BY games DESC, opponent LIMIT 5"
).show()
PY
```

**What it proves:** the lake is queryable as one table, nested fields included,
with no server to start.

**Say:** DuckDB reads the Parquet files in place, columnar, with the partition
column recovered from the directory names. The second query reaches into
`summary`, which is a struct column: bronze keeps the blob's nesting instead of
flattening it, so a contract change shows up as a schema difference rather than
as a column that quietly went missing. Ten games means every opponent count is
one, which is the honest scale of the fixture set; the same two queries are what
runs over the full lake.

### 6. Break one on purpose

```
mkdir -p /tmp/demo/broken
sed '1,/"kind": "attack"/ s/"kind": "attack"/"kind": "teleport"/' \
  tests/fixtures/game-01-a35e2e68.json > /tmp/demo/broken/unknown-kind.json
head -c 400 tests/fixtures/game-02-0d213f9a.json > /tmp/demo/broken/truncated.json
HANDLE_HMAC_KEY=$(openssl rand -hex 32) uv run python -m pipeline.backfill \
  --source-dir /tmp/demo/broken \
  --bronze-dir /tmp/demo/bronze \
  --quarantine-dir /tmp/demo/quarantine
cat /tmp/demo/quarantine/*/*.meta.json
```

**What it proves:** bad input is routed and explained rather than dropped or
allowed to kill the run.

**Say:** One file now claims an action kind the contract does not know, and the
other is cut off mid-JSON. Both are quarantined by reason, with the body kept
exactly as received next to a sidecar that names the file, the reason and the
failing field path. The sidecar is the part that gets pasted into a bug report,
so it carries no handle and no blob content, only paths and messages. Dropping
the row silently would have been the cheaper option and the one that loses data.

### 7. Show CI

```
gh run list --limit 3
```

(Or open the Actions tab in a browser and point at the last green run.)

**What it proves:** the same checks run on every pull request, not just on this
machine.

**Say:** Four gates on every pull request: ruff for lint, ruff for formatting,
mypy with untyped definitions banned across the package, the tests and scripts,
and pytest with a coverage floor that fails the build. The contract test from
step 2 runs here too, so the committed schema file and the models cannot drift
apart on a branch. Two Python versions in the matrix.

### 8. Close

```
open docs/data-handling.md
```

**What it proves:** the handling rules are written down, not remembered.

**Say:** Two things to leave you with. `docs/data-handling.md` is where the
anonymization and consent rules live, including why an opponent's decklist is
kept when it is in the blob and why a quarantine sidecar never is. And the
architecture diagram in the README is the map of where this goes next: the
stages after bronze are planned in `docs/stages.md` with the same level of
detail.

## When something breaks live

1. Say what the step was supposed to prove, then move on. The demo is a
   sequence of claims, and a claim you cannot show right now is still a claim
   you can state.
2. Never debug for more than one minute in front of people. Note it, keep
   going, fix it afterwards.
3. The tests and the CI badge are the fallback proof. Step 2 and step 7 cover
   almost everything the other steps show, and both are one command.

## Later stages (outside the timed sequence)

- **Spark run** (silver): runs today, but out of the timed sequence because it
  needs Java and spends five seconds starting a Java Virtual Machine. After
  step 5: `uv run python -m pipeline.silver --bronze-dir /tmp/demo/bronze
  --silver-dir /tmp/demo/lake/silver --catalog tests/catalog.json`, then query
  `/tmp/demo/lake/silver/game_sides/**/*.parquet` with the DuckDB snippet from
  step 5. The output directory is `lake/silver` because the gold step below
  reads `$PIPELINE_DATA_DIR/lake/silver`, which is the layout a real run
  writes. What it proves: the grain change, the archetype alias map and the
  reconciliation that fails the run when the counts do not add up.
- **dbt docs** (gold): runs today, and also out of the timed sequence because
  it needs the Spark run above to have produced silver first. After the Spark
  run: `PIPELINE_DATA_DIR=/tmp/demo uv run python -m pipeline.gold`, which
  builds the star schema and the marts into `/tmp/demo/warehouse/meta.duckdb`
  and then runs the 105 dbt tests, followed by
  `uv run dbt docs generate --project-dir dbt --profiles-dir dbt` and
  `uv run dbt docs serve --project-dir dbt --profiles-dir dbt` for the lineage
  graph and the column descriptions. Query the result with the DuckDB snippet
  from step 5, pointed at the warehouse file instead of the Parquet glob:
  `select archetype_name, opponent_archetype_name, games, win_rate from
  mart_matchups order by games desc limit 5`. What it proves: the grain change
  becomes a star schema, every model and column is documented, and the tests
  (keys, relationships, accepted values, two sides per game, matchup symmetry)
  run as part of the build rather than beside it.
- **MLflow UI**: runs today, after the gold step above, because the model needs
  the `features_turn` that dbt run built. `PIPELINE_DATA_DIR=/tmp/demo uv run
  python -m pipeline.train` trains the model and the baseline it is scored
  against and registers the result, then `PIPELINE_DATA_DIR=/tmp/demo uv run
  python -m pipeline.promote` judges that version against whatever holds the
  `production` alias and prints its one-line decision. Read both in a browser
  with `MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri
  /tmp/demo/mlruns`: the Experiments tab puts the model run next to the
  baseline run on the same holdout, and the Models tab shows the version, its
  holdout tags and which alias it holds. What it proves: the comparison is
  recorded rather than asserted, and nothing is served because a training run
  happened.
- **FastAPI predict**: runs today, with a version promoted.
  `PIPELINE_DATA_DIR=/tmp/demo uv run python -m pipeline.serve` loads
  `models:/win-probability@production`, and then:

  ```bash
  curl -s localhost:8000/health
  curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
    "turn_number": 8, "went_first": true,
    "archetype_key": "name:charizard-ex", "opponent_archetype_key": "name:gardevoir-ex",
    "prizes_taken_self": 3, "prizes_taken_opp": 1,
    "knockouts_self": 3, "knockouts_opp": 1, "cards_drawn_self": 26,
    "energy_attached_self": 5, "pokemon_played_self": 6, "trainers_played_self": 15,
    "evolutions_self": 2, "attacks_self": 4, "turns_played_self": 4 }'
  ```

  The reply carries the probability and the version that produced it, and
  `localhost:8000/docs` is the generated schema with a description on every
  field. What it proves: the service loads by alias rather than by version, so
  the deployment is the promotion; and an archetype the fixtures never trained
  on comes back named in `unknown_archetypes` rather than as an error.
- **Drift report**: runs today, after the gold step, and it needs no model at
  all: the default reference is the training split of the feature table.
  `PIPELINE_DATA_DIR=/tmp/demo uv run python -m pipeline.drift --window-days 3`
  compares the last three days of `features_turn` against the training window,
  prints the verdict line, and writes `/tmp/demo/drift/drift_report.md` beside
  a JSON summary, both logged as artifacts of a run in the
  `win-probability-drift` experiment. Three days rather than the default
  thirty because the corpus spans ten, and a thirty-day window would contain
  the training window whole and compare it with itself. What it proves: the
  expiry of a model is measured rather than assumed, the archetype mix is
  where a set release shows up first, and the command flags and exits 0
  instead of retraining anything.
- **Agent question**: not yet. Placeholder for asking the agent a matchup
  question and watching it write the mart query.
- **Airflow DAG**: not yet. Placeholder for the scheduled
  `bronze >> silver >> gold >> quality_gate` run.
- **Dashboard**: not yet. Placeholder for the published archetype and matchup
  views.

## Rehearsal log

| Date | Time taken | Notes |
| --- | --- | --- |
| 2026-09-21 | 12s of commands | Steps 1 to 7 back to back from a fresh clone, warm `uv` cache. Step 2 is 10 of the 12 seconds; steps 3 to 7 are under a second each. Cold cache and a real network clone add well under a minute, so the five minutes is narration, not waiting. Two things to watch: `sed` is deliberately bounded to the first match, because corrupting every `attack` entry makes the sidecar a wall of enum text, and `/tmp/demo/quarantine` survives a rehearsal, so clear `/tmp/demo` first or step 6 prints stale sidecars. |
|  |  |  |

Clean up after a rehearsal with `rm -rf /tmp/demo`.
