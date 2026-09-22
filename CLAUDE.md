# pokemon-meta-pipeline: notes for coding agents and new contributors

- Python 3.11+, `uv` for everything (`uv sync --group dev`, `uv run ...`), ruff + mypy + pytest with a 70% coverage floor; `uv run pytest` is the fast suite, `-m spark`, `-m dbt`, `-m ml` need Java or model downloads and run separately in CI.
- Secrets and identifiers come from 1Password through `op run --env-file=.env.op` (prod) or `.env.dev.op` (dev). Both files hold `op://` references only. Never commit a bucket name, queue URL, role ARN, account id or key; `scripts/check_history.sh` scans every commit and CI runs it.
- `data/` is gitignored and may hold the real production lake, warehouse and MLflow store. Never delete anything under `data/` that you did not create in the current task; use `--data-dir` / `--bronze-dir` / `PIPELINE_DATA_DIR` to point a run at a scratch directory instead.
- Real player data never enters the repository. Fixtures under `tests/fixtures/` are produced only by `scripts/refresh_fixtures.py` (anonymized, throwaway key). Do not print handles, user ids or raw blobs in logs, tests or reports.
- The parsed-blob contract is `contract/parsed-blob.schema.json`, vendored from the producer; `pipeline/contract/` mirrors it and `tests/test_contract.py` compares them. A producer schema change means re-vendoring plus a model change, never loosening `extra="forbid"`.
- Every stage is a CLI (`python -m pipeline.<stage>`) that logs JSON lines and writes a `run_metrics` row; the Airflow DAG and `run_all` only call those CLIs.
- No `print` under `pipeline/` or `scripts/`; use the logger and `emit_summary`.
- Commit messages: imperative subject, a short body saying why, no attribution trailers.
