"""Gold stage: one command that builds and tests the dbt project over silver.

dbt is a command-line tool, so this is a thin wrapper rather than a library
call: it pins the project and profiles directories (both are `dbt/` in this
repository, so a fresh clone needs no `~/.dbt`), passes the lake location
through `PIPELINE_DATA_DIR` so the models and the Python package cannot
disagree about where the data is, and runs the steps in order. Orchestration
later calls one command per stage, and this is the gold one.

The counts that reach `run_metrics` are read back out of `target/run_results.json`
rather than scraped from dbt's output: dbt writes that file after every
invocation, and parsing the artifact it already produces beats matching on a
line of console text that changes between minor versions. It is also why dbt's
output is left streaming to the terminal instead of being captured.

Where the warehouse goes is the one thing an `s3://` lake root changes, and it
changes it in a way worth spelling out. dbt reads the silver Parquet in place
through DuckDB's `httpfs` extension, with the same credential chain everything
else uses, so `sources.yml` needs nothing but the root it already interpolates.
DuckDB cannot write its own database file over object storage, though: it is a
file it seeks around in, not a stream it appends to. So the build always happens
on the task's local disk, and the finished `meta.duckdb` is uploaded under
`warehouse/` afterwards as an artifact. That is honest about what it is. The
warehouse "holds no state worth keeping" (docs/stages.md) and is rebuilt from
the lake every run, so the upload is a convenience for the stages that read it
next and for a person who wants last night's numbers, not a database anyone
writes to in place.

The marts are also written out as Parquet, under `warehouse/marts/<mart>.parquet`,
one file per table dbt materialized. Two reasons. The next design step reads the
marts without DuckDB at all, from whatever engine happens to be in front of them,
and a Parquet file is what every engine can open; a DuckDB file is one process at
a time and one version of one library. And a reader that only wants the matchup
matrix should not have to download a whole warehouse to get it. The views are not
exported, because a view here is a query over Parquet that is already in the lake.

`--steps` and `--select` exist for the orchestrator, not for a person. A DAG
wants `dbt run` and `dbt test` to be two nodes, so that a failed test is a task
a reader can retry on its own and a failed build is a different one, and it
wants the feature table to be a node with a name. Split that way, each
invocation still writes its own `run_metrics` row, which is what `--stage-name`
is for: three calls under one run identifier would otherwise write three rows to
one file and keep only the last. Called with no arguments, the command is what
it always was, one `gold` stage that runs both steps over the whole project.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from pipeline.config import PIPELINE_DATA_DIR, REPO_ROOT
from pipeline.observability import STATUS_FAILED, configure_logging, emit_summary, stage_run
from pipeline.storage import AnyLocation, Location, duckdb_s3_profile, location

logger = logging.getLogger(__name__)

DBT_DIR = REPO_ROOT / "dbt"
STAGE = "gold"
RUN_RESULTS = Path("target") / "run_results.json"
# What dbt calls a node that did what it was asked to: `success` for a model,
# `pass` for a test.
OK_STATUSES = frozenset({"success", "pass"})
# The two dbt steps this wraps, in the only order they make sense in. `deps` is
# not in the list because it is not a choice: it runs when the project has
# packages and does nothing to the warehouse either way.
STEPS = ("run", "test")
# The dbt profile target per kind of root. `dev` reads and writes local files;
# `s3` is the same build with `httpfs` loaded and a credential-chain secret, so
# the sources resolve over object storage. One profile, two targets, rather than
# two profiles, because everything else about the build is identical.
LOCAL_TARGET = "dev"
S3_TARGET = "s3"
WAREHOUSE_NAME = "meta.duckdb"
MARTS_DIR = "marts"
# Where the profile reads the DuckDB path from. Separate from `PIPELINE_DATA_DIR`
# because on an S3 root the two are not the same place: the lake is the bucket
# and the database file is the task's disk.
WAREHOUSE_VAR = "PRA_WAREHOUSE_PATH"
# The three values the `s3` target's DuckDB secret interpolates. They are
# exported rather than written into the profile because dbt parses the YAML
# before it renders any value, so a key cannot be left out conditionally;
# `pipeline.storage.duckdb_s3_profile` says what they default to and why.
DUCKDB_S3_VARS = {
    "endpoint": "PRA_DUCKDB_S3_ENDPOINT",
    "url_style": "PRA_DUCKDB_S3_URL_STYLE",
    "use_ssl": "PRA_DUCKDB_S3_USE_SSL",
}


@dataclass
class GoldSummary:
    """What the build did, per dbt step, in the order the steps ran, and what it published."""

    steps: dict[str, tuple[int, int]] = field(default_factory=dict)
    marts: list[str] = field(default_factory=list)

    @property
    def models_run(self) -> int:
        """Nodes the `run` step attempted."""
        return self.steps.get("run", (0, 0))[0]

    @property
    def models_passed(self) -> int:
        """Nodes the `run` step built without an error."""
        return self.steps.get("run", (0, 0))[1]

    @property
    def tests_run(self) -> int:
        """Data tests the `test` step attempted."""
        return self.steps.get("test", (0, 0))[0]

    @property
    def tests_passed(self) -> int:
        """Data tests that passed."""
        return self.steps.get("test", (0, 0))[1]

    def __str__(self) -> str:
        lines = []
        if "run" in self.steps:
            lines.append(f"models: {self.models_passed}/{self.models_run} built")
        if "test" in self.steps:
            lines.append(f"tests:  {self.tests_passed}/{self.tests_run} passed")
        if self.marts:
            lines.append(f"marts:  {len(self.marts)} written as Parquet")
        return "\n".join(lines) or "no dbt step ran"


def dbt_executable() -> str:
    """The `dbt` beside this interpreter, so a virtual environment run finds its own."""
    local = Path(sys.executable).parent / "dbt"
    return str(local) if local.is_file() else "dbt"


def read_run_results(dbt_dir: Path) -> tuple[int, int]:
    """Nodes attempted and nodes that passed, from the artifact the last step wrote.

    `(0, 0)` when there is no artifact or it cannot be read: a missing count is
    worth a zero in a metrics row, never a failed build.
    """
    path = dbt_dir / RUN_RESULTS
    try:
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError):
        return (0, 0)
    return (len(results), sum(1 for node in results if node.get("status") in OK_STATUSES))


def run_gold(
    *,
    data_dir: AnyLocation,
    target: str | None = None,
    dbt_dir: Path = DBT_DIR,
    summary: GoldSummary | None = None,
    steps: Sequence[str] = STEPS,
    select: str | None = None,
) -> int:
    """Run `dbt deps` (when there are packages) and then the requested steps.

    `summary` is filled in as the steps run when one is passed. It is an
    argument rather than the return value because the exit code is what every
    caller and every test already reads, and a build that failed halfway still
    has counts worth recording.

    `select` is passed through to dbt untouched, so it takes whatever dbt's node
    selection takes: a model name, `tag:ml`, a `+` graph operator.

    `target` defaults to the one that matches the root, `dev` for a directory and
    `s3` for a bucket, so nothing has to be told twice. Passing one overrides
    that, which is what a build against a different profile output needs.
    """
    root = location(data_dir)
    chosen = target or (S3_TARGET if root.is_s3 else LOCAL_TARGET)
    warehouse = root / "warehouse" / WAREHOUSE_NAME
    with _build_directory(root) as built:
        env = {
            **os.environ,
            "PIPELINE_DATA_DIR": str(root),
            WAREHOUSE_VAR: str(built),
            **{DUCKDB_S3_VARS[key]: value for key, value in duckdb_s3_profile().items()},
        }
        wanted = list(steps)
        if (dbt_dir / "packages.yml").is_file():
            wanted.insert(0, "deps")
        for step in wanted:
            argv = [dbt_executable(), step, "--project-dir", str(dbt_dir)]
            if step != "deps":
                argv += ["--profiles-dir", str(dbt_dir), "--target", chosen]
                if select:
                    argv += ["--select", select]
            code = subprocess.run(argv, env=env, check=False).returncode
            if summary is not None and step != "deps":
                summary.steps[step] = read_run_results(dbt_dir)
            if code:
                return code
        publish_warehouse(built, root, summary=summary)
    logger.info("warehouse built", extra={"warehouse": str(warehouse), "target": chosen})
    return 0


@contextmanager
def _build_directory(root: Location) -> Iterator[Path]:
    """The local path dbt builds `meta.duckdb` at, for the length of the build.

    A local root builds in place, exactly where it always did, so a laptop run
    is unchanged and a rerun reuses the file dbt already knows how to rebuild.
    An S3 root builds in a temporary directory, because the one thing DuckDB
    cannot do is open a database over object storage.
    """
    if not root.is_s3:
        (root / "warehouse").mkdir()
        yield (root / "warehouse" / WAREHOUSE_NAME).path
        return
    with tempfile.TemporaryDirectory(prefix="pra-warehouse-") as scratch:
        yield Path(scratch) / WAREHOUSE_NAME


def publish_warehouse(
    built: Path, root: Location, *, summary: GoldSummary | None = None
) -> list[str]:
    """Write each materialized table out as Parquet, and the database beside them.

    Returns the mart names written. Called for both kinds of root: on a local
    one this is a copy next to the file it came from, which costs a second and
    means the lake has the same shape either way, so the reader that comes next
    does not have to ask which root it is looking at.
    """
    if not built.is_file():
        return []
    names = materialized_tables(built)
    connection = duckdb.connect(str(built), read_only=True)
    try:
        for name in names:
            table = connection.sql(f"select * from {name}").to_arrow_table()
            (root / "warehouse" / MARTS_DIR / f"{name}.parquet").write_table(table)
    finally:
        connection.close()
    if root.is_s3:
        # Only uploaded when the build was not already in place: a local root
        # built the file where it belongs and copying it onto itself is a way
        # to truncate it.
        (root / "warehouse" / WAREHOUSE_NAME).upload_file(built)
    if summary is not None:
        summary.marts = list(names)
    return list(names)


def materialized_tables(built: Path) -> list[str]:
    """The base tables in the warehouse, sorted: the marts and the feature table.

    Views are left out on purpose. A view here is a `read_parquet` over silver or
    over the run metrics, so exporting one would copy a file out of the lake and
    back into it under a different name.
    """
    connection = duckdb.connect(str(built), read_only=True)
    try:
        rows = connection.sql(
            "select table_name from duckdb_tables() where not internal order by table_name"
        ).fetchall()
    except duckdb.Error:
        return []
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def main(argv: list[str] | None = None) -> int:
    """Build the gold models from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.gold", description="Build and test the dbt gold models."
    )
    parser.add_argument(
        "--target",
        default=None,
        help=f"dbt profile target (default: {LOCAL_TARGET} for a directory, {S3_TARGET} for s3://)",
    )
    parser.add_argument(
        "--data-dir",
        type=location,
        default=PIPELINE_DATA_DIR,
        metavar="PATH",
        help="lake root, a directory or an s3:// prefix",
    )
    parser.add_argument(
        "--steps",
        default=",".join(STEPS),
        metavar="STEP[,STEP]",
        help=f"which dbt steps to run, in order (default: {','.join(STEPS)})",
    )
    parser.add_argument(
        "--select",
        default=None,
        metavar="EXPR",
        help="dbt node selection, for example a model name or tag:ml (default: every node)",
    )
    parser.add_argument(
        "--stage-name",
        default=STAGE,
        metavar="NAME",
        help=(
            f"the stage this run records itself as (default: {STAGE}); give each partial "
            "build its own name so two of them under one run identifier keep both rows"
        ),
    )
    args = parser.parse_args(argv)
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in STEPS]
    if unknown or not steps:
        parser.exit(2, f"{parser.prog}: --steps takes {' and '.join(STEPS)}, got {args.steps!r}\n")
    configure_logging(args.stage_name)

    summary = GoldSummary()
    # Pinned to the directory this run was pointed at, not to the ambient
    # `PIPELINE_DATA_DIR`: `--data-dir` moves the lake the models read, so a run
    # whose metrics landed somewhere else would be a row about a warehouse it
    # does not sit beside.
    with stage_run(args.stage_name, directory=args.data_dir / "lake" / "run_metrics") as metrics:
        code = run_gold(
            data_dir=args.data_dir,
            target=args.target,
            summary=summary,
            steps=steps,
            select=args.select,
        )
        metrics.rows_in = summary.models_run
        metrics.rows_out = summary.models_passed
        metrics.rows_quarantined = 0
        metrics.extra = {
            "tests_run": summary.tests_run,
            "tests_passed": summary.tests_passed,
            "target": args.target,
            "marts": summary.marts,
            "steps": steps,
            "select": args.select,
            "exit_code": code,
        }
        if code:
            metrics.status = STATUS_FAILED
            metrics.error = f"dbt exited {code}"

    emit_summary(
        logger,
        "gold summary",
        {
            "models_run": summary.models_run,
            "models_passed": summary.models_passed,
            "tests_run": summary.tests_run,
            "tests_passed": summary.tests_passed,
            "marts": summary.marts,
            "exit_code": code,
        },
        text=str(summary),
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
