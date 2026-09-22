"""The whole pipeline as one command, with no scheduler involved.

`orchestration/airflow/dags/play_rough_pipeline.py` is the real orchestrator,
and this is the same graph with the graph taken out: the stages in order, one
after another, stopping at the first one that fails. It exists for three
reasons. A reviewer with a clone and no Docker can run the pipeline end to end
in one line. The tests can exercise the ordering, the skipping and the failure
behaviour without installing Airflow. And when a DAG task goes red, the way to
reproduce it locally is to run the same command the task ran, which is the same
command this runs.

Each stage is a subprocess of **this** interpreter (`sys.executable -m
pipeline.<stage>`), not an in-process call. Three things follow from that, all
of them wanted:

- a stage that dies takes nothing with it, and its exit code is a number this
  can report rather than a traceback it has to interpret;
- Spark gets a fresh Java Virtual Machine of its own and gives it back;
- the command a task runs and the command a person runs are byte for byte the
  same string, so "it works locally" means something.

Everything shares one `PRA_RUN_ID`, generated here when the environment has
none and exported to every child, so all the `run_metrics` rows and every log
line of one invocation carry the same identifier. That is the whole point of
the identifier, and a runner that let each stage invent its own would quietly
undo it.

Stages are skipped rather than dropped, each with a logged reason:

- `backfill`, when `PRA_INGEST_MODE=consumer`. The event-driven consumer is
  then the live ingest and bronze is already being written, so a backfill would
  be reading the same bucket a second time to land rows it already has.
- `train`, `promote` and `drift`, when `features_turn` holds no rows. A
  warehouse that built cleanly and has nothing modellable in it is a small
  corpus, not a broken run. The three commands make the same judgement for
  themselves when they are called directly, which is what the DAG does; the
  check here saves three process starts and keeps the summary honest about
  what actually ran.
- `build_card_index`, until `pipeline.card_index` exists. The retriever index
  is a later ticket. The step is listed rather than left out so the shape of
  the finished pipeline is visible in the one place that runs it.
- `publish`, when `PRA_INSIGHTS_TABLE` is unset. A clone with no table named
  has nowhere to publish to, which is the normal state of a reviewer's laptop
  and of every test in this repository; naming the variable in the reason is
  what turns "it did not publish" into "set this".
"""

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import duckdb

from pipeline.config import PIPELINE_DATA_DIR, REPO_ROOT
from pipeline.observability import (
    RUN_ID_VAR,
    STATUS_FAILED,
    STATUS_OK,
    configure_logging,
    current_run_id,
    emit_summary,
    stage_run,
)
from pipeline.settings import INSIGHTS_TABLE_VAR

logger = logging.getLogger(__name__)

STAGE: Final = "run_all"
INGEST_MODE_VAR: Final = "PRA_INGEST_MODE"
CONSUMER_MODE: Final = "consumer"
DATA_DIR_VAR: Final = "PIPELINE_DATA_DIR"
FEATURE_TABLE: Final = "features_turn"
CARD_INDEX_MODULE: Final = "pipeline.card_index"

STATUS_SKIPPED: Final = "skipped"


@dataclass(frozen=True)
class Stage:
    """One step of the run: what to call it, what to run, and when not to.

    Every conditional here is one a stage would make for itself if it were
    started, so the flags save a process start rather than deciding anything the
    command would not: three commands read `features_turn` and have nothing to
    do when it is empty, and the publish has nowhere to write when no table is
    named.
    """

    name: str
    module: str
    args: tuple[str, ...] = ()
    #: Passed `--source-dir` when the run was given one; otherwise reads S3.
    takes_source_dir: bool = False
    #: Skipped when `features_turn` holds no rows.
    needs_features: bool = False
    #: Skipped when its module is not importable, which is how an unbuilt
    #: stage stays visible in the list instead of being absent from it.
    optional_module: bool = False
    #: Skipped when no insights table is named, because there is nowhere to write.
    needs_insights_table: bool = False


# The order is the dependency order, and it is the DAG's order flattened: the
# gate is last because it judges every row the run wrote, including the last
# one. `gold` is a single step here and three tasks in the DAG, for the reason
# `pipeline.gold`'s `--steps` flag documents.
STAGES: Final[tuple[Stage, ...]] = (
    Stage(name="backfill", module="pipeline.backfill", takes_source_dir=True),
    Stage(name="silver", module="pipeline.silver"),
    Stage(name="gold", module="pipeline.gold"),
    Stage(name="train", module="pipeline.train", needs_features=True),
    Stage(
        name="promote",
        module="pipeline.promote",
        args=("--candidate", "latest"),
        needs_features=True,
    ),
    Stage(name="drift", module="pipeline.drift", needs_features=True),
    Stage(name="build_card_index", module=CARD_INDEX_MODULE, optional_module=True),
    Stage(name="quality_gate", module="pipeline.quality_gate"),
    # After the gate, not before it: publishing numbers the gate was about to
    # refuse would put a known-bad matchup matrix in front of every reader of
    # the application, and the runner stops at the first failure, so a red gate
    # is also the thing that stops this.
    Stage(name="publish", module="pipeline.publish", needs_insights_table=True),
)
STAGE_NAMES: Final[tuple[str, ...]] = tuple(stage.name for stage in STAGES)


@dataclass
class StageResult:
    """What one stage did: run and passed, run and failed, or never started."""

    name: str
    status: str
    duration_s: float = 0.0
    exit_code: int | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """The row as it goes into the summary's `extra`, without the empty fields."""
        row: dict[str, object] = {
            "stage": self.name,
            "status": self.status,
            "duration_s": round(self.duration_s, 4),
        }
        if self.exit_code is not None:
            row["exit_code"] = self.exit_code
        if self.reason:
            row["reason"] = self.reason
        return row


@dataclass
class RunAllSummary:
    """Every stage's result, in the order they were considered."""

    results: list[StageResult] = field(default_factory=list)

    @property
    def failed(self) -> StageResult | None:
        """The stage that stopped the run, or None."""
        return next((result for result in self.results if result.status == STATUS_FAILED), None)

    @property
    def exit_code(self) -> int:
        """The failing stage's code, or 0."""
        failure = self.failed
        return failure.exit_code or 1 if failure else 0

    def counts(self) -> dict[str, int]:
        """How many stages ended in each status."""
        counts = {STATUS_OK: 0, STATUS_SKIPPED: 0, STATUS_FAILED: 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def __str__(self) -> str:
        width = max((len(result.name) for result in self.results), default=len(STAGE))
        lines = [f"{'stage':<{width}}  status    seconds  note"]
        for result in self.results:
            note = result.reason or ""
            if result.status == STATUS_FAILED:
                note = f"exit {result.exit_code}"
            lines.append(
                f"{result.name:<{width}}  {result.status:<8}  {result.duration_s:>7.1f}  {note}"
            )
        counts = self.counts()
        lines.append("")
        lines.append(
            f"{counts[STATUS_OK]} ran, {counts[STATUS_SKIPPED]} skipped, "
            f"{counts[STATUS_FAILED]} failed"
        )
        return "\n".join(lines)


def feature_rows(warehouse: Path) -> int:
    """How many rows `features_turn` holds, or 0 when there is no table to ask.

    A missing warehouse and an unbuilt table both answer zero rather than
    raising: the question this asks is "is there anything to model", and all
    three ways of saying no mean the same thing to the caller.
    """
    if not warehouse.is_file():
        return 0
    try:
        connection = duckdb.connect(str(warehouse), read_only=True)
    except duckdb.Error:
        return 0
    try:
        row = connection.sql(f"select count(*) from {FEATURE_TABLE}").fetchone()
    except duckdb.Error:
        return 0
    finally:
        connection.close()
    return int(row[0]) if row else 0


def module_exists(module: str) -> bool:
    """Whether a stage's module is importable, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def skip_reason(stage: Stage, *, skipped: Sequence[str], features: int | None) -> str | None:
    """Why this stage should not run, or None to run it.

    `features` is None until something has needed the count, so a run that
    stops before the model stages never opens the warehouse.
    """
    if stage.name in skipped:
        return "skipped by --skip"
    if stage.name == "backfill" and os.environ.get(INGEST_MODE_VAR, "").strip() == CONSUMER_MODE:
        return f"{INGEST_MODE_VAR}={CONSUMER_MODE}: the consumer is the live ingest"
    if stage.optional_module and not module_exists(stage.module):
        return f"{stage.module} is not implemented yet"
    if stage.needs_features and features == 0:
        return f"{FEATURE_TABLE} holds no rows"
    if stage.needs_insights_table and not os.environ.get(INSIGHTS_TABLE_VAR, "").strip():
        return f"{INSIGHTS_TABLE_VAR} is unset: there is no table to publish to"
    return None


def stage_command(stage: Stage, *, source_dir: Path | None) -> list[str]:
    """The exact argument vector the stage is run as."""
    argv = [sys.executable, "-m", stage.module, *stage.args]
    if stage.takes_source_dir and source_dir is not None:
        argv += ["--source-dir", str(source_dir)]
    return argv


def run_stage(stage: Stage, *, argv: Sequence[str], env: dict[str, str]) -> StageResult:
    """Run one stage to completion and time it, whatever it exits with."""
    logger.info("stage running", extra={"stage_name": stage.name, "command": " ".join(argv)})
    started = time.monotonic()
    code = subprocess.run(list(argv), env=env, cwd=REPO_ROOT, check=False).returncode
    duration = time.monotonic() - started
    return StageResult(
        name=stage.name,
        status=STATUS_OK if code == 0 else STATUS_FAILED,
        duration_s=duration,
        exit_code=code,
    )


def run_all(
    *,
    data_dir: Path,
    source_dir: Path | None = None,
    skip: Sequence[str] = (),
    stop_after: str | None = None,
    stages: Sequence[Stage] = STAGES,
    env: dict[str, str] | None = None,
) -> RunAllSummary:
    """Run the stages in order, stopping at the first failure or at `stop_after`."""
    child_env = dict(os.environ if env is None else env)
    child_env[DATA_DIR_VAR] = str(data_dir)
    warehouse = data_dir / "warehouse" / "meta.duckdb"
    summary = RunAllSummary()
    features: int | None = None

    for stage in stages:
        if stage.needs_features and features is None:
            features = feature_rows(warehouse)
            logger.info(
                "feature rows counted",
                extra={"rows": features, "warehouse": str(warehouse)},
            )
        reason = skip_reason(stage, skipped=skip, features=features)
        if reason is not None:
            logger.info("stage skipped", extra={"stage_name": stage.name, "reason": reason})
            summary.results.append(
                StageResult(name=stage.name, status=STATUS_SKIPPED, reason=reason)
            )
        else:
            result = run_stage(
                stage, argv=stage_command(stage, source_dir=source_dir), env=child_env
            )
            summary.results.append(result)
            if result.status == STATUS_FAILED:
                logger.error(
                    "stage failed, stopping",
                    extra={"stage_name": stage.name, "exit_code": result.exit_code},
                )
                break
        if stop_after and stage.name == stop_after:
            break
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the whole pipeline from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.run_all",
        description="Run every pipeline stage in order, under one run identifier.",
        epilog="stages, in order: " + ", ".join(STAGE_NAMES),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="read the blobs from a directory instead of S3, for example tests/fixtures",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PIPELINE_DATA_DIR,
        metavar="PATH",
        help="the lake and warehouse root every stage is pointed at",
    )
    parser.add_argument(
        "--skip",
        default="",
        metavar="STAGE[,STAGE]",
        help="stages to leave out, by name",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        metavar="STAGE",
        help="run no further than this stage",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help=f"the identifier every stage shares (default: ${RUN_ID_VAR}, else a fresh one)",
    )
    args = parser.parse_args(argv)

    skip = [name.strip() for name in args.skip.split(",") if name.strip()]
    named = [*skip, *([args.stop_after] if args.stop_after else [])]
    unknown = [name for name in named if name not in STAGE_NAMES]
    if unknown:
        parser.exit(2, f"{parser.prog}: unknown stage(s) {', '.join(unknown)}\n")
    if args.source_dir is not None and not args.source_dir.is_dir():
        parser.exit(2, f"{parser.prog}: --source-dir is not a directory: {args.source_dir}\n")

    # Set on this process rather than only on the children: the run identifier
    # has to be the one this command's own summary row carries, and the data
    # directory has to be the one that row is written to.
    os.environ[DATA_DIR_VAR] = str(args.data_dir)
    run_id = configure_logging(STAGE, args.run_id)
    os.environ[RUN_ID_VAR] = run_id

    with stage_run(STAGE) as metrics:
        summary = run_all(
            data_dir=args.data_dir,
            source_dir=args.source_dir,
            skip=skip,
            stop_after=args.stop_after,
        )
        counts = summary.counts()
        metrics.rows_in = len(summary.results)
        metrics.rows_out = counts[STATUS_OK]
        metrics.rows_quarantined = 0
        metrics.extra = {
            "stages": [result.as_dict() for result in summary.results],
            "ran": counts[STATUS_OK],
            "skipped": counts[STATUS_SKIPPED],
            "failed": counts[STATUS_FAILED],
            "source_dir": str(args.source_dir) if args.source_dir else None,
            "data_dir": str(args.data_dir),
            "exit_code": summary.exit_code,
        }
        failure = summary.failed
        if failure is not None:
            metrics.status = STATUS_FAILED
            metrics.error = f"{failure.name} exited {failure.exit_code}"

    emit_summary(
        logger,
        "run_all summary",
        {
            "run_id": current_run_id(),
            "stages": [result.as_dict() for result in summary.results],
            "exit_code": summary.exit_code,
        },
        text=str(summary),
        level=logging.ERROR if summary.failed else logging.INFO,
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
