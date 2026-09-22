"""Quality gate: the last task of a run, and the one allowed to fail it.

Every stage already writes a `run_metrics` row and two dbt models already read
those rows back (`run_metrics` and `mart_pipeline_health`, under
`dbt/models/ops/`). What was missing was something that acts on them. A row
saying `status = "failed"` in a table nobody queries is not an alert; it is a
fact with no consequence.

So this command asks the mart two questions and turns the answers into an exit
code:

- did any stage's **last** run fail, and
- is any stage's `quarantine_rate_over_threshold` true, meaning it has thrown
  away more than `quarantine_rate_alert` of what it read over its last ten runs.

Either one exits `EXIT_FAILED`, which is what makes the scheduler's run red.
Neither exits 0. It reads the warehouse and writes nothing to it, so running the
gate twice says the same thing twice.

Two deliberate choices. The gate reads the mart rather than the Parquet files,
because the rule it enforces (the rate over a window, weighted by rows rather
than by run) is defined in SQL in one place and a second definition in Python
would be the one that drifts. And it judges every stage the warehouse knows
about, not only the stages of the current run: a stage that failed last night
and was not rerun is still a broken stage this morning, and a gate that looked
only at what just ran would report green over it.

`mart_pipeline_health` is a view, so a stage that finished a second ago is
already in it and the gate does not need a dbt rebuild between the last stage
and itself.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from pipeline.config import WAREHOUSE_PATH
from pipeline.observability import (
    STATUS_FAILED,
    RunMetrics,
    configure_logging,
    emit_summary,
    stage_run,
)

logger = logging.getLogger(__name__)

STAGE: Final = "quality_gate"
HEALTH_TABLE: Final = "mart_pipeline_health"
# A gate that refuses is not a crashed gate, so the refusal gets its own code:
# 1 means "the pipeline is unhealthy", 2 (argparse's) means "the gate could not
# be run at all", which is a different thing to wake up to.
EXIT_FAILED: Final = 1

# The gate's own row is excluded from what the gate judges. It writes one like
# every other stage, and it writes it as `failed` when it refused, which is the
# truth about that run; but a gate that then read its own refusal back would
# stay red for ever, long after the stage that caused it was fixed. The verdict
# is about the pipeline, not about the verdict.
QUERY: Final = f"""
select
    stage,
    last_run_id,
    last_status,
    last_error,
    quarantine_rate,
    quarantine_rate_over_threshold
from {HEALTH_TABLE}
where stage != '{STAGE}'
order by stage
"""


class QualityGateError(RuntimeError):
    """The gate could not read the mart; the message says what was missing."""


@dataclass(frozen=True)
class StageHealth:
    """One row of `mart_pipeline_health`, reduced to what the gate judges on."""

    stage: str
    last_run_id: str
    last_status: str
    last_error: str | None
    quarantine_rate: float
    quarantine_rate_over_threshold: bool

    @property
    def failed(self) -> bool:
        """Whether this stage is a reason to fail the run."""
        return self.last_status == STATUS_FAILED or self.quarantine_rate_over_threshold

    @property
    def reason(self) -> str:
        """Why this stage failed the gate, in one clause, or an empty string."""
        if self.last_status == STATUS_FAILED:
            return f"last run {self.last_run_id} failed: {self.last_error or 'no error recorded'}"
        if self.quarantine_rate_over_threshold:
            return f"quarantine rate {self.quarantine_rate:.1%} is over the threshold"
        return ""


def read_health(warehouse: Path) -> list[StageHealth]:
    """The mart, one object per stage, from a read-only connection."""
    if not warehouse.is_file():
        raise QualityGateError(f"no warehouse at {warehouse}; run `python -m pipeline.gold` first")
    try:
        connection = duckdb.connect(str(warehouse), read_only=True)
    except duckdb.Error as unreadable:
        raise QualityGateError(f"{warehouse} could not be opened: {unreadable}") from unreadable
    try:
        rows = connection.sql(QUERY).fetchall()
    except duckdb.Error as missing:
        raise QualityGateError(
            f"{warehouse} has no readable {HEALTH_TABLE}; run `python -m pipeline.gold` first"
        ) from missing
    finally:
        connection.close()
    return [
        StageHealth(
            stage=str(stage),
            last_run_id=str(run_id),
            last_status=str(status),
            last_error=error,
            quarantine_rate=float(rate or 0.0),
            quarantine_rate_over_threshold=bool(over),
        )
        for stage, run_id, status, error, rate, over in rows
    ]


def report(health: list[StageHealth]) -> str:
    """The table a person reads, widest stage name first so the columns line up."""
    if not health:
        return f"{HEALTH_TABLE} is empty: no stage has recorded a run against this warehouse."
    width = max(len(row.stage) for row in health)
    lines = [f"{'stage':<{width}}  status    quarantine  gate"]
    for row in sorted(health, key=lambda item: item.stage):
        verdict = "FAIL" if row.failed else "ok"
        lines.append(
            f"{row.stage:<{width}}  {row.last_status:<8}  {row.quarantine_rate:>9.1%}  {verdict}"
        )
    offenders = [row for row in health if row.failed]
    lines.append("")
    if offenders:
        lines.append(f"gate failed on {len(offenders)} stage(s):")
        lines += [f"  {row.stage}: {row.reason}" for row in offenders]
    else:
        lines.append(f"gate passed: {len(health)} stage(s) healthy.")
    return "\n".join(lines)


def run_quality_gate(*, warehouse: Path, metrics: RunMetrics | None = None) -> int:
    """Read the mart, print the verdict, return 0 or `EXIT_FAILED`."""
    health = read_health(warehouse)
    offenders = [row for row in health if row.failed]
    code = EXIT_FAILED if offenders else 0
    emit_summary(
        logger,
        "quality gate",
        {
            "stages": len(health),
            "failed_stages": [row.stage for row in offenders],
            "reasons": {row.stage: row.reason for row in offenders},
            "warehouse": str(warehouse),
            "exit_code": code,
        },
        text=report(health),
        level=logging.ERROR if offenders else logging.INFO,
    )
    if metrics is not None:
        metrics.rows_in = len(health)
        metrics.rows_out = len(health) - len(offenders)
        metrics.rows_quarantined = 0
        metrics.extra = {
            "failed_stages": [row.stage for row in offenders],
            "reasons": {row.stage: row.reason for row in offenders},
            "exit_code": code,
        }
        if offenders:
            metrics.status = STATUS_FAILED
            metrics.error = f"{len(offenders)} stage(s) failed the gate"
    return code


def main(argv: list[str] | None = None) -> int:
    """Judge the pipeline's own health from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.quality_gate",
        description=(
            "Read mart_pipeline_health and fail when a stage's last run failed or its "
            "quarantine rate is over the threshold."
        ),
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help=f"DuckDB warehouse holding {HEALTH_TABLE}",
    )
    args = parser.parse_args(argv)
    configure_logging(STAGE)
    try:
        with stage_run(STAGE) as metrics:
            return run_quality_gate(warehouse=args.warehouse, metrics=metrics)
    except QualityGateError as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
