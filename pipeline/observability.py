"""Structured logging and per-stage run metrics: one format, one run id, one row per stage.

Every stage is a separate process invoked by a scheduler, so the only thing
tying a bronze run to the silver run that followed it is an identifier carried
between them. That identifier is `PRA_RUN_ID`: the orchestrator sets one for a
whole DAG run, every command picks it up, and a run that is started by hand gets
a fresh one instead of nothing. It is held in a `contextvars.ContextVar` and
attached to every record by a `logging.Filter`, so no call site has to remember
to pass it.

Two outputs, deliberately on two streams:

- The log goes to stderr, one JSON object per line by default, because that is
  what a log shipper reads and what `PRA_LOG_FORMAT=console` turns into a line a
  person reads. `json_output` defaults to `not sys.stderr.isatty()`, so a
  terminal gets the readable form and a pipe, a container and continuous
  integration all get JSON with nothing to configure.
- The command's own result goes to stdout, through `emit_summary`, which logs
  the summary as one record with its fields in `extra` and writes the
  human-readable block to stdout. Keeping the two apart is what lets
  `python -m pipeline.backfill 2>/dev/null` stay readable and
  `... 1>/dev/null | jq` stay parseable, and it is why no stage needs `print`.

Run metrics are the same idea at the run level. `stage_run` times a stage,
records what went in and what came out, and writes exactly one Parquet row per
run per stage under `$PIPELINE_DATA_DIR/lake/run_metrics/`, named
`<run id>-<stage>.parquet`. One small file per run rather than an appended
table: two stages of the same DAG run finish at unpredictable times, sometimes
concurrently, and a writer that rewrites a shared file loses one of them. The
schema is pinned rather than inferred, the same reason bronze and silver pin
theirs: a run with no quarantined rows must not type `rows_quarantined`
differently from a run that had some.

A failing stage still writes its row, with `status = "failed"` and the error
class and message, and then re-raises. A stage that fails silently and leaves no
trace is the one case the table exists to cover.

Nothing here logs a handle, a user id or a token: the fields are counts,
durations, stage names and paths, and the one free-text field, `error`, carries
the exception class and its message, which the pipeline's own exceptions keep
free of blob content (see `pipeline.quarantine`).
"""

import contextvars
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import REPO_ROOT, RUN_METRICS_DIR

RUN_ID_VAR: Final = "PRA_RUN_ID"
LOG_FORMAT_VAR: Final = "PRA_LOG_FORMAT"
DATA_DIR_VAR: Final = "PIPELINE_DATA_DIR"
RUN_ID_LENGTH: Final = 16

STATUS_OK: Final = "ok"
STATUS_FAILED: Final = "failed"

# Set by `configure_logging` and by `stage_run`, read by the filter below. A
# context variable rather than a global so a test, or a future in-process
# orchestrator, can run two stages without them overwriting each other.
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("pra_run_id", default="")
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("pra_stage", default="")

# Everything the logging module itself puts on a record. Anything else came from
# an `extra=` and is a field of the event.
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)
# The three the filter adds, which the formatter prints in fixed positions
# rather than among the event's own fields.
_CONTEXT: Final[frozenset[str]] = frozenset({"run_id", "stage", "logger"})

_FIELDS: Final[list[pa.Field]] = [
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("stage", pa.string(), nullable=False),
    pa.field("started_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("finished_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("duration_s", pa.float64(), nullable=False),
    pa.field("rows_in", pa.int64(), nullable=True),
    pa.field("rows_out", pa.int64(), nullable=True),
    pa.field("rows_quarantined", pa.int64(), nullable=True),
    pa.field("status", pa.string(), nullable=False),
    pa.field("error", pa.string(), nullable=True),
    pa.field("extra_json", pa.string(), nullable=False),
    pa.field("git_commit", pa.string(), nullable=True),
    pa.field("hostname", pa.string(), nullable=True),
]
RUN_METRICS_SCHEMA: Final = pa.schema(_FIELDS)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- identity --


def new_run_id() -> str:
    """A fresh run identifier: the first 16 hex characters of a version 4 UUID.

    Short enough to read out of a log line and paste into a query, and random
    enough that two runs started in the same second do not collide. Not a ULID:
    the sort order a ULID buys would only matter if the identifier were the
    table's clustering key, and `started_at` already is.
    """
    return uuid.uuid4().hex[:RUN_ID_LENGTH]


def current_run_id() -> str:
    """The run identifier this process is logging under, resolving one if it has none.

    `PRA_RUN_ID` wins, so every stage of one orchestrated DAG run shares an
    identifier without any of them knowing about the others.
    """
    existing = _run_id.get()
    if existing:
        return existing
    resolved = os.environ.get(RUN_ID_VAR, "").strip() or new_run_id()
    _run_id.set(resolved)
    return resolved


def git_commit() -> str | None:
    """The commit this run was made at, or None outside a checkout.

    None rather than a raise when git is not there at all, which is the normal
    case in a container: the Airflow image ships no git, and a run that cannot
    name its commit must still be able to write the row that says what it did.
    Before this caught `OSError` a task in that image wrote no `run_metrics` row
    at all, because the column's value raised while the row was being built.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# ---------------------------------------------------------------- logging --


class ContextFilter(logging.Filter):
    """Puts the run identifier and the stage on every record, whoever emitted it.

    A filter rather than a `LoggerAdapter` because the records this has to reach
    include the ones PySpark, MLflow and boto3 emit, which know nothing about
    this module.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = getattr(record, "run_id", "") or _run_id.get()
        record.stage = getattr(record, "stage", "") or _stage.get()
        return True


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    """Whatever an `extra=` put on this record, and nothing the logging module put there."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED and key not in _CONTEXT and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line: the context first, then the event's own fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "stage": getattr(record, "stage", ""),
            "run_id": getattr(record, "run_id", ""),
            "msg": record.getMessage(),
        }
        payload.update(_fields(record))
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc_message"] = str(record.exc_info[1])
            payload["stack"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["stack"] = record.exc_text
        # `default=str` rather than a custom encoder: a field that is a Path, a
        # date or a set is worth one line of output, and a log call is not a
        # place to raise.
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """The same record as one compact line, stage and run identifier prefixed."""

    def format(self, record: logging.LogRecord) -> str:
        when = datetime.fromtimestamp(record.created, UTC).strftime("%H:%M:%S")
        stage = getattr(record, "stage", "") or "-"
        run_id = getattr(record, "run_id", "") or "-"
        line = f"{when} {record.levelname:<7} [{stage} {run_id}] {record.getMessage()}"
        fields = " ".join(f"{key}={value}" for key, value in sorted(_fields(record).items()))
        if fields:
            line = f"{line} {fields}"
        if record.exc_info and record.exc_info[0] is not None:
            line = f"{line} {record.exc_info[0].__name__}: {record.exc_info[1]}"
        return line


def json_logging(json_output: bool | None = None) -> bool:
    """Whether records are rendered as JSON: the argument, then the variable, then the terminal."""
    if json_output is not None:
        return json_output
    configured = os.environ.get(LOG_FORMAT_VAR, "").strip().lower()
    if configured == "json":
        return True
    if configured == "console":
        return False
    return not sys.stderr.isatty()


def configure_logging(
    stage: str,
    run_id: str | None = None,
    json_output: bool | None = None,
    *,
    level: int = logging.INFO,
) -> str:
    """Install the handler for this process and return the run identifier it will carry.

    Called once at the top of every stage entry point. Only the handler this
    module installed is replaced on a second call, so pytest's own capture
    handler, and anything a host application added, survive.
    """
    resolved = (run_id or "").strip() or os.environ.get(RUN_ID_VAR, "").strip() or new_run_id()
    _run_id.set(resolved)
    _stage.set(stage)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logging(json_output) else ConsoleFormatter())
    handler.addFilter(ContextFilter())
    handler.set_name("pra")

    root = logging.getLogger()
    for existing in [found for found in root.handlers if found.get_name() == "pra"]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    return resolved


def safe_extra(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Field names the logging module will accept, with any collision suffixed.

    `logging` refuses an `extra=` key that would overwrite a record attribute,
    by raising, which would turn a badly named field into a crashed stage.
    """
    return {(f"{key}_" if key in _RESERVED else key): value for key, value in fields.items()}


def emit_summary(
    log: logging.Logger,
    event: str,
    fields: Mapping[str, Any],
    *,
    text: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Log a command's summary as one record, and write its readable form to stdout.

    Two audiences, two streams. The record carries the fields, so a collector
    can chart them; `text` is the block the demo shows, so a reader still gets
    the table they had before this module existed. Writing it to stdout rather
    than only in console mode keeps a command's result pipeable on its own, with
    the log stream free of it either way.
    """
    log.log(level, event, extra=safe_extra(fields))
    if text:
        sys.stdout.write(text.rstrip("\n") + "\n")
        sys.stdout.flush()


# ------------------------------------------------------------ run metrics --


def run_metrics_dir() -> Path:
    """Where the run-metrics rows go, honouring a `PIPELINE_DATA_DIR` set after import.

    `config.RUN_METRICS_DIR` is resolved when the package is imported, which is
    the right answer for a command line and the wrong one for a test, or an
    orchestrator, that points the data directory somewhere else afterwards.
    """
    override = os.environ.get(DATA_DIR_VAR, "").strip()
    if override:
        return Path(override) / "lake" / "run_metrics"
    return RUN_METRICS_DIR


def _slug(value: str) -> str:
    """A run identifier or stage name as one safe filename component."""
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value) or "unknown"


@dataclass
class RunMetrics:
    """One stage's row, filled in by the stage while it runs.

    The three row counts are `None` until the stage sets them, and `None` is
    written rather than zero: a stage that never counted its input is not a
    stage that read nothing.
    """

    stage: str
    run_id: str
    started_at: datetime
    rows_in: int | None = None
    rows_out: int | None = None
    rows_quarantined: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_OK
    error: str | None = None
    finished_at: datetime | None = None
    duration_s: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """The row as the pinned schema wants it, with `extra` collapsed to JSON.

        `extra` is JSON in one column rather than a struct because every stage
        puts different keys in it, and a struct would make the table's schema a
        union of every stage's incidental fields.
        """
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_quarantined": self.rows_quarantined,
            "status": self.status,
            "error": self.error,
            "extra_json": json.dumps(self.extra, default=str, sort_keys=True),
            "git_commit": git_commit(),
            "hostname": socket.gethostname(),
        }


def write_run_metrics(metrics: RunMetrics, directory: Path | None = None) -> Path:
    """Write one row as its own Parquet file and return the path."""
    target = directory if directory is not None else run_metrics_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{_slug(metrics.run_id)}-{_slug(metrics.stage)}.parquet"
    table = pa.Table.from_pylist([metrics.as_row()], schema=RUN_METRICS_SCHEMA)
    pq.write_table(table, path)
    return path


@contextmanager
def stage_run(
    stage: str,
    *,
    run_id: str | None = None,
    directory: Path | None = None,
) -> Iterator[RunMetrics]:
    """Time a stage, write its row on the way out, and log `stage complete`.

        with stage_run("silver") as metrics:
            summary = run_silver(...)
            metrics.rows_in = summary.games_in
            metrics.rows_out = summary.rows["games"]

    The row is written whether the block returns or raises; on a raise it
    records `failed` with the exception class and message and the exception
    continues. A failure to write the row is logged and swallowed, because a
    stage that did its work must not be reported as broken by its own bookkeeping.
    """
    resolved = (run_id or "").strip() or current_run_id()
    metrics = RunMetrics(stage=stage, run_id=resolved, started_at=datetime.now(UTC))
    stage_token = _stage.set(stage)
    run_token = _run_id.set(resolved)
    started = time.monotonic()
    logger.info("stage start", extra={"stage": stage, "run_id": resolved})
    try:
        yield metrics
    except BaseException as failure:
        metrics.status = STATUS_FAILED
        metrics.error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        metrics.duration_s = time.monotonic() - started
        metrics.finished_at = datetime.now(UTC)
        _finish(metrics, directory)
        _stage.reset(stage_token)
        _run_id.reset(run_token)


def _finish(metrics: RunMetrics, directory: Path | None) -> None:
    """Write the row and log it, never letting either failure replace the stage's own."""
    path: Path | None = None
    try:
        path = write_run_metrics(metrics, directory)
    except (OSError, pa.ArrowInvalid) as failure:
        logger.warning(
            "run metrics were not written",
            extra={"error": f"{type(failure).__name__}: {failure}"},
        )
    logger.log(
        logging.INFO if metrics.status == STATUS_OK else logging.ERROR,
        "stage complete",
        extra=safe_extra(
            {
                "started_at": metrics.started_at.isoformat(),
                "finished_at": (metrics.finished_at or metrics.started_at).isoformat(),
                "duration_s": round(metrics.duration_s, 4),
                "rows_in": metrics.rows_in,
                "rows_out": metrics.rows_out,
                "rows_quarantined": metrics.rows_quarantined,
                "status": metrics.status,
                "error": metrics.error,
                "extra": metrics.extra,
                "run_metrics_path": str(path) if path else None,
            }
        ),
    )
