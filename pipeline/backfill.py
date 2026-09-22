"""Bronze backfill: read every parsed blob, validate it, land it or quarantine it.

This is the command that fills bronze from scratch and the command that refills
it after the anonymization key rotates. It is a batch walk over a prefix rather
than an event consumer on purpose: the history has to be ingestible at any time,
and the same walk re-run over the same objects must produce the same lake, which
the partition-replace write gives for free.

Where the blobs come from is injected, not assumed: the run takes a `Source`
(`pipeline.source`) that lists objects and reads one by key. In production that
is `S3Source` over the application's bucket; `--source-dir PATH` swaps in
`LocalSource` over a directory of files, which is how the committed fixtures are
ingested with no bucket, no credentials and no network. Everything after the
read is the same code on the same objects, so a local run is a real run of this
stage and not a simulation of one.

One object at a time is validated and anonymized, but the write happens once at
the end: `write_partitions` replaces each touched `play_date` directory whole, so
writing per object would rewrite the same partition once per game and leave a
partial day behind if the run died halfway.

Routing, per object:

- not UTF-8 JSON              -> quarantine `invalid_json`
- fails the contract          -> quarantine `contract_violation`
- v1 (no `schemaVersion`)     -> quarantine `v1_blob`, not anonymized, not written
- otherwise                   -> anonymized, re-validated, landed in bronze

Why v1 blobs are quarantined rather than upgraded here: a v1 blob has no
`summary`, so it has no play date, and the only honest substitute is the S3
last-modified time of the object, which is when it was uploaded and not when the
game was played. Guessing that would put games in the wrong partition and the
partition is what a re-run replaces. The producer can re-parse them to v2, which
is a one-command admin job upstream, so this stage names the key and waits.

Every game lands, blob untouched, including the ones a modified client exported
with both decklists. An opponent's list is only in a blob because the opponent
chose to share it in-game, so it is consented data and bronze keeps it
(docs/data-handling.md); `summary.hasFullDecklists` is counted on the way past
and is informational, not a route.

Why the leak check does not sink the batch: `write_partitions` refuses the whole
batch when a single handle survives anonymization, which is the right default for
a writer. Here that would mean one unusual handle (one that the rewrite misses,
say a handle that is a substring pattern the regex cannot bound) blocking every
other game forever. So the batch is re-checked per record, the records that leak
are quarantined as `handle_leak_check_failed`, and the rest are written. A leak
never reaches the lake either way.

Nothing here logs a handle or any blob content: the logs carry keys, reasons and
counts, and the validator summaries carry paths and messages only.

What the event consumer (`pipeline.consume`) shares with this module: the
per-object routing is `process_object`, and the write is `land_records`, so an
object that arrives by S3 event is validated, anonymized, leak-checked and
landed by exactly the code a backfill would have run over it. The consumer
differs in two places only, both of which it argues for in its own docstring:
it writes one game at a time (`merge=True`, so the day's other games survive)
and it does not quarantine a failed write, because a queue already has a
retry and a dead-letter queue for that.
"""

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from pipeline.anonymize import anonymize, assert_no_handles, handles_in
from pipeline.bronze import (
    BronzeLeakError,
    BronzeRecord,
    play_date_for,
    upsert_records,
    write_partitions,
)
from pipeline.config import BRONZE_DIR, QUARANTINE_DIR
from pipeline.contract import ContractError, ParsedBlobV2, parse_blob
from pipeline.observability import configure_logging, emit_summary, stage_run
from pipeline.quarantine import (
    CONTRACT_VIOLATION,
    HANDLE_LEAK_CHECK_FAILED,
    INVALID_JSON,
    V1_BLOB,
    WRITE_FAILED,
    write_quarantine,
)
from pipeline.settings import Settings
from pipeline.source import LocalSource, S3Source, Source, SourceBlob, SourceObject

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

V1_HINT = (
    "v1 blob: no schemaVersion and no summary, so the game has no play date; "
    "re-parse it upstream to contract v2 and run the backfill again"
)
AFTER_ANONYMIZATION = "after anonymization: "
LEAK_PATHS_SHOWN = 5
STAGE = "bronze_backfill"


@dataclass
class BackfillSummary:
    """What one run did. `partitions` maps play date to rows written for that date."""

    read: int = 0
    landed: int = 0
    quarantined: dict[str, int] = field(default_factory=dict)
    full_decklists_landed: int = 0
    partitions: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    def __str__(self) -> str:
        return "\n".join(
            [
                f"read: {self.read}",
                f"landed: {self.landed}",
                f"quarantined: {_counts(self.quarantined)}",
                f"full_decklists_landed: {self.full_decklists_landed}",
                f"partitions: {_counts(self.partitions)}",
                f"duration_s: {self.duration_s:.2f}",
            ]
        )


@dataclass(frozen=True)
class Prepared:
    """A validated, anonymized game waiting to be written.

    `handles` are its pre-anonymization handles, kept for the leak check, and
    `raw` is the body as read, kept in case the record has to be quarantined
    after all.
    """

    record: BronzeRecord
    handles: set[str]
    raw: bytes


@dataclass(frozen=True)
class Outcome:
    """Where one object came out of `process_object`.

    Exactly one of `prepared` and `reason` is set: the game is ready to be
    written, or it was quarantined and `reason` is the code it was filed under.
    `full_decklists` is the summary flag of a prepared game, which is counted
    and never routed on.
    """

    key: str
    prepared: Prepared | None = None
    reason: str | None = None
    full_decklists: bool = False


class Quarantiner:
    """Counts every rejection and writes it, unless the run is a dry run."""

    def __init__(self, directory: Path, when: datetime, *, dry_run: bool) -> None:
        self.directory = directory
        self.when = when
        self.dry_run = dry_run
        self.counts: dict[str, int] = {}

    def add(
        self,
        source_key: str,
        raw: bytes,
        reason: str,
        detail: str,
        *,
        contract_version_seen: Any = None,
    ) -> str:
        """Record one rejection and return its reason, so a caller can report it."""
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if self.dry_run:
            logger.warning("would quarantine %s as %s: %s", source_key, reason, detail)
            return reason
        write_quarantine(
            self.directory,
            source_key,
            raw,
            reason,
            detail,
            self.when,
            contract_version_seen=contract_version_seen,
        )
        return reason


def run_backfill(
    settings: Settings,
    s3: "S3Client | None" = None,
    bronze_dir: Path = BRONZE_DIR,
    quarantine_dir: Path = QUARANTINE_DIR,
    *,
    source: Source | None = None,
    now: datetime | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> BackfillSummary:
    """Walk the source once and return what the run did.

    `source` is where the blobs are read from; when it is None the run reads the
    bucket in `settings`, through `s3` if a client is passed and through a
    default client otherwise. `now` is the run timestamp, the same value for
    every row and every quarantine record; it defaults to the current UTC time.
    `limit` stops after that many listed blobs, and `dry_run` does all the
    reading, validating and counting while writing neither bronze nor quarantine.
    """
    started = time.monotonic()
    when = now or datetime.now(UTC)
    reading = source if source is not None else _s3_source(settings, s3)
    rejects = Quarantiner(quarantine_dir, when, dry_run=dry_run)
    summary = BackfillSummary()
    pending: list[Prepared] = []

    logger.info("listing %s", reading.label)
    for obj in islice(reading.list(), limit):
        summary.read += 1
        outcome = process_object(reading, obj.key, settings, rejects, listed=obj)
        if outcome.prepared is None:
            continue
        pending.append(outcome.prepared)
        if outcome.full_decklists:
            summary.full_decklists_landed += 1

    logger.info(
        "read %d object(s): %d ready, %d quarantined, %d with full decklists",
        summary.read,
        len(pending),
        sum(rejects.counts.values()),
        summary.full_decklists_landed,
    )
    summary.partitions = _land(pending, bronze_dir, when, rejects, dry_run=dry_run)
    summary.landed = sum(summary.partitions.values())
    summary.quarantined = dict(sorted(rejects.counts.items()))
    summary.duration_s = time.monotonic() - started
    return summary


def process_object(
    source: Source,
    key: str,
    settings: Settings,
    rejects: Quarantiner,
    *,
    listed: SourceObject | None = None,
) -> Outcome:
    """Read one object and route it: ready to write, or quarantined with a reason.

    The whole per-object path in one call (read, decode, contract, anonymize,
    re-validate), so the backfill's walk and the event consumer's queue are two
    ways of choosing keys and one way of handling them. `listed` is the listing
    entry when the caller has one, for the `source_last_modified` lineage column;
    a consumer that was handed a single key does not, and the column is then
    null, which is honest: nothing listed that object.

    Reading is the caller's risk: an S3 error is raised, not caught, because a
    batch and a queue answer it differently.
    """
    blob = source.get(key)
    obj = listed if listed is not None else SourceObject(key=key, size=len(blob.body))
    validated = _validate(blob, rejects)
    if isinstance(validated, str):
        return Outcome(key=key, reason=validated)
    model, data = validated
    prepared = _prepare(data, blob, obj, settings, rejects)
    if isinstance(prepared, str):
        return Outcome(key=key, reason=prepared)
    return Outcome(key=key, prepared=prepared, full_decklists=model.summary.has_full_decklists)


def _validate(blob: SourceBlob, rejects: Quarantiner) -> tuple[ParsedBlobV2, dict[str, Any]] | str:
    """The blob as a v2 model plus the dict it decoded from, or the reason it was rejected.

    Both come back because the model answers the routing and counting questions
    while the dict is what gets anonymized: rewriting the dict the producer sent,
    rather than a re-dump of the model, keeps the keys and the optional-key
    choices exactly as they arrived.
    """
    try:
        data = blob.decode()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # These messages report a position and an expected token, never content.
        return rejects.add(blob.key, blob.body, INVALID_JSON, f"{type(exc).__name__}: {exc}")
    try:
        parsed = parse_blob(data)
    except ContractError as exc:
        return rejects.add(
            blob.key,
            blob.body,
            CONTRACT_VIOLATION,
            exc.summary(),
            contract_version_seen=exc.schema_version_seen,
        )
    if not isinstance(parsed, ParsedBlobV2):
        return rejects.add(blob.key, blob.body, V1_BLOB, V1_HINT)
    return parsed, data


def _prepare(
    data: dict[str, Any],
    blob: SourceBlob,
    obj: SourceObject,
    settings: Settings,
    rejects: Quarantiner,
) -> Prepared | str:
    """Anonymize a valid v2 blob and re-validate it, or return why it cannot be landed.

    Re-validating after the rewrite is what proves the anonymizer changed the
    blob's strings and not its shape; a rewrite that broke the contract would
    otherwise reach the writer as a schema mismatch with no sign of the cause.
    """
    handles = handles_in(data)
    anonymized = anonymize(data, settings.hmac_key)
    try:
        clean = ParsedBlobV2.model_validate(anonymized)
    except ValidationError as exc:
        detail = AFTER_ANONYMIZATION + ContractError(exc).summary()
        return rejects.add(obj.key, blob.body, CONTRACT_VIOLATION, detail)
    record = BronzeRecord(
        blob=clean,
        source_key=obj.key,
        source_version_id=blob.version_id,
        source_last_modified=obj.last_modified,
    )
    return Prepared(record=record, handles=handles, raw=blob.body)


def land_records(
    pending: Sequence[Prepared],
    bronze_dir: Path,
    when: datetime,
    real_handles: set[str],
    *,
    merge: bool = False,
) -> dict[str, int]:
    """Write prepared games to bronze and return the rows landed per play date.

    Nothing is caught here: a leak raises `BronzeLeakError` and a filesystem
    error raises `OSError`, because the two callers answer them differently (the
    backfill quarantines, the consumer leaves the message on the queue).

    `merge` picks the write. The default replaces each touched partition whole,
    which is right for a caller holding every game of the day and is what makes
    a re-run drop games deleted upstream. `merge=True` keeps the games already in
    the partition, which is what a caller holding one game needs: a lone event
    must not empty the rest of its day.
    """
    write = upsert_records if merge else write_partitions
    return write([item.record for item in pending], bronze_dir, when, real_handles=real_handles)


def _land(
    pending: list[Prepared],
    bronze_dir: Path,
    when: datetime,
    rejects: Quarantiner,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Write every pending record that is clean; quarantine the ones that are not."""
    if not pending:
        return {}
    real_handles: set[str] = set().union(*(item.handles for item in pending))

    if dry_run:
        clean = _quarantine_leaks(pending, real_handles, rejects)
        return _partition_counts(clean)

    try:
        return land_records(pending, bronze_dir, when, real_handles)
    except BronzeLeakError:
        logger.warning("leak check failed for the batch; re-checking one game at a time")
    except OSError as exc:
        # Only the class name: the message of a filesystem error carries local
        # paths, and a sidecar is meant to be pasteable into an issue as it is.
        logger.exception("the bronze write failed; the batch is quarantined")
        for item in pending:
            rejects.add(item.record.source_key, item.raw, WRITE_FAILED, type(exc).__name__)
        return {}

    clean = _quarantine_leaks(pending, real_handles, rejects)
    if not clean:
        return {}
    return land_records(clean, bronze_dir, when, real_handles)


def _quarantine_leaks(
    pending: list[Prepared], real_handles: set[str], rejects: Quarantiner
) -> list[Prepared]:
    """Split the batch, recording each leaking record; returns the records that are clean."""
    clean: list[Prepared] = []
    for item in pending:
        paths = assert_no_handles(item.record.blob.model_dump(mode="json"), real_handles)
        if not paths:
            clean.append(item)
            continue
        # The paths are already masked: a leaking dict key reads as `<key>`.
        detail = f"{len(paths)} path(s) still hold a handle: {paths[:LEAK_PATHS_SHOWN]}"
        rejects.add(item.record.source_key, item.raw, HANDLE_LEAK_CHECK_FAILED, detail)
    return clean


def _partition_counts(pending: list[Prepared]) -> dict[str, int]:
    """Rows per play date, without writing: what a dry run would have landed."""
    counts: dict[str, int] = {}
    for item in pending:
        date = play_date_for(item.record.blob)
        counts[date] = counts.get(date, 0) + 1
    return dict(sorted(counts.items()))


def _s3_source(settings: Settings, s3: "S3Client | None") -> S3Source:
    """The production source: the bucket and prefix the settings name."""
    client = s3 if s3 is not None else _default_client(settings)
    return S3Source(client=client, bucket=settings.bucket, prefix=settings.prefix)


def _default_client(settings: Settings) -> "S3Client":
    import boto3

    return boto3.client("s3", region_name=settings.region)


def _counts(counts: dict[str, int]) -> str:
    return " ".join(f"{name}={value}" for name, value in sorted(counts.items())) or "none"


def main(argv: list[str] | None = None) -> int:
    """Run the backfill from the command line and report the summary."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.backfill",
        description=(
            "Backfill bronze from the parsed blobs in the application's S3 bucket, "
            "or from a local directory of blobs with --source-dir."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after this many blobs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and validate everything, write neither bronze nor quarantine",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="read every *.json under PATH instead of S3; needs no bucket and no credentials",
    )
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="where the landed rows go (default: the configured bronze directory)",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="where rejected blobs go (default: the configured quarantine directory)",
    )
    args = parser.parse_args(argv)

    configure_logging(STAGE)

    source: Source | None = None
    if args.source_dir is not None:
        if not args.source_dir.is_dir():
            parser.error(f"--source-dir is not a directory: {args.source_dir}")
        source = LocalSource(args.source_dir)
    # A local run reads no bucket, so it must not be blocked by an unset one.
    settings = Settings.from_env(require_bucket=source is None)

    with stage_run(STAGE) as metrics:
        summary = run_backfill(
            settings,
            bronze_dir=args.bronze_dir or BRONZE_DIR,
            quarantine_dir=args.quarantine_dir or QUARANTINE_DIR,
            source=source,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        metrics.rows_in = summary.read
        metrics.rows_out = summary.landed
        metrics.rows_quarantined = sum(summary.quarantined.values())
        metrics.extra = {
            "full_decklists_landed": summary.full_decklists_landed,
            "partitions": summary.partitions,
            "quarantined_by_reason": summary.quarantined,
            "dry_run": args.dry_run,
        }

    emit_summary(
        logger,
        "backfill summary",
        {
            "read": summary.read,
            "landed": summary.landed,
            "quarantined": sum(summary.quarantined.values()),
            "quarantined_by_reason": summary.quarantined,
            "full_decklists_landed": summary.full_decklists_landed,
            "partitions": summary.partitions,
            "duration_s": round(summary.duration_s, 4),
        },
        text=str(summary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
