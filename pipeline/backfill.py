"""Bronze backfill: read every parsed blob in S3, validate it, land it or quarantine it.

This is the command that fills bronze from scratch and the command that refills
it after the anonymization key rotates. It is a batch walk over a prefix rather
than an event consumer on purpose: the history has to be ingestible at any time,
and the same walk re-run over the same objects must produce the same lake, which
the partition-replace write gives for free.

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
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from pipeline.anonymize import anonymize, assert_no_handles, handles_in
from pipeline.bronze import BronzeLeakError, BronzeRecord, play_date_for, write_partitions
from pipeline.config import BRONZE_DIR, QUARANTINE_DIR
from pipeline.contract import ContractError, ParsedBlobV2, parse_blob
from pipeline.quarantine import (
    CONTRACT_VIOLATION,
    HANDLE_LEAK_CHECK_FAILED,
    INVALID_JSON,
    V1_BLOB,
    WRITE_FAILED,
    write_quarantine,
)
from pipeline.settings import Settings
from pipeline.source import S3Object, SourceBlob, get_blob, list_parsed_keys

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

V1_HINT = (
    "v1 blob: no schemaVersion and no summary, so the game has no play date; "
    "re-parse it upstream to contract v2 and run the backfill again"
)
AFTER_ANONYMIZATION = "after anonymization: "
LEAK_PATHS_SHOWN = 5


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
class _Pending:
    """A validated, anonymized game waiting for the end-of-run write.

    `handles` are its pre-anonymization handles, kept for the leak check, and
    `raw` is the body as read, kept in case the record has to be quarantined
    after all.
    """

    record: BronzeRecord
    handles: set[str]
    raw: bytes


class _Quarantiner:
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
    ) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if self.dry_run:
            logger.warning("would quarantine %s as %s: %s", source_key, reason, detail)
            return
        write_quarantine(
            self.directory,
            source_key,
            raw,
            reason,
            detail,
            self.when,
            contract_version_seen=contract_version_seen,
        )


def run_backfill(
    settings: Settings,
    s3: "S3Client | None" = None,
    bronze_dir: Path = BRONZE_DIR,
    quarantine_dir: Path = QUARANTINE_DIR,
    *,
    now: datetime | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> BackfillSummary:
    """Walk the source prefix once and return what the run did.

    `now` is the run timestamp, the same value for every row and every
    quarantine record; it defaults to the current UTC time. `limit` stops after
    that many listed blobs, and `dry_run` does all the reading, validating and
    counting while writing neither bronze nor quarantine.
    """
    started = time.monotonic()
    when = now or datetime.now(UTC)
    client = s3 if s3 is not None else _default_client(settings)
    rejects = _Quarantiner(quarantine_dir, when, dry_run=dry_run)
    summary = BackfillSummary()
    pending: list[_Pending] = []

    logger.info("listing s3://%s/%s", settings.bucket, settings.prefix)
    listing = list_parsed_keys(client, settings.bucket, settings.prefix)
    for obj in islice(listing, limit):
        summary.read += 1
        blob = get_blob(client, settings.bucket, obj.key)
        validated = _validate(blob, rejects)
        if validated is None:
            continue
        model, data = validated
        prepared = _prepare(data, blob, obj, settings, rejects)
        if prepared is None:
            continue
        pending.append(prepared)
        if model.summary.has_full_decklists:
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


def _validate(
    blob: SourceBlob, rejects: _Quarantiner
) -> tuple[ParsedBlobV2, dict[str, Any]] | None:
    """The blob as a v2 model plus the dict it decoded from, or None after recording why not.

    Both come back because the model answers the routing and counting questions
    while the dict is what gets anonymized: rewriting the dict the producer sent,
    rather than a re-dump of the model, keeps the keys and the optional-key
    choices exactly as they arrived.
    """
    try:
        data = blob.decode()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # These messages report a position and an expected token, never content.
        rejects.add(blob.key, blob.body, INVALID_JSON, f"{type(exc).__name__}: {exc}")
        return None
    try:
        parsed = parse_blob(data)
    except ContractError as exc:
        rejects.add(
            blob.key,
            blob.body,
            CONTRACT_VIOLATION,
            exc.summary(),
            contract_version_seen=exc.schema_version_seen,
        )
        return None
    if not isinstance(parsed, ParsedBlobV2):
        rejects.add(blob.key, blob.body, V1_BLOB, V1_HINT)
        return None
    return parsed, data


def _prepare(
    data: dict[str, Any],
    blob: SourceBlob,
    obj: S3Object,
    settings: Settings,
    rejects: _Quarantiner,
) -> _Pending | None:
    """Anonymize a valid v2 blob and re-validate it, or record why it cannot be landed.

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
        rejects.add(obj.key, blob.body, CONTRACT_VIOLATION, detail)
        return None
    record = BronzeRecord(
        blob=clean,
        source_key=obj.key,
        source_version_id=blob.version_id,
        source_last_modified=obj.last_modified,
    )
    return _Pending(record=record, handles=handles, raw=blob.body)


def _land(
    pending: list[_Pending],
    bronze_dir: Path,
    when: datetime,
    rejects: _Quarantiner,
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
        return write_partitions(
            [item.record for item in pending], bronze_dir, when, real_handles=real_handles
        )
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
    return write_partitions(
        [item.record for item in clean], bronze_dir, when, real_handles=real_handles
    )


def _quarantine_leaks(
    pending: list[_Pending], real_handles: set[str], rejects: _Quarantiner
) -> list[_Pending]:
    """Split the batch, recording each leaking record; returns the records that are clean."""
    clean: list[_Pending] = []
    for item in pending:
        paths = assert_no_handles(item.record.blob.model_dump(mode="json"), real_handles)
        if not paths:
            clean.append(item)
            continue
        # The paths are already masked: a leaking dict key reads as `<key>`.
        detail = f"{len(paths)} path(s) still hold a handle: {paths[:LEAK_PATHS_SHOWN]}"
        rejects.add(item.record.source_key, item.raw, HANDLE_LEAK_CHECK_FAILED, detail)
    return clean


def _partition_counts(pending: list[_Pending]) -> dict[str, int]:
    """Rows per play date, without writing: what a dry run would have landed."""
    counts: dict[str, int] = {}
    for item in pending:
        date = play_date_for(item.record.blob)
        counts[date] = counts.get(date, 0) + 1
    return dict(sorted(counts.items()))


def _default_client(settings: Settings) -> "S3Client":
    import boto3

    return boto3.client("s3", region_name=settings.region)


def _counts(counts: dict[str, int]) -> str:
    return " ".join(f"{name}={value}" for name, value in sorted(counts.items())) or "none"


def main(argv: list[str] | None = None) -> int:
    """Run the backfill from the command line and print the summary."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.backfill",
        description="Backfill bronze from the parsed blobs in the application's S3 bucket.",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after this many blobs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and validate everything, write neither bronze nor quarantine",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_backfill(Settings.from_env(), limit=args.limit, dry_run=args.dry_run)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
