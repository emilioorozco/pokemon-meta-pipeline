"""Quarantine: the rejected blob kept as received, next to a sidecar saying why.

Why keep the body at all: every rejection here is a question for the producer
("which key does this contract error come from, and what did the blob actually
look like"), and re-fetching from S3 later is not the same object once the
application has rewritten or deleted it. The body is the evidence.

Why the body is safe to keep: quarantine lives under `PIPELINE_DATA_DIR`, which
is either the gitignored `data/` directory or a private bucket prefix, and is
never published either way. It is the one part of the lake that holds a blob as
the producer sent it, so the bucket it lands in on an `s3://` root is the same
private bucket the rest of the lake is in, with the same access policy. The body
is
written before anonymization because a blob that failed validation cannot be
trusted to anonymize correctly, so it must not be treated as if it had been.

Why the sidecar is separate: the sidecar is the part that gets read, pasted into
an issue and grepped across runs, so it must never carry a handle. It holds only
the source key, the reason, a validator summary and timestamps. Callers pass
`ContractError.summary()` (paths and messages, no values) or the bronze leak
paths (already masked to `<key>`); nothing that echoes blob content belongs in
`detail`.

Layout, one pair of files per rejected object:

    quarantine/<reason>/<source key, slashes as __>.json        body as received
    quarantine/<reason>/<source key, slashes as __>.meta.json   the sidecar

The key is flattened rather than nested so that one reason directory lists every
object that failed that way, and re-quarantining the same key overwrites its
pair instead of accumulating copies, which keeps a re-run idempotent.
"""

import json
import logging
from datetime import datetime
from typing import Any, Final

from pipeline.storage import AnyLocation, Location, location

logger = logging.getLogger(__name__)

# Reason codes. One per way a blob can fail to reach bronze.
INVALID_JSON: Final = "invalid_json"
CONTRACT_VIOLATION: Final = "contract_violation"
V1_BLOB: Final = "v1_blob"
HANDLE_LEAK_CHECK_FAILED: Final = "handle_leak_check_failed"
WRITE_FAILED: Final = "write_failed"

REASONS: Final = (
    INVALID_JSON,
    CONTRACT_VIOLATION,
    V1_BLOB,
    HANDLE_LEAK_CHECK_FAILED,
    WRITE_FAILED,
)

SIDECAR_SUFFIX: Final = ".meta.json"
BODY_SUFFIX: Final = ".json"
SEPARATOR: Final = "__"


def flatten_key(source_key: str) -> str:
    """An S3 key as one filename component: slashes become `__`."""
    return source_key.replace("/", SEPARATOR)


def write_quarantine(
    quarantine_dir: AnyLocation,
    source_key: str,
    raw_bytes: bytes,
    reason: str,
    detail: str,
    when: datetime,
    *,
    contract_version_seen: Any = None,
) -> Location:
    """Write the body and its sidecar under `quarantine_dir/<reason>/`; return the body path.

    `detail` must already be handle-free; see the module docstring. `when` is the
    run time, passed in so every record of one run carries the same timestamp.
    """
    if reason not in REASONS:
        raise ValueError(f"unknown quarantine reason: {reason!r}")
    reason_dir = location(quarantine_dir) / reason
    reason_dir.mkdir()

    # The source keys already end in .json, so the suffix is stripped before the
    # two names are built; otherwise the body would land as `....json.json`.
    stem = flatten_key(source_key).removesuffix(BODY_SUFFIX)
    body_path = reason_dir / f"{stem}{BODY_SUFFIX}"
    sidecar_path = reason_dir / f"{stem}{SIDECAR_SUFFIX}"

    body_path.write_bytes(raw_bytes)
    sidecar = {
        "source_key": source_key,
        "reason": reason,
        "detail": detail,
        "quarantined_at": when.isoformat(),
        "contract_version_seen": contract_version_seen,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    logger.warning("quarantined %s as %s: %s", source_key, reason, detail)
    return body_path
