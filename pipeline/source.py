"""Reading the parsed-blob objects the backfill ingests, from S3 or from a directory.

The two things the backfill needs from a source, and nothing else: an ordered
listing of the objects under a prefix, and one object's body with the lineage
the bronze row carries. That pair is the `Source` protocol, and the backfill is
written against it so the same walk runs over the application's bucket
(`S3Source`) or over a directory of files (`LocalSource`) with no other
difference: the same validation, the same anonymization, the same routing.

Why a local source exists at all: the committed fixtures are real games in the
contract's shape, so running the stage over them proves the stage end to end on
a machine with no bucket, no credentials and no network. That is what a demo and
a first clone need, and it costs one small class rather than a parallel code
path.

Why a paginator rather than a single `list_objects_v2`: the call returns at most
1000 keys and the source grows past that, so a plain call would silently ingest
a prefix of the bucket. The paginator also yields page by page, so a run that
stops early (`--limit`) never materializes the whole listing.

Why the version id comes from `get_object` and not from the listing:
`list_objects_v2` does not report versions, and `list_object_versions` would
report every historical version of every key. The `GetObject` response carries
the id of the version actually read, which is the one bronze should record. It
is absent when the bucket is not versioned, hence `str | None`; a local file has
no version at all, so `LocalSource` reports `None` for the same reason.

Why `get_blob` hands back the undecoded bytes as well as the decoded body: a
body that is not JSON still has to be quarantined as received, so the bytes must
survive the decode failure. Decoding is therefore a method on the result rather
than something `get_blob` does on the way out.

This module never writes: not to the source bucket, not to the source directory.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client

BLOB_SUFFIX: Final = ".json"
ENCODING: Final = "utf-8"


@dataclass(frozen=True)
class SourceObject:
    """One listed object: enough to read it and to record where a row came from."""

    key: str
    size: int
    last_modified: datetime | None = None


@dataclass(frozen=True)
class SourceBlob:
    """One object as read: the bytes exactly as stored, plus the version id read."""

    key: str
    body: bytes
    version_id: str | None = None

    def decode(self) -> Any:
        """The body as decoded JSON.

        Raises `json.JSONDecodeError` when it is not JSON and `UnicodeDecodeError`
        when it is not UTF-8; the caller quarantines both the same way. Whatever
        JSON holds is returned as it comes, so a body that is valid JSON but not
        an object reaches the contract and fails there instead of here.
        """
        return json.loads(self.body.decode(ENCODING))


class Source(Protocol):
    """What the backfill consumes: a listing and a reader, plus a name for the log line."""

    @property
    def label(self) -> str:
        """Where this run is reading from, safe to log."""

    def list(self) -> Iterator[SourceObject]:
        """Every blob under the source, in a stable order."""

    def get(self, key: str) -> SourceBlob:
        """One blob, as stored."""


@dataclass(frozen=True)
class S3Source:
    """The application's bucket: the listing and the read the rest of this module does."""

    client: "S3Client"
    bucket: str
    prefix: str

    @property
    def label(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def list(self) -> Iterator[SourceObject]:
        return list_parsed_keys(self.client, self.bucket, self.prefix)

    def get(self, key: str) -> SourceBlob:
        return get_blob(self.client, self.bucket, key)


@dataclass(frozen=True)
class LocalSource:
    """A directory of blob files, read recursively in sorted order.

    Keys are paths relative to the root, so a quarantine sidecar names the file
    the way the operator typed it rather than by absolute path. The last-modified
    time is the file's mtime, which is only ever lineage (bronze partitions on
    `summary.playedAt`), and there is no version id to report.
    """

    root: Path

    @property
    def label(self) -> str:
        return str(self.root)

    def list(self) -> Iterator[SourceObject]:
        for path in sorted(self.root.rglob(f"*{BLOB_SUFFIX}")):
            if not path.is_file():
                continue
            stat = path.stat()
            yield SourceObject(
                key=path.relative_to(self.root).as_posix(),
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime, UTC),
            )

    def get(self, key: str) -> SourceBlob:
        return SourceBlob(key=key, body=(self.root / key).read_bytes(), version_id=None)


def list_parsed_keys(s3: "S3Client", bucket: str, prefix: str) -> Iterator[SourceObject]:
    """Yield every `.json` object under `prefix`, page by page, in listing order.

    Keys that are not blobs are skipped rather than reported: the prefix also
    holds directory placeholders and whatever else the application writes there,
    and a key that is not a blob is not a failed blob.
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(BLOB_SUFFIX):
                continue
            yield SourceObject(
                key=key, size=int(item.get("Size", 0)), last_modified=item.get("LastModified")
            )


def get_blob(s3: "S3Client", bucket: str, key: str) -> SourceBlob:
    """Read one object: its body and the version id of the version served."""
    response = s3.get_object(Bucket=bucket, Key=key)
    return SourceBlob(key=key, body=response["Body"].read(), version_id=response.get("VersionId"))
