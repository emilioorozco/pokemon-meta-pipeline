"""Reading the application's parsed-blob objects out of S3.

The two things the backfill needs from the source, and nothing else: an ordered
listing of the objects under a prefix, and one object's body with the lineage
the bronze row carries.

Why a paginator rather than a single `list_objects_v2`: the call returns at most
1000 keys and the source grows past that, so a plain call would silently ingest
a prefix of the bucket. The paginator also yields page by page, so a run that
stops early (`--limit`) never materializes the whole listing.

Why the version id comes from `get_object` and not from the listing:
`list_objects_v2` does not report versions, and `list_object_versions` would
report every historical version of every key. The `GetObject` response carries
the id of the version actually read, which is the one bronze should record. It
is absent when the bucket is not versioned, hence `str | None`.

Why `get_blob` hands back the undecoded bytes as well as the decoded body: a
body that is not JSON still has to be quarantined as received, so the bytes must
survive the decode failure. Decoding is therefore a method on the result rather
than something `get_blob` does on the way out.

This module never writes to the source bucket.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client

BLOB_SUFFIX: Final = ".json"
ENCODING: Final = "utf-8"


@dataclass(frozen=True)
class S3Object:
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


def list_parsed_keys(s3: "S3Client", bucket: str, prefix: str) -> Iterator[S3Object]:
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
            yield S3Object(
                key=key, size=int(item.get("Size", 0)), last_modified=item.get("LastModified")
            )


def get_blob(s3: "S3Client", bucket: str, key: str) -> SourceBlob:
    """Read one object: its body and the version id of the version served."""
    response = s3.get_object(Bucket=bucket, Key=key)
    return SourceBlob(key=key, body=response["Body"].read(), version_id=response.get("VersionId"))
