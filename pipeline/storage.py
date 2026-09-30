"""Where the lake lives: one root, either a local directory or an S3 prefix.

Every stage already takes a directory (`--data-dir`, `--bronze-dir`,
`PIPELINE_DATA_DIR`) and builds the rest of its paths under it. This module is
the one change that lets that directory be `s3://bucket/prefix` instead of
`./data`, with the layout under the prefix unchanged: `lake/bronze`,
`lake/silver`, `lake/quarantine`, `lake/run_metrics`, `catalog/`, `warehouse/`,
`mlruns/`, `mlruns-artifacts/`, `drift/`. Only the root moves, so a run on a
laptop and a run in a container differ by one environment variable and nothing
else.

`Location` is the whole interface. The operations on it are the ones the stages
actually perform and no others: list the files under a prefix, read bytes,
write bytes atomically, read and write a Parquet table, replace a directory,
delete a prefix, and ask whether something is there. It is deliberately not a
filesystem. There is no rename, no copy between roots, no partial read, no
append, because nothing here needs them and each one is a behaviour that
differs between a disk and an object store in a way a caller would have to know
about.

Why the operations hang off the location rather than off a separate filesystem
handle: every stage already threads a directory from its command line down to
its writer, and a handle would mean threading two things that must agree. A
`Location` knows which kind of root it came from, so `bronze_dir /
f"play_date={date}"` is the same line of code it was before.

Why boto3 rather than `pyarrow.fs.S3FileSystem` or fsspec's `s3fs`, which is the
deviation from the obvious choice worth arguing:

- Credentials. The requirement is the boto3 default chain, which is what the
  rest of this repository already uses (`pipeline.source.S3Source`,
  `pipeline.publish`, `scripts.fetch_catalog`). `pyarrow.fs.S3FileSystem` runs
  the AWS SDK for C++ chain instead, which resolves profiles, SSO sessions and
  container credentials on its own rules; two chains in one process is two ways
  for a task role to be picked up and one of them to be wrong.
- Testing. moto's in-process `mock_aws` intercepts botocore, so a boto3 client
  is mocked with no server and no endpoint override, which is how the bucket
  and queue tests in this repository already work. Neither the C++ SDK nor
  DuckDB's `httpfs` goes through botocore, so both need `moto_server` running on
  a port, which is a second kind of test fixture for the same coverage.
- Dependencies. `s3fs` pulls `aiobotocore`, which pins `botocore` to a narrow
  range; this project already depends on boto3 directly and on moto, and a
  three-way pin between them is a lock file that stops resolving on a Tuesday.
- Size. What goes through here is small: a bronze partition is a few dozen rows,
  a run-metrics row is one, the card index is a few megabytes. The streaming and
  multipart machinery pyarrow's filesystem buys is not machinery this traffic
  needs. The two places where the data is genuinely large, Spark's silver write
  and DuckDB's read of silver, do not come through this module at all: they use
  the `s3a://` connector and `httpfs` respectively, each with its own credential
  handling, and this module only derives the URIs for them.

Region comes from `AWS_REGION`, and an endpoint override from
`AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL`, both of which boto3 reads itself:
naming them here would be a second place that decides what the first one already
decided.

There is no rename on object storage, and that is the one behaviour a caller has
to know about. `replace_dir` writes the new files first and deletes the old ones
second, in that order. A reader that lists the prefix between the two steps sees
both sets: the new file and whatever old files the new set does not name. The
alternative order, delete then write, has a window in which the partition is
empty, and an empty partition reads as "this day has no games", which is a wrong
answer rather than a duplicated one. The partition is still the unit of work,
because the file name is fixed (`part-0.parquet`) and a rewrite of the same
partition therefore overwrites rather than adds: the overlap window only exists
when the set of file names changes, which for bronze it does not. The window is
documented rather than closed because closing it means a manifest, which means a
table format (Apache Iceberg, Delta Lake), which is the next design step and not
this one.

Local roots behave exactly as they did before this module existed: a write goes
to a temporary name in the destination directory and is moved into place with
`os.replace`, and a directory replace removes the directory and rewrites it.
"""

import contextlib
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from duckdb import DuckDBPyConnection
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

S3_SCHEME: Final = "s3://"
SPARK_SCHEME: Final = "s3a://"
# S3 takes at most this many keys in one `delete_objects` call.
DELETE_BATCH: Final = 1000
# The suffix that makes a synced store's artifact prefix out of its own name:
# `.../mlruns` keeps its metadata and `.../mlruns-artifacts` holds the files.
# A sibling rather than a child, so the sync never walks the artifacts.
ARTIFACTS_SUFFIX: Final = "-artifacts"
# The Hadoop S3 connector, matched to the Hadoop client PySpark 4 ships.
# Overridable, because the pairing is a property of the image and not of this code.
DEFAULT_SPARK_PACKAGES: Final = "org.apache.hadoop:hadoop-aws:3.4.2"
# Only used to spell an endpoint when `AWS_REGION` is unset; every client here
# resolves its own region from the environment the same way boto3 does.
DEFAULT_REGION: Final = "us-west-2"

_client: "S3Client | None" = None
# Files pulled down by `local_file`, and the directory they live in. Both are
# per process and both go away when it does.
_downloads: dict[str, Path] = {}
_scratch: tempfile.TemporaryDirectory[str] | None = None
# The artifact prefix that goes with each synced tracking store, keyed by the
# `file:` URI `tracking_store` handed out. Filled on entry and dropped on exit,
# so it only ever holds the store the running command is inside; `experiment_id`
# is the only reader. See `tracking_store` for why it is a lookup rather than a
# second yielded value.
_artifact_roots: dict[str, str] = {}


class StorageError(RuntimeError):
    """A root, or an operation on one, that cannot mean what the caller asked for."""


def s3_client() -> "S3Client":
    """The one S3 client this process uses, built on first use and kept.

    Built lazily rather than at import so that a run which never touches S3
    never resolves credentials, and cached because a client costs a session, a
    config load and an endpoint resolution, none of which are worth paying per
    partition. `reset_s3_client` exists for the tests, which stand a mock up
    after this module has already been imported.
    """
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or None)
    return _client


def reset_s3_client() -> None:
    """Forget the cached client and downloads, so the next call starts from scratch.

    For the tests, which stand a mock up and tear it down inside one process.
    """
    global _client
    _client = None
    _downloads.clear()


class Location:
    """A file or a directory in the lake, on the local disk or under an S3 prefix.

    Constructed from a `Path`, a string path or an `s3://bucket/key` string, and
    from another `Location`, so a caller that already holds one of those does not
    have to know which. Joining with `/` keeps the kind, which is what lets a
    stage build every path it needs from the root it was handed.

    A local location compares equal to the `Path` and the string it wraps. That
    is not laxity, it is the point: the conversion from `Path` to `Location` is
    meant to be invisible to code and tests that already held a path, so the
    only thing a reader has to look at is the S3 branch.
    """

    __slots__ = ("_bucket", "_key", "_path")

    _bucket: str
    _key: str
    _path: Path | None

    def __init__(self, value: "Location | Path | str") -> None:
        if isinstance(value, Location):
            self._bucket = value._bucket
            self._key = value._key
            self._path = value._path
            return
        if isinstance(value, Path):
            self._bucket = ""
            self._key = ""
            self._path = value
            return
        text = str(value)
        if not text.lower().startswith(S3_SCHEME):
            self._bucket = ""
            self._key = ""
            self._path = Path(text)
            return
        # Not stripped before the split: `s3:///lake` has no bucket, and
        # stripping first would promote `lake` into the bucket position and
        # write a production lake into whatever bucket happened to be named.
        bucket, _, key = text[len(S3_SCHEME) :].partition("/")
        key = key.strip("/")
        if not bucket:
            raise StorageError(f"an s3:// root needs a bucket: {text!r}")
        self._bucket = bucket
        self._key = key
        self._path = None

    # ------------------------------------------------------------ identity --

    @property
    def is_s3(self) -> bool:
        """True when this lives in object storage rather than on a disk."""
        return self._path is None

    @property
    def bucket(self) -> str:
        """The bucket; raises for a local location, which has none."""
        if self._path is not None:
            raise StorageError(f"{self} is a local path, not an S3 location")
        return self._bucket

    @property
    def key(self) -> str:
        """The key or prefix under the bucket; raises for a local location."""
        if self._path is not None:
            raise StorageError(f"{self} is a local path, not an S3 location")
        return self._key

    @property
    def path(self) -> Path:
        """The local path; raises for an S3 location.

        The raise is the useful half. DuckDB cannot open a database file over
        S3 and Spark cannot be handed a `Path`, so the places that genuinely
        need a local file ask for one here and get a clear error rather than a
        string that reads like a path and is not one.
        """
        if self._path is None:
            raise StorageError(f"{self} is an S3 location; it has no local path")
        return self._path

    def __fspath__(self) -> str:
        """The local path, for the standard library and for DuckDB; raises on S3."""
        return str(self.path)

    def __str__(self) -> str:
        if self._path is not None:
            return str(self._path)
        return (
            f"{S3_SCHEME}{self._bucket}/{self._key}" if self._key else f"{S3_SCHEME}{self._bucket}"
        )

    def __repr__(self) -> str:
        return f"Location({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Location):
            return (self._bucket, self._key, self._path) == (other._bucket, other._key, other._path)
        if isinstance(other, Path):
            return self._path is not None and self._path == other
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))

    def __truediv__(self, other: str) -> "Location":
        if self._path is not None:
            return Location(self._path / other)
        joined = str(PurePosixPath(self._key) / other) if self._key else str(other).strip("/")
        return Location(f"{S3_SCHEME}{self._bucket}/{joined}")

    @property
    def name(self) -> str:
        """The last component of the path or key."""
        if self._path is not None:
            return self._path.name
        return PurePosixPath(self._key).name if self._key else ""

    @property
    def parent(self) -> "Location":
        """The enclosing directory or prefix; the bucket root is its own parent."""
        if self._path is not None:
            return Location(self._path.parent)
        if not self._key:
            return self
        parent = str(PurePosixPath(self._key).parent)
        return Location(f"{S3_SCHEME}{self._bucket}" + ("" if parent == "." else f"/{parent}"))

    @property
    def spark_uri(self) -> str:
        """This location as Spark's Hadoop filesystem names it.

        `s3a://` rather than `s3://`: the Hadoop connector Spark uses registers
        itself under `s3a`, and the `s3` scheme in a Spark job means Amazon EMR's
        own connector, which is not on this classpath.
        """
        return SPARK_SCHEME + str(self)[len(S3_SCHEME) :] if self.is_s3 else str(self)

    # ------------------------------------------------------------ presence --

    def exists(self) -> bool:
        """Whether anything is here: a file, or a prefix with at least one object."""
        if self._path is not None:
            return self._path.exists()
        return self.is_file() or self.is_dir()

    def is_file(self) -> bool:
        """Whether this names one object that is there."""
        if self._path is not None:
            return self._path.is_file()
        from botocore.exceptions import ClientError

        try:
            s3_client().head_object(Bucket=self._bucket, Key=self._key)
        except ClientError:
            return False
        return True

    def is_dir(self) -> bool:
        """Whether this names a prefix that holds at least one object.

        There are no directories in object storage, so "is a directory" can only
        mean "something is under it". A prefix that has been emptied therefore
        answers False, which is the same answer a deleted local directory gives.
        """
        if self._path is not None:
            return self._path.is_dir()
        prefix = f"{self._key}/" if self._key else ""
        response = s3_client().list_objects_v2(Bucket=self._bucket, Prefix=prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0

    def mkdir(self, *, parents: bool = True, exist_ok: bool = True) -> None:
        """Create the directory locally; a no-op on S3, which has no directories."""
        if self._path is not None:
            self._path.mkdir(parents=parents, exist_ok=exist_ok)

    # -------------------------------------------------------------- listing --

    def iter_files(self, suffix: str | None = None) -> list["Location"]:
        """Every file under this prefix, recursively, sorted by path.

        Sorted because two callers depend on the order: the bronze scan for a
        source key reads partitions oldest first, and a test that compares two
        listings needs them to be comparable. `suffix` filters on the end of the
        name, which is the only filter anything here needs.
        """
        if self._path is not None:
            if not self._path.is_dir():
                return []
            found = [item for item in self._path.rglob("*") if item.is_file()]
            chosen = [item for item in found if suffix is None or item.name.endswith(suffix)]
            return [Location(item) for item in sorted(chosen)]
        keys = [
            key
            for key in self._all_keys()
            # A key ending in a slash is an empty-directory marker, not a file;
            # `upload_tree` says what it is for.
            if not key.endswith("/") and (suffix is None or key.rsplit("/", 1)[-1].endswith(suffix))
        ]
        return [Location(f"{S3_SCHEME}{self._bucket}/{key}") for key in keys]

    # ------------------------------------------------------------- content --

    def read_bytes(self) -> bytes:
        """This file's whole content."""
        if self._path is not None:
            return self._path.read_bytes()
        body = s3_client().get_object(Bucket=self._bucket, Key=self._key)["Body"]
        data = body.read()
        return bytes(data)

    def read_text(self, encoding: str = "utf-8") -> str:
        """This file's whole content, decoded."""
        return self.read_bytes().decode(encoding)

    def write_bytes(self, data: bytes) -> None:
        """Write this file, atomically as far as a reader is concerned.

        Locally that means a temporary name in the same directory and an
        `os.replace`, which is what the stages did before this module existed. On
        S3 a `PutObject` is already atomic: a reader sees the whole old object or
        the whole new one, never a half-written file, so there is nothing to
        stage.
        """
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            staged = self._path.parent / f".{self._path.name}.{uuid.uuid4().hex[:8]}.tmp"
            staged.write_bytes(data)
            os.replace(staged, self._path)
            return
        s3_client().put_object(Bucket=self._bucket, Key=self._key, Body=data)

    def write_text(self, text: str, encoding: str = "utf-8") -> None:
        """Write this file from a string, through the same atomic write."""
        self.write_bytes(text.encode(encoding))

    def read_table(self, columns: list[str] | None = None) -> pa.Table:
        """This Parquet file as an Arrow table."""
        if self._path is not None:
            return pq.read_table(self._path, columns=columns)
        return pq.read_table(pa.BufferReader(self.read_bytes()), columns=columns)

    def write_table(self, table: pa.Table, *, compression: str | None = None) -> None:
        """Write an Arrow table as one Parquet file, through the atomic write above."""
        self.write_bytes(table_bytes(table, compression=compression))

    # ------------------------------------------------------------ wholesale --

    def replace_dir(self, files: Mapping[str, bytes]) -> None:
        """Make this directory hold exactly `files`, keyed by name.

        The unit of work for a bronze partition. Locally the directory is
        removed and rewritten, which is what it was. On S3 the new objects are
        written first and the ones the new set does not name are deleted second;
        see the module docstring for why that order and what a reader can see in
        between.
        """
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)
            self._path.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                (self / name).write_bytes(data)
            return
        kept = set()
        for name, data in files.items():
            target = self / name
            target.write_bytes(data)
            kept.add(target.key)
        stale = [found.key for found in self.iter_files() if found.key not in kept]
        self._delete_keys(stale)

    def delete_prefix(self) -> None:
        """Remove everything under here, and here itself when it is a local directory."""
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)
            return
        # Every key and not only the files, so the empty-directory markers
        # `upload_tree` writes go with the rest: a local root is removed whole.
        self._delete_keys(self._all_keys())

    def _delete_keys(self, keys: list[str]) -> None:
        """Delete the named objects, in the batches the API takes."""
        for start in range(0, len(keys), DELETE_BATCH):
            batch = keys[start : start + DELETE_BATCH]
            if not batch:
                continue
            s3_client().delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )

    # ------------------------------------------------------------ transfers --

    def download_file(self, destination: Path) -> Path:
        """Copy this one file to a local path and return it."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self._path is not None:
            shutil.copyfile(self._path, destination)
            return destination
        s3_client().download_file(self._bucket, self._key, str(destination))
        return destination

    def _all_keys(self) -> list[str]:
        """Every key under this prefix, including the empty-directory markers.

        `iter_files` drops the markers because a marker is not a file and no
        caller wants one; the tree sync is the one place that has to see them.
        """
        prefix = f"{self._key}/" if self._key else ""
        keys: list[str] = []
        paginator = s3_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(keys)

    def download_tree(self, destination: Path) -> Path:
        """Copy everything under this prefix into a local directory and return it.

        Empty directories come back too, from the markers `upload_tree` left;
        see there for why a directory with nothing in it is worth carrying.
        """
        destination.mkdir(parents=True, exist_ok=True)
        if self._path is not None:
            if self._path.is_dir():
                shutil.copytree(self._path, destination, dirs_exist_ok=True)
            return destination
        prefix = f"{self._key}/" if self._key else ""
        for key in self._all_keys():
            target = destination / key[len(prefix) :]
            if key.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            Location(f"{S3_SCHEME}{self._bucket}/{key}").download_file(target)
        return destination

    def upload_tree(self, source: Path) -> None:
        """Copy a local directory to this prefix, leaving anything else under it alone.

        Not a mirror: nothing is deleted. The one caller is the MLflow sync,
        where the local copy came from this prefix in the first place and a
        deletion would mean a run disappearing because a file was not read back.

        An empty directory is written as a zero-byte object whose key ends in a
        slash, the convention every S3 tool already draws as a folder, because
        the sync has to round-trip a directory and not only its contents.
        MLflow's file store validates that every run holds `metrics`, `params`
        and `artifacts` subdirectories and treats a run missing any of them as a
        run that does not exist; with the artifacts on S3 the local `artifacts`
        directory is always empty, so without the markers the first command to
        sync a store would delete every run in it from the next command's view.
        `iter_files` skips the markers, so nothing else sees them.
        """
        if self._path is not None:
            if source.resolve() == self._path.resolve():
                return
            shutil.copytree(source, self._path, dirs_exist_ok=True)
            return
        for item in sorted(source.rglob("*")):
            relative = item.relative_to(source).as_posix()
            if item.is_file():
                (self / relative).upload_file(item)
            elif item.is_dir() and not any(item.iterdir()):
                marker = f"{self._key}/{relative}/" if self._key else f"{relative}/"
                s3_client().put_object(Bucket=self._bucket, Key=marker, Body=b"")

    def upload_file(self, source: Path) -> None:
        """Copy one local file here."""
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, self._path)
            return
        s3_client().upload_file(str(source), self._bucket, self._key)


AnyLocation = Location | Path | str


def location(value: AnyLocation) -> Location:
    """A `Location` from whatever the caller has: a path, a string, or one of these.

    Every public entry point coerces through this, which is what lets the tests
    and the existing callers keep passing `Path` objects while the stage bodies
    work in one type.
    """
    return value if isinstance(value, Location) else Location(value)


def table_bytes(table: pa.Table, *, compression: str | None = None) -> bytes:
    """One Arrow table serialized as a Parquet file in memory.

    In memory rather than to a file because the S3 write needs a body and the
    local write needs something to stage; the files this pipeline writes are a
    few dozen rows each, so the buffer is smaller than the page cache the
    alternative would use anyway.
    """
    sink = pa.BufferOutputStream()
    if compression is None:
        pq.write_table(table, sink)
    else:
        pq.write_table(table, sink, compression=cast(Any, compression))
    return bytes(sink.getvalue().to_pybytes())


def local_file(target: AnyLocation) -> Path:
    """A local path for a file that may be on S3, downloaded once per process.

    DuckDB opens a database file, not a stream: `meta.duckdb` on S3 has to be on
    a local disk before anything can query it. A local location is returned
    untouched, so nothing is copied on a laptop; an S3 location is downloaded
    into a scratch directory that the interpreter removes when it exits.

    Once per process, and cached, for two reasons. A stage is a process here
    (`pipeline.run_all` runs each one as a subprocess), so the cache lives
    exactly as long as the run that made it. And the agent opens its warehouse
    when it builds its tools and keeps the connection for the life of the
    service, so there is no scope a context manager could bracket; downloading
    per query would be a warehouse fetched per question.

    The assumption the cache makes is that nothing rewrites the file underneath
    a running process. That holds: the only writer is `pipeline.gold`, which
    builds the warehouse on local disk and uploads it, and never reads one back
    through here.
    """
    resolved = location(target)
    if not resolved.is_s3:
        return resolved.path
    cached = _downloads.get(str(resolved))
    if cached is not None and cached.is_file():
        return cached
    # The basename is kept exactly, in a directory of its own rather than with a
    # unique prefix, because DuckDB names a database after the file's stem: a
    # copy called `a1b2-meta.duckdb` is a catalog called `a1b2-meta`, and every
    # view dbt wrote as `meta.main.x` then fails to bind.
    holder = Path(_scratch_dir()) / uuid.uuid4().hex[:8]
    holder.mkdir(parents=True, exist_ok=True)
    local = holder / (resolved.name or "download")
    logger.info("downloading a copy to read locally", extra={"source": str(resolved)})
    resolved.download_file(local)
    _downloads[str(resolved)] = local
    return local


def _scratch_dir() -> str:
    """The process's scratch directory, made on first use and removed on exit."""
    global _scratch
    if _scratch is None:
        _scratch = tempfile.TemporaryDirectory(prefix="pra-lake-")
    return _scratch.name


@contextlib.contextmanager
def synced_dir(target: AnyLocation, *, write_back: bool = True) -> Iterator[Path]:
    """A local directory backed by `target`, downloaded on entry and uploaded on exit.

    The MLflow store, and nothing else. See `tracking_store` for why a synced
    directory rather than a tracking server, and for the single-writer
    assumption that makes it safe.

    `write_back=False` skips the upload, for the commands that only read the
    registry (`publish`, and the serving application loading the aliased model).
    A reader that uploaded the store back would be a second writer for the
    length of its own upload, which is exactly the thing the single-writer
    assumption asks nobody to be.

    Nothing is uploaded when the block raises either: a stage that died halfway
    has a half-written store, and the copy in the lake is the last one that a
    command finished.
    """
    resolved = location(target)
    if not resolved.is_s3:
        resolved.mkdir()
        yield resolved.path
        return
    with tempfile.TemporaryDirectory(prefix="pra-mlruns-") as scratch:
        local = Path(scratch)
        resolved.download_tree(local)
        logger.info("mlflow store downloaded", extra={"source": str(resolved)})
        yield local
        if write_back:
            resolved.upload_tree(local)
            logger.info("mlflow store uploaded", extra={"target": str(resolved)})


@contextlib.contextmanager
def tracking_store(uri: str, *, write_back: bool = True) -> Iterator[str]:
    """An MLflow tracking URI usable from here, syncing an `s3://` file store around the run.

    A tracking URI that is not an S3 location is yielded unchanged: a local
    `file:` directory and an MLflow server both already work, and this must not
    change either.

    An `s3://` URI means the file store lives in the lake. MLflow's file store
    wants a directory it can stat, list and rename, and object storage has none
    of those, so the prefix is downloaded to a temporary directory, used as
    `file:<temp>` for the length of the command, and uploaded back afterwards.

    Why the sync and not a tracking server: the server is the right answer and
    it is already the recommendation in `docs/orchestration-on-aws.md`, because
    the file store's own documentation says it is unsafe under concurrent
    writers. It is also an always-on service with a database behind it, and
    today there is exactly one writer, the nightly job, running one stage at a
    time under one run identifier. The sync is free, needs nothing standing up,
    and is reversible: point `MLFLOW_TRACKING_URI` at a server and none of this
    code runs. The single writer is the assumption, and it is an assumption and
    not a guarantee: two commands syncing the same prefix at once will each
    upload their own view and the later one wins for any file they both touched.
    The day there is a second writer is the day the server stops being a later
    step.

    Nothing is deleted on the way back up, so a run that was in the store before
    this command is still there afterwards even if the command never read it.

    Only the metadata is synced. The artifacts stay on S3 and are read and
    written in place, because a file store records absolute paths and a
    temporary directory is a different absolute path in every command;
    `experiment_id` is where that is arranged and why.

    The temporary directory is still temporary, and can stay that way: the only
    thing that used to outlive a command through it was an artifact, and
    artifacts are no longer in it. A stable path per run would buy nothing and
    would cost the guarantee that two commands never share a half-written copy.
    """
    if not uri.startswith(S3_SCHEME):
        yield uri
        return
    store = location(uri)
    with synced_dir(store, write_back=write_back) as local:
        tracking_uri = f"file:{local}"
        _artifact_roots[tracking_uri] = str(store.parent / (store.name + ARTIFACTS_SUFFIX))
        try:
            yield tracking_uri
        finally:
            _artifact_roots.pop(tracking_uri, None)


def artifact_root(tracking_uri: str) -> str | None:
    """Where a synced store's artifacts belong, or `None` when MLflow's default is right.

    `None` for a local directory, for a tracking server and for anything this
    process is not currently inside a `tracking_store` block for. Each of those
    already puts artifacts somewhere that is still there next time: a laptop
    writes them beside the runs and a server owns its own store.
    """
    return _artifact_roots.get(tracking_uri)


def experiment_id(name: str, tracking_uri: str) -> str:
    """The named experiment's id, created with an S3 artifact location when the store is synced.

    The one place an experiment is obtained, because an experiment's artifact
    location is written once, when it is created, and is then copied onto every
    run and every logged model underneath it. Getting it wrong in one command
    poisons the records the other commands read, so there is one call and not
    four.

    What goes wrong without this: MLflow's file store writes absolute paths. An
    experiment created inside a synced store records `artifact_location` as the
    temporary directory of the command that created it, every run under it
    records an `artifact_uri` under that, and every registered version points at
    the same place. The directory is gone by the time the next command runs, so
    `promote` cannot read a candidate's artifacts, `serve` cannot load
    `models:/win-probability@production`, and `drift` cannot open the reference
    it just wrote. Naming an `s3://` location instead makes every one of those
    URIs absolute in a place that outlives the command and is the same from any
    machine.

    The location is a sibling of the store (`mlruns` and `mlruns-artifacts`),
    not a directory inside it, and that is the whole reason for the sibling: the
    sync copies the store down and back on every command, so artifacts under it
    would mean downloading every model and figure ever logged in order to record
    one metric. Both sit under the lake root, so a lake is still one prefix to
    grant, to copy and to delete.

    The experiment name becomes a path component, so it is expected to read like
    one; the three this repository uses are `win-probability`, `drift` and the
    evaluation experiment, all plain identifiers.

    A local root keeps exactly the behaviour it had: no artifact location is
    named and MLflow puts the files beside the runs, which is what a laptop
    wants and what `mlflow ui` expects to find.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    root = artifact_root(tracking_uri)
    if root is None:
        return str(client.create_experiment(name))
    return str(client.create_experiment(name, artifact_location=f"{root}/{name}"))


def spark_configuration(*targets: AnyLocation) -> dict[str, str]:
    """The Spark settings an `s3a://` read or write needs, or nothing for local paths.

    Empty when every target is local, which is the common case and the one that
    must not change: a laptop run gets the session it always got.

    The credentials provider is the one that walks the same places boto3 walks
    (environment, profile, container and instance metadata), so a task role is
    picked up with no configuration. The connector itself is not on PySpark's
    classpath, so `spark.jars.packages` fetches it from Maven at session start;
    `PRA_SPARK_PACKAGES` overrides the coordinate for a cluster whose image
    already ships it, and for a run that has no network to Maven Central.
    """
    if not any(location(target).is_s3 for target in targets):
        return {}
    packages = os.environ.get("PRA_SPARK_PACKAGES", "").strip() or DEFAULT_SPARK_PACKAGES
    settings = {
        "spark.jars.packages": packages,
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": (
            "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider"
        ),
    }
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        # Only ever set for a test server: a real bucket is reached through the
        # regional endpoint the connector derives on its own.
        settings["spark.hadoop.fs.s3a.endpoint"] = endpoint
        settings["spark.hadoop.fs.s3a.path.style.access"] = "true"
    region = os.environ.get("AWS_REGION", "").strip()
    if region:
        settings["spark.hadoop.fs.s3a.endpoint.region"] = region
    return settings


def duckdb_s3_profile() -> dict[str, str]:
    """The three values a DuckDB S3 secret needs beyond the credential chain.

    dbt parses `profiles.yml` as YAML and only then renders each value, so the
    profile cannot decide whether to write an endpoint: every key it names has
    to be there with a usable value. This computes those values in one place
    instead, and `pipeline.gold` exports them for the profile to interpolate.

    The defaults are the production ones, the regional AWS endpoint over TLS
    with virtual-host addressing, so a real bucket needs nothing set. An
    endpoint in the environment (`AWS_ENDPOINT_URL_S3`, else `AWS_ENDPOINT_URL`,
    the same two boto3 reads) replaces them with path-style access, which is
    what a local S3 stand-in such as moto, LocalStack or MinIO needs and what a
    real bucket never does.
    """
    region = os.environ.get("AWS_REGION", "").strip() or DEFAULT_REGION
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL") or ""
    if not endpoint:
        return {
            "endpoint": f"s3.{region}.amazonaws.com",
            "url_style": "vhost",
            "use_ssl": "true",
        }
    return {
        "endpoint": endpoint.split("://", 1)[-1],
        "url_style": "path",
        "use_ssl": "true" if endpoint.startswith("https") else "false",
    }


def duckdb_connect(warehouse: AnyLocation, *, read_only: bool = True) -> "DuckDBPyConnection":
    """Open the warehouse, downloading it and teaching DuckDB about S3 if it has to.

    Every reader of the warehouse goes through here, and the reason is a case
    that is easy to miss: the gold build over an S3 lake leaves views in
    `meta.duckdb` whose definitions are `read_parquet('s3://...')`. The
    `mart_pipeline_health` the quality gate reads is one of them. Downloading the
    file is therefore not enough; the connection that opens it needs `httpfs` and
    the same credential-chain secret dbt used, or the view resolves to a
    permission error and the gate reports a warehouse that was built correctly as
    unreadable.

    The extensions are loaded only when the warehouse is on S3, so a local read
    stays offline and starts as fast as it did.
    """
    import duckdb

    target = location(warehouse)
    connection = duckdb.connect(str(local_file(target)), read_only=read_only)
    if not target.is_s3:
        return connection
    profile = duckdb_s3_profile()
    region = os.environ.get("AWS_REGION", "").strip() or DEFAULT_REGION
    connection.execute("INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws")
    connection.execute(
        "CREATE OR REPLACE SECRET pra_lake ("
        "TYPE s3, PROVIDER credential_chain, "
        f"REGION '{region}', ENDPOINT '{profile['endpoint']}', "
        f"URL_STYLE '{profile['url_style']}', USE_SSL {profile['use_ssl']})"
    )
    return connection


def describe(root: AnyLocation) -> dict[str, Any]:
    """The root as a log field: what it is and, for S3, which bucket holds it.

    A bucket name is an identifier, not a secret, but it is also not something
    this repository commits, so it is logged at run time and never written down.
    """
    resolved = location(root)
    return {"root": str(resolved), "kind": "s3" if resolved.is_s3 else "local"}
