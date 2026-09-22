"""Backfill against a moto S3 bucket: what lands, what is quarantined, what is counted.

moto serves a real boto3 client against an in-process fake, so the listing, the
pagination and the version ids are exercised without an AWS account and without
network access; CI needs no credentials beyond the fake ones this module sets.

The bucket is loaded with one object per route the backfill can take, so a single
run asserts the whole routing table at once. Handles are invented (PlayerA,
PlayerB) and every blob is built through the contract models, so a fixture that
drifts from the contract fails here rather than in production.
"""

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq
import pytest
from moto import mock_aws

from pipeline import backfill, quarantine
from pipeline.backfill import BackfillSummary, run_backfill
from pipeline.bronze import read_smoke
from pipeline.contract import (
    CardRef,
    Decklist,
    DecklistSource,
    Entry,
    GameSummary,
    ParsedBlobV1,
    ParsedBlobV2,
    RoleStats,
    Segment,
    SideStats,
    SubEntry,
)
from pipeline.settings import Settings, SettingsError
from pipeline.source import list_parsed_keys

BUCKET = "pra-test-bucket"
PREFIX = "parsed/"
REGION: Final = "us-west-2"
KEY = b"test-key-not-a-real-secret"
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
ME = "PlayerA"
OPPONENT = "PlayerB"

VALID_ONE = "parsed/user-1/game-1.json"
VALID_TWO = "parsed/user-1/game-2.json"
FULL_DECKLISTS = "parsed/user-1/game-3.json"
V1 = "parsed/user-1/game-4.json"
BROKEN_CONTRACT = "parsed/user-1/game-5.json"
NOT_JSON = "parsed/user-1/game-6.json"
PASTED_OPPONENT_LIST = "parsed/user-1/game-7.json"
NOT_A_BLOB = "parsed/user-1/notes.txt"


def side_stats(**overrides: Any) -> SideStats:
    base: dict[str, Any] = {
        "cards_drawn": 12,
        "energy_attached": 4,
        "damage_dealt": 300,
        "knockouts": 3,
        "prizes_taken": 6,
        "mulligans": 0,
        "turns_taken": 8,
        "pokemon_played": ["Pikachu"],
        "cards_played": ["Professor's Research"],
        "evolutions": [],
        "attacks": ["Thunder Shock"],
    }
    return SideStats(**{**base, **overrides})


def segments() -> list[Segment]:
    entry = Entry(
        line=4,
        text=f"{ME} used Thunder Shock for 60 damage",
        kind="attack",
        actor=ME,
        fields={"attack": "Thunder Shock", "damage": 60},
        subs=[
            SubEntry(
                line=5,
                text=f"{OPPONENT}'s Pokemon took 60 damage",
                kind="took_damage",
                actor=OPPONENT,
                fields={"n": 60},
                details=[f"{OPPONENT} placed 60 damage counters"],
            )
        ],
    )
    return [
        Segment(kind="setup", title="Setup", entries=[]),
        Segment(kind="turn", turn_number=1, player=ME, title=f"{ME}'s Turn", entries=[entry]),
    ]


def decklist(source: str) -> Decklist:
    return Decklist(
        cards=[CardRef(card_id="sv1-1", name="Pikachu", set="SV1", number="1", count=4)],
        card_count=60,
        complete=True,
        source=DecklistSource(source),
    )


def blob_v2(game_id: str, played_at: str, *, full_decklists: bool = False) -> dict[str, Any]:
    """A valid v2 blob as it arrives on the wire: camelCase, optional keys absent."""
    summary = GameSummary(
        game_id=game_id,
        user_id="user-1",
        uploaded_at=played_at,
        played_at=played_at,
        export_variant="debug" if full_decklists else "stock",
        parser_version=7,
        unparsed_count=1,
        players=[ME, OPPONENT],
        my_side=0,
        result="win",
        end_reason="prizes",
        winner=ME,
        turn_count=2,
        stats=RoleStats(me=side_stats(), opponent=side_stats(knockouts=1)),
        has_full_decklists=full_decklists,
    )
    blob = ParsedBlobV2(
        schema_version=2,
        summary=summary,
        segments=segments(),
        stats_by_player={ME: side_stats(), OPPONENT: side_stats(knockouts=1)},
        unparsed_lines=[f"{ME} did something no pattern matched"],
        extras={"Note": [f"a trailer line naming {OPPONENT}"]},
        my_decklist=decklist("debug") if full_decklists else None,
        opponent_decklist=decklist("debug") if full_decklists else None,
    )
    return blob.model_dump(mode="json", by_alias=True, exclude_none=True)


def blob_with_pasted_opponent_list() -> dict[str, Any]:
    """A stock game whose uploader pasted the opponent's list, so `hasFullDecklists` is false."""
    blob = blob_v2("game-7", "2026-09-03T15:00:00.000Z")
    blob["summary"]["opponentDecklistSource"] = DecklistSource.PASTE.value
    blob["opponentDecklist"] = decklist("paste").model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    return blob


def blob_v1() -> dict[str, Any]:
    """The pre-contract shape: no schemaVersion, no summary, therefore no play date."""
    blob = ParsedBlobV1(
        segments=segments(),
        stats_by_player={ME: side_stats(), OPPONENT: side_stats()},
        unparsed_lines=[],
        extras={},
    )
    return blob.model_dump(mode="json", by_alias=True, exclude_none=True)


def blob_with_unknown_kind() -> dict[str, Any]:
    """A v2 blob whose first entry carries an action kind the contract does not know."""
    blob = blob_v2("game-5", "2026-09-01T21:00:00.000Z")
    blob["segments"][1]["entries"][0]["kind"] = "teleport"
    return blob


@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials so a misconfigured run can never reach a real account."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def s3(aws_credentials: None) -> Iterator[Any]:
    """A versioned bucket holding one object per route the backfill can take."""
    import boto3

    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        client.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
        bodies: dict[str, bytes] = {
            VALID_ONE: _json(blob_v2("game-1", "2026-09-01T18:22:00.000Z")),
            VALID_TWO: _json(blob_v2("game-2", "2026-09-02T09:10:00.000Z")),
            FULL_DECKLISTS: _json(
                blob_v2("game-3", "2026-09-02T11:00:00.000Z", full_decklists=True)
            ),
            V1: _json(blob_v1()),
            BROKEN_CONTRACT: _json(blob_with_unknown_kind()),
            NOT_JSON: b"this is not json, it is a log line",
            NOT_A_BLOB: b"a note that is not a blob",
        }
        for key, body in bodies.items():
            client.put_object(Bucket=BUCKET, Key=key, Body=body)
        yield client


@pytest.fixture
def settings() -> Settings:
    return Settings(bucket=BUCKET, hmac_key=KEY, prefix=PREFIX, region=REGION)


@pytest.fixture
def lake(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "bronze", tmp_path / "quarantine"


def _json(blob: dict[str, Any]) -> bytes:
    return json.dumps(blob).encode("utf-8")


def comparable(summary: BackfillSummary) -> BackfillSummary:
    """The summary without its duration, which differs between two identical runs."""
    return replace(summary, duration_s=0.0)


def landed_rows(bronze_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(bronze_dir.rglob("*.parquet")):
        rows += pq.read_table(path).to_pylist()
    return rows


def row_for(bronze_dir: Path, game_id: str) -> dict[str, Any]:
    return next(row for row in landed_rows(bronze_dir) if row["game_id"] == game_id)


def parquet_text(bronze_dir: Path) -> str:
    """Every value in every written row, as one JSON string, for leak assertions."""
    return json.dumps(landed_rows(bronze_dir), default=str)


def sidecar(quarantine_dir: Path, reason: str, source_key: str) -> dict[str, Any]:
    stem = quarantine.flatten_key(source_key).removesuffix(".json")
    path = quarantine_dir / reason / f"{stem}.meta.json"
    return dict(json.loads(path.read_text()))


def test_only_json_keys_are_listed(s3: Any, settings: Settings) -> None:
    keys = [obj.key for obj in list_parsed_keys(s3, BUCKET, PREFIX)]

    assert NOT_A_BLOB not in keys
    assert keys == [VALID_ONE, VALID_TWO, FULL_DECKLISTS, V1, BROKEN_CONTRACT, NOT_JSON]
    assert all(obj.size > 0 and obj.last_modified for obj in list_parsed_keys(s3, BUCKET, PREFIX))


def test_a_run_routes_every_object(s3: Any, settings: Settings, lake: tuple[Path, Path]) -> None:
    bronze_dir, quarantine_dir = lake

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert (summary.read, summary.landed, summary.full_decklists_landed) == (6, 3, 1)
    assert summary.quarantined == {"contract_violation": 1, "invalid_json": 1, "v1_blob": 1}
    assert summary.partitions == {"2026-09-01": 1, "2026-09-02": 2}
    assert read_smoke(bronze_dir) == [("2026-09-01", 1), ("2026-09-02", 2)]
    assert summary.duration_s >= 0


def test_a_full_decklist_game_lands_with_both_lists(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """The modified client's game is ingested like any other; the flag only counts it."""
    bronze_dir, quarantine_dir = lake

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert summary.full_decklists_landed == 1
    row = row_for(bronze_dir, "game-3")
    assert row["summary"]["has_full_decklists"] is True
    assert row["my_decklist"]["card_count"] == 60
    assert row["opponent_decklist"]["card_count"] == 60
    assert row["opponent_decklist"]["cards"][0]["name"] == "Pikachu"


def test_a_pasted_opponent_list_lands_and_is_not_counted(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """`full_decklists_landed` follows the summary flag, not the presence of a list."""
    bronze_dir, quarantine_dir = lake
    s3.put_object(
        Bucket=BUCKET, Key=PASTED_OPPONENT_LIST, Body=_json(blob_with_pasted_opponent_list())
    )

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert (summary.read, summary.landed, summary.full_decklists_landed) == (7, 4, 1)
    row = row_for(bronze_dir, "game-7")
    assert row["summary"]["has_full_decklists"] is False
    assert row["summary"]["opponent_decklist_source"] == "paste"
    assert row["opponent_decklist"]["source"] == "paste"
    assert row["my_decklist"] is None


def test_the_landed_rows_carry_their_source_version_and_no_handle(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, _ = lake

    run_backfill(settings, s3, bronze_dir, lake[1], now=NOW)

    rows = landed_rows(bronze_dir)
    assert sorted(row["game_id"] for row in rows) == ["game-1", "game-2", "game-3"]
    assert all(row["source_version_id"] for row in rows)
    assert all(row["source_last_modified"] is not None for row in rows)
    assert {row["source_key"] for row in rows} == {VALID_ONE, VALID_TWO, FULL_DECKLISTS}

    written = parquet_text(bronze_dir)
    assert ME not in written
    assert OPPONENT not in written
    # The handles were there to begin with: the tokens replaced them, not nothing.
    assert "used Thunder Shock" in written


def test_quarantine_keeps_the_body_and_a_handle_free_sidecar(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, quarantine_dir = lake

    run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    bodies = {
        "invalid_json": quarantine_dir / "invalid_json" / "parsed__user-1__game-6.json",
        "v1_blob": quarantine_dir / "v1_blob" / "parsed__user-1__game-4.json",
        "contract_violation": quarantine_dir / "contract_violation" / "parsed__user-1__game-5.json",
    }
    assert all(path.is_file() for path in bodies.values())
    assert bodies["invalid_json"].read_bytes() == b"this is not json, it is a log line"

    broken = sidecar(quarantine_dir, "contract_violation", BROKEN_CONTRACT)
    assert broken["source_key"] == BROKEN_CONTRACT
    assert broken["reason"] == "contract_violation"
    assert broken["contract_version_seen"] == 2
    assert "segments.1.entries.0.kind" in broken["detail"]

    v1 = sidecar(quarantine_dir, "v1_blob", V1)
    assert v1["contract_version_seen"] is None
    assert "re-parse it upstream" in v1["detail"]
    assert v1["quarantined_at"] == NOW.isoformat()

    for reason, key in (
        ("invalid_json", NOT_JSON),
        ("v1_blob", V1),
        ("contract_violation", BROKEN_CONTRACT),
    ):
        text = json.dumps(sidecar(quarantine_dir, reason, key))
        assert ME not in text and OPPONENT not in text


def test_running_twice_lands_the_same_rows(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, quarantine_dir = lake

    first = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)
    second = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert comparable(first) == comparable(second)
    assert read_smoke(bronze_dir) == [("2026-09-01", 1), ("2026-09-02", 2)]
    assert len(sorted(bronze_dir.rglob("*.parquet"))) == 2


def test_limit_stops_after_that_many_blobs(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    summary = run_backfill(settings, s3, lake[0], lake[1], now=NOW, limit=2)

    assert (summary.read, summary.landed) == (2, 2)
    assert summary.quarantined == {}


def test_a_dry_run_reports_the_same_counts_and_writes_nothing(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, quarantine_dir = lake

    dry = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW, dry_run=True)

    assert list(bronze_dir.rglob("*.parquet")) == []
    assert not quarantine_dir.exists()

    wet = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert comparable(dry) == comparable(wet)


def test_a_surviving_handle_quarantines_the_game_instead_of_landing_it(
    s3: Any, settings: Settings, lake: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An anonymizer that rewrites nothing must land nothing and name every game."""
    bronze_dir, quarantine_dir = lake
    monkeypatch.setattr(backfill, "anonymize", lambda blob, key: blob)

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert summary.landed == 0
    assert summary.partitions == {}
    assert summary.quarantined["handle_leak_check_failed"] == 3
    assert read_smoke(bronze_dir) == []
    leaked = quarantine_dir / "handle_leak_check_failed"
    assert sorted(path.name for path in leaked.glob("*.json")) == [
        "parsed__user-1__game-1.json",
        "parsed__user-1__game-1.meta.json",
        "parsed__user-1__game-2.json",
        "parsed__user-1__game-2.meta.json",
        "parsed__user-1__game-3.json",
        "parsed__user-1__game-3.meta.json",
    ]
    detail = sidecar(quarantine_dir, "handle_leak_check_failed", VALID_ONE)["detail"]
    assert "path(s) still hold a handle" in detail
    assert ME not in detail and OPPONENT not in detail


def test_one_leaking_game_does_not_stop_the_others(
    s3: Any, settings: Settings, lake: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch is re-checked per game, so a single leak costs one game, not the run."""
    bronze_dir, quarantine_dir = lake
    real = backfill.anonymize

    def leak_one(blob: dict[str, Any], key: bytes) -> dict[str, Any]:
        if blob.get("summary", {}).get("gameId") == "game-1":
            return blob
        return real(blob, key)

    monkeypatch.setattr(backfill, "anonymize", leak_one)

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert summary.landed == 2
    assert summary.partitions == {"2026-09-02": 2}
    assert summary.quarantined["handle_leak_check_failed"] == 1
    assert read_smoke(bronze_dir) == [("2026-09-02", 2)]
    assert ME not in parquet_text(bronze_dir)


def test_an_anonymizer_that_breaks_the_contract_is_a_contract_violation(
    s3: Any, settings: Settings, lake: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    bronze_dir, quarantine_dir = lake

    def drop_summary(blob: dict[str, Any], key: bytes) -> dict[str, Any]:
        return {name: value for name, value in blob.items() if name != "summary"}

    monkeypatch.setattr(backfill, "anonymize", drop_summary)

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert summary.landed == 0
    assert summary.quarantined["contract_violation"] == 4
    detail = sidecar(quarantine_dir, "contract_violation", VALID_ONE)["detail"]
    assert detail.startswith("after anonymization: ")


def test_a_failed_write_quarantines_the_batch_rather_than_losing_it(
    s3: Any, settings: Settings, lake: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    bronze_dir, quarantine_dir = lake

    def no_disk(*args: Any, **kwargs: Any) -> dict[str, int]:
        raise OSError("no space left on device")

    monkeypatch.setattr(backfill, "write_partitions", no_disk)

    summary = run_backfill(settings, s3, bronze_dir, quarantine_dir, now=NOW)

    assert summary.landed == 0
    assert summary.quarantined["write_failed"] == 3
    assert sidecar(quarantine_dir, "write_failed", VALID_ONE)["detail"] == "OSError"


def test_an_empty_prefix_is_a_run_that_does_nothing(s3: Any, lake: tuple[Path, Path]) -> None:
    empty = Settings(bucket=BUCKET, hmac_key=KEY, prefix="parsed/user-2/", region=REGION)

    summary = run_backfill(empty, s3, lake[0], lake[1], now=NOW)

    assert (summary.read, summary.landed, summary.quarantined) == (0, 0, {})
    assert read_smoke(lake[0]) == []


def test_the_summary_prints_one_line_per_field(
    s3: Any, settings: Settings, lake: tuple[Path, Path]
) -> None:
    summary = run_backfill(settings, s3, lake[0], lake[1], now=NOW)

    lines = str(summary).splitlines()
    assert [line.split(":")[0] for line in lines] == [
        "read",
        "landed",
        "quarantined",
        "full_decklists_landed",
        "partitions",
        "duration_s",
    ]
    assert "invalid_json=1" in lines[2]
    assert lines[3] == "full_decklists_landed: 1"
    assert lines[4] == "partitions: 2026-09-01=1 2026-09-02=2"
    assert str(BackfillSummary()).splitlines()[2] == "quarantined: none"


def test_the_command_line_prints_the_summary(
    s3: Any,
    lake: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PRA_BUCKET", BUCKET)
    monkeypatch.setenv("PRA_PREFIX", PREFIX)
    monkeypatch.setenv("HANDLE_HMAC_KEY", "test-key-not-a-real-secret")
    monkeypatch.setattr(backfill, "BRONZE_DIR", lake[0])
    monkeypatch.setattr(backfill, "QUARANTINE_DIR", lake[1])
    monkeypatch.setattr(backfill, "_default_client", lambda settings: s3)

    exit_code = backfill.main(["--limit", "2"])

    assert exit_code == 0
    assert "read: 2" in capsys.readouterr().out


def test_an_unknown_quarantine_reason_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown quarantine reason"):
        quarantine.write_quarantine(tmp_path, "parsed/a/b.json", b"{}", "oops", "why", NOW)


def test_settings_name_every_missing_variable_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRA_BUCKET", raising=False)
    monkeypatch.delenv("HANDLE_HMAC_KEY", raising=False)

    with pytest.raises(SettingsError) as raised:
        Settings.from_env()

    assert raised.value.missing == ["PRA_BUCKET", "HANDLE_HMAC_KEY"]
    assert "PRA_BUCKET" in str(raised.value)
    assert "HANDLE_HMAC_KEY" in str(raised.value)


def test_an_empty_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRA_BUCKET", BUCKET)
    monkeypatch.setenv("HANDLE_HMAC_KEY", "")

    with pytest.raises(SettingsError) as raised:
        Settings.from_env()

    assert raised.value.missing == ["HANDLE_HMAC_KEY"]


def test_settings_defaults_and_secrecy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRA_BUCKET", BUCKET)
    monkeypatch.setenv("HANDLE_HMAC_KEY", "a-key")
    monkeypatch.delenv("PRA_PREFIX", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    loaded = Settings.from_env()

    assert (loaded.bucket, loaded.prefix, loaded.region) == (BUCKET, "parsed/", "us-west-2")
    assert loaded.hmac_key == b"a-key"
    # The key must not be reachable by printing the settings anywhere.
    assert "a-key" not in repr(loaded)
