"""Enrich: the on-disk cache is the source of truth; ids already landed are never re-fetched,
and parsing indexes per-agent fields by seat."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.legacy.kaggle import enrich

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def make_episode(ep_id: int, *, agents: int = 2) -> dict:
    seats = [
        # Kaggle omits `index` for seat 0
        {"submissionId": 111, "teamId": 1, "initialScore": 1000.0, "updatedScore": 990.0},
        {
            "index": 1,
            "submissionId": 222,
            "teamId": 2,
            "initialScore": 1100.0,
            "updatedScore": 1110.0,
        },
    ]
    return {
        "id": ep_id,
        "createTime": "2026-07-03T06:08:10Z",
        "endTime": "2026-07-03T06:10:16Z",
        "agents": seats[:agents],
    }


def land_chunk(cache_dir: Path, *ids: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"episodes": [make_episode(i) for i in ids]}
    (cache_dir / f"chunk-{ids[0]}-{ids[-1]}.json").write_text(json.dumps(payload))


def test_empty_or_missing_cache_has_no_ids(tmp_path: Path) -> None:
    assert enrich.cached_episode_ids(tmp_path / "does-not-exist") == set()
    assert enrich.load_meta(tmp_path / "does-not-exist") == {}


def test_fetch_missing_skips_cached_ids_and_lands_raw_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cache = tmp_path / "episode_meta"
    land_chunk(cache, 1)
    calls: list[list[int]] = []

    def fake_fetch(ids: list[int]) -> dict:
        calls.append(ids)
        # id 3 is "unknown" to the API: it must be reported, not silently dropped
        return {"episodes": [make_episode(i) for i in ids if i != 3]}

    monkeypatch.setattr(enrich, "_fetch_chunk", fake_fetch)
    monkeypatch.setattr(enrich, "CHUNK_SIZE", 2)
    monkeypatch.setattr(enrich, "SLEEP_BETWEEN_CALLS_S", 0)

    with caplog.at_level(logging.WARNING, logger=enrich.__name__):
        assert enrich.fetch_missing([1, 2, 3, 4], cache) == 3
    assert calls == [[2, 3], [4]]  # cached id 1 skipped, remainder chunked
    # The warning is a log record now, not a printed line: this is a library
    # function, and stdout belongs to whichever command called it.
    warned = [record for record in caplog.records if record.message == "ids not returned"]
    assert [record.__dict__["missing"] for record in warned] == [1]
    assert enrich.cached_episode_ids(cache) == {1, 2, 4}
    assert sorted(p.name for p in cache.glob("chunk-*.json")) == [
        "chunk-1-1.json",
        "chunk-2-3.json",
        "chunk-4-4.json",
    ]

    # everything known is now cached: no further API calls
    assert enrich.fetch_missing([1, 2, 4], cache) == 0
    assert len(calls) == 2


def test_load_meta_indexes_agent_fields_by_seat(tmp_path: Path) -> None:
    cache = tmp_path / "episode_meta"
    land_chunk(cache, 10, 11)
    meta = enrich.load_meta(cache)

    assert set(meta) == {10, 11}
    m = meta[10]
    assert m["played_at"] == datetime(2026, 7, 3, 6, 8, 10, tzinfo=UTC)
    assert m["ended_at"] == datetime(2026, 7, 3, 6, 10, 16, tzinfo=UTC)
    assert m["submission_id"] == [111, 222]
    assert m["team_id"] == [1, 2]
    assert m["rating_before"] == [1000.0, 1100.0]
    assert m["rating_after"] == [990.0, 1110.0]


def test_load_meta_tolerates_missing_seat_and_timestamps(tmp_path: Path) -> None:
    cache = tmp_path / "episode_meta"
    cache.mkdir()
    ep = make_episode(7, agents=1)
    del ep["endTime"]
    (cache / "chunk-7-7.json").write_text(json.dumps({"episodes": [ep]}))

    m = enrich.load_meta(cache)[7]
    assert m["ended_at"] is None
    assert m["submission_id"] == [111, None]
    assert m["rating_after"] == [990.0, None]
