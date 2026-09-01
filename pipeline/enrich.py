"""Enrich: recover episode metadata (timestamps, ratings, submission ids) from Kaggle.

The replay JSONs on disk carry no timestamps — that metadata lives on Kaggle's
side and the original corpus pull discarded it. Kaggle's ListEpisodes endpoint
accepts explicit episode ids and still answers for old episodes, returning per
episode: createTime/endTime, and per agent: submissionId, reward, and the ladder
rating before/after the game.

Caching contract: every raw API response is landed verbatim under
LAKE_DIR/raw/episode_meta/ before anything parses it, and ids found in the cache
are never re-fetched. Land the whole response — you can't re-ask a dead
competition for fields you dropped.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from pipeline.config import LAKE_DIR

LIST_EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
META_CACHE_DIR = LAKE_DIR / "raw" / "episode_meta"
CHUNK_SIZE = 100
SLEEP_BETWEEN_CALLS_S = 1.0


def _cached_chunks(cache_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(cache_dir.glob("chunk-*.json"))]


def cached_episode_ids(cache_dir: Path = META_CACHE_DIR) -> set[int]:
    ids: set[int] = set()
    for chunk in _cached_chunks(cache_dir):
        ids.update(ep["id"] for ep in chunk.get("episodes", []))
    return ids


def _fetch_chunk(ids: list[int]) -> dict:
    req = urllib.request.Request(
        LIST_EPISODES_URL,
        data=json.dumps({"ids": ids}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def fetch_missing(episode_ids: list[int], cache_dir: Path = META_CACHE_DIR) -> int:
    """Fetch metadata for any ids not already cached. Returns count fetched."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = sorted(set(episode_ids) - cached_episode_ids(cache_dir))
    for i in range(0, len(missing), CHUNK_SIZE):
        chunk = missing[i : i + CHUNK_SIZE]
        payload = _fetch_chunk(chunk)
        got = {ep["id"] for ep in payload.get("episodes", [])}
        if not_found := set(chunk) - got:
            print(f"  warning: {len(not_found)} ids not returned: {sorted(not_found)[:5]}...")
        out = cache_dir / f"chunk-{chunk[0]}-{chunk[-1]}.json"
        out.write_text(json.dumps(payload))
        print(f"  fetched {len(got)} episodes -> {out.name}")
        if i + CHUNK_SIZE < len(missing):
            time.sleep(SLEEP_BETWEEN_CALLS_S)
    return len(missing)


def _parse_ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def load_meta(cache_dir: Path = META_CACHE_DIR) -> dict[int, dict]:
    """Parse the cache into {episode_id: metadata} for the ingest join.

    Per-seat lists are indexed by the agent's seat (its `index` field; Kaggle
    omits it for seat 0), which matches the replay's seat ordering.
    """
    meta: dict[int, dict] = {}
    for chunk in _cached_chunks(cache_dir):
        for ep in chunk.get("episodes", []):
            seats: list[dict | None] = [None, None]
            for agent in ep.get("agents", []):
                seats[agent.get("index", 0)] = agent
            meta[ep["id"]] = {
                "played_at": _parse_ts(ep.get("createTime")),
                "ended_at": _parse_ts(ep.get("endTime")),
                "submission_id": [a and a.get("submissionId") for a in seats],
                "team_id": [a and a.get("teamId") for a in seats],
                "rating_before": [a and a.get("initialScore") for a in seats],
                "rating_after": [a and a.get("updatedScore") for a in seats],
            }
    return meta
