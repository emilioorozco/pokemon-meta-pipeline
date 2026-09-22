"""The fixture refresh script: what it picks, what it drops, what it writes.

The bucket is served by moto, so the listing and the reads are exercised through
a real boto3 client with no AWS account and no network. The blobs are built here
through the contract models with invented handles (PlayerA, PlayerB), one per
route the script can take: four games that between them cover three archetypes,
both seats, a concede, the longest game and both results, a spare candidate, and
five objects that must never become a fixture.

Nothing in this module reads or writes the real fixture directory: every run
writes into tmp_path.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
from moto import mock_aws

from pipeline.contract import (
    CardRef,
    CompetitiveElo,
    Decklist,
    DecklistSource,
    EndReason,
    Entry,
    ExportVariant,
    GameResult,
    GameSummary,
    ParsedBlobV1,
    ParsedBlobV2,
    RoleStats,
    Segment,
    SideStats,
    SubEntry,
)
from scripts.refresh_fixtures import (
    SCRUBBED_KEYS,
    Candidate,
    main,
    scrub,
    select_fixtures,
    user_token,
)

BUCKET: Final = "parsed-blobs-test-bucket"
REGION: Final = "us-west-2"
KEY: Final = b"test-key-not-a-real-secret"
ME: Final = "PlayerA"
OPPONENT: Final = "PlayerB"
RAW_USER_ID: Final = "acct-91f3"
HEX16: Final = re.compile(r"^[0-9a-f]{16}$")
USER_ID: Final = re.compile(r"^user-[0-9a-f]{8}$")

CHARIZARD: Final = "Charizard ex"
GARDEVOIR: Final = "Gardevoir ex"
MIRAIDON: Final = "Miraidon ex"

# gameIds are content hashes upstream; these sort in the order they are numbered.
G1: Final = "1f1e2d3c4b5a6978"
G2: Final = "2f1e2d3c4b5a6978"
G3: Final = "3f1e2d3c4b5a6978"
G4: Final = "4f1e2d3c4b5a6978"
G5: Final = "5f1e2d3c4b5a6978"
FULL_DECKLISTS: Final = "6f1e2d3c4b5a6978"
MANUAL: Final = "7f1e2d3c4b5a6978"
PASTED_LIST: Final = "8f1e2d3c4b5a6978"
EXCLUDED: Final = "9f1e2d3c4b5a6978"
NO_SEGMENTS: Final = "af1e2d3c4b5a6978"

REJECTED: Final = (FULL_DECKLISTS, MANUAL, PASTED_LIST, EXCLUDED, NO_SEGMENTS)


def side_stats(**overrides: Any) -> SideStats:
    base: dict[str, Any] = {
        "cards_drawn": 11,
        "energy_attached": 3,
        "damage_dealt": 250,
        "knockouts": 2,
        "prizes_taken": 4,
        "mulligans": 0,
        "turns_taken": 7,
        "pokemon_played": ["Pikachu"],
        "cards_played": ["Professor's Research"],
        "evolutions": [],
        "attacks": ["Thunder Shock"],
    }
    return SideStats(**{**base, **overrides})


def segments() -> list[Segment]:
    """Two segments with a handle in the player, the title, an actor, text and details."""
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


def decklist(source: DecklistSource = DecklistSource.PASTE, complete: bool = False) -> Decklist:
    return Decklist(
        cards=[CardRef(card_id="sv1-1", name="Pikachu", set="SV1", number="1", count=4)],
        card_count=60,
        complete=complete,
        source=source,
    )


def blob_v2(
    game_id: str,
    *,
    played_at: str = "2026-09-02T18:22:00.000Z",
    export_variant: ExportVariant = "stock",
    my_side: int = 0,
    result: GameResult = "win",
    end_reason: EndReason = "prizes",
    turn_count: int = 10,
    archetype: str | None = CHARIZARD,
    full_decklists: bool = False,
    excluded: bool | None = None,
    opponent_list: bool = False,
    with_segments: bool = True,
) -> dict[str, Any]:
    """A v2 blob as it arrives on the wire: camelCase, optional keys absent when unset."""
    summary = GameSummary(
        game_id=game_id,
        user_id=RAW_USER_ID,
        uploaded_at=played_at,
        played_at=played_at,
        export_variant=export_variant,
        parser_version=7,
        unparsed_count=1,
        players=[ME, OPPONENT],
        my_side=my_side,
        opponent_name=OPPONENT,
        result=result,
        end_reason=end_reason,
        winner=ME if result == "win" else OPPONENT,
        turn_count=turn_count,
        stats=RoleStats(me=side_stats(), opponent=side_stats(knockouts=1)),
        has_full_decklists=full_decklists,
        opponent_archetype=archetype,
        my_archetype=MIRAIDON,
        excluded_from_stats=excluded,
        # Everything below is scrubbed away; `notes` even carries a handle, which
        # the anonymizer does not rewrite, so the key has to go.
        match_id="match-7781",
        upload_token_id="tok-4410",
        upload_client="overlay-mac 1.4.2",
        deck_id="deck-1029",
        deck_name="my list",
        deck_version=3,
        notes=f"friendly against {OPPONENT}",
        tournament_id="cup-12",
        tournament_name="Regional Cup",
        round="R3",
        elo=CompetitiveElo(season_id="s-9", previous_elo=1500, new_elo=1512, delta=12),
    )
    blob = ParsedBlobV2(
        schema_version=2,
        summary=summary,
        segments=segments() if with_segments else [],
        stats_by_player={ME: side_stats(), OPPONENT: side_stats(knockouts=1)},
        unparsed_lines=[f"{ME} did something no pattern matched"],
        extras={"Note": [f"a trailer line naming {OPPONENT}"]},
        my_decklist=decklist(DecklistSource.DEBUG, True) if full_decklists else None,
        opponent_decklist=decklist(DecklistSource.DEBUG, True) if full_decklists else None,
    )
    body = blob.model_dump(mode="json", by_alias=True, exclude_none=True)
    if opponent_list:
        body["opponentDecklist"] = decklist().model_dump(mode="json", by_alias=True)
    return body


def blob_v1() -> dict[str, Any]:
    """The pre-contract shape: no schemaVersion and no summary, so never a fixture."""
    blob = ParsedBlobV1(
        segments=segments(),
        stats_by_player={ME: side_stats(), OPPONENT: side_stats()},
        unparsed_lines=[],
        extras={},
    )
    return blob.model_dump(mode="json", by_alias=True, exclude_none=True)


def bodies() -> dict[str, bytes]:
    """One object per route: five candidates, five rejects, one unreadable object."""
    blobs = {
        G1: blob_v2(G1, my_side=0, result="win", end_reason="prizes", turn_count=14),
        G2: blob_v2(
            G2,
            my_side=1,
            result="loss",
            end_reason="concede",
            turn_count=9,
            archetype=GARDEVOIR,
        ),
        G3: blob_v2(G3, my_side=0, result="loss", end_reason="prizes", turn_count=22),
        G4: blob_v2(
            G4,
            my_side=1,
            result="win",
            end_reason="opponent_concede",
            turn_count=11,
            archetype=MIRAIDON,
        ),
        G5: blob_v2(G5, my_side=0, result="win", end_reason="prizes", turn_count=5),
        FULL_DECKLISTS: blob_v2(FULL_DECKLISTS, export_variant="debug", full_decklists=True),
        MANUAL: blob_v2(MANUAL, export_variant="manual"),
        PASTED_LIST: blob_v2(PASTED_LIST, opponent_list=True),
        EXCLUDED: blob_v2(EXCLUDED, excluded=True),
        NO_SEGMENTS: blob_v2(NO_SEGMENTS, with_segments=False),
    }
    objects = {f"parsed/{RAW_USER_ID}/{name}.json": _json(body) for name, body in blobs.items()}
    objects[f"parsed/{RAW_USER_ID}/legacy.json"] = _json(blob_v1())
    objects[f"parsed/{RAW_USER_ID}/broken.json"] = b"this is not json, it is a log line"
    return objects


def candidate(
    game_id: str,
    *,
    seat: int | None = 0,
    result: str = "win",
    end_reason: str = "prizes",
    turns: int = 10,
    archetype: str | None = CHARIZARD,
) -> Candidate:
    """A selection-only candidate: no body, because `select_fixtures` never reads one."""
    return Candidate(
        game_id=game_id,
        my_side=seat,
        result=result,
        end_reason=end_reason,
        turn_count=turns,
        opponent_archetype=archetype,
        raw={},
    )


@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials so a misconfigured run can never reach a real account."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def s3(aws_credentials: None) -> Iterator[Any]:
    """A bucket holding one object per route the script can take."""
    import boto3

    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        for key, body in bodies().items():
            client.put_object(Bucket=BUCKET, Key=key, Body=body)
        yield client


def _json(blob: dict[str, Any]) -> bytes:
    return json.dumps(blob).encode("utf-8")


def written(out_dir: Path) -> list[dict[str, Any]]:
    """Every written fixture, in filename order."""
    return [json.loads(path.read_text()) for path in sorted(out_dir.glob("game-*.json"))]


# ---- the whole run, against moto ----


def test_a_run_writes_one_file_per_pick(s3: Any, tmp_path: Path) -> None:
    assert main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "4"]) == 0

    names = [path.name for path in sorted(tmp_path.glob("game-*.json"))]
    assert names == [
        f"game-01-{G2[:8]}.json",
        f"game-02-{G3[:8]}.json",
        f"game-03-{G4[:8]}.json",
        f"game-04-{G1[:8]}.json",
    ]
    assert len(list(tmp_path.iterdir())) == 4


def test_the_games_that_may_not_be_fixtures_are_not_written(s3: Any, tmp_path: Path) -> None:
    """Debug exports, manual games, a pasted opponent list, excluded and empty games."""
    main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "4"])

    picked = {blob["summary"]["gameId"] for blob in written(tmp_path)}
    assert picked == {G1, G2, G3, G4}
    assert picked.isdisjoint(REJECTED)


def test_the_picked_set_covers_what_the_candidates_allow(s3: Any, tmp_path: Path) -> None:
    main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "4"])

    summaries = [blob["summary"] for blob in written(tmp_path)]
    assert {summary["opponentArchetype"] for summary in summaries} == {
        CHARIZARD,
        GARDEVOIR,
        MIRAIDON,
    }
    assert {summary["result"] for summary in summaries} >= {"win", "loss"}
    assert any(summary["endReason"] in ("concede", "opponent_concede") for summary in summaries)
    assert max(summary["turnCount"] for summary in summaries) == 22
    assert {summary["mySide"] for summary in summaries} == {0, 1}


def test_every_written_game_is_anonymized_and_scrubbed(s3: Any, tmp_path: Path) -> None:
    main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "4"])

    for path in sorted(tmp_path.glob("game-*.json")):
        text = path.read_text()
        assert ME not in text
        assert OPPONENT not in text
        assert RAW_USER_ID not in text
        # The handles were there to begin with: tokens replaced them, not nothing.
        assert "used Thunder Shock" in text

        blob = json.loads(text)
        summary = blob["summary"]
        assert all(HEX16.match(player) for player in summary["players"])
        assert all(HEX16.match(handle) for handle in blob["statsByPlayer"])
        assert HEX16.match(summary["winner"])
        assert USER_ID.match(summary["userId"])
        assert [name for name in SCRUBBED_KEYS if name in summary] == []
        assert "opponentDecklist" not in blob
        assert summary["exportVariant"] == "stock"
        assert summary["hasFullDecklists"] is False


def test_the_summary_reports_the_counts_and_the_unmet_target(
    s3: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "4"])

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[:4] == ["listed: 12", "not a v2 blob: 2", "candidates: 5", "picked: 4"]
    assert lines[4] == (
        f"01 {G2[:8]} seat=1 result=loss end=concede turns=9 opp_archetype={GARDEVOIR}"
    )
    assert "coverage met: archetypes: 3 of 3 distinct" in out
    assert "coverage unmet: seats: 2 on seat 0 and 2 on seat 1, want 3 each" in out
    assert ME not in out and OPPONENT not in out and RAW_USER_ID not in out
    assert "parsed/" not in out


def test_a_refresh_replaces_the_previous_set(s3: Any, tmp_path: Path) -> None:
    stale = tmp_path / "game-99-deadbeef.json"
    stale.write_text("{}\n")
    keep = tmp_path / "README.md"
    keep.write_text("not a fixture\n")

    main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "2"])

    assert not stale.exists()
    assert keep.exists()
    assert len(list(tmp_path.glob("game-*.json"))) == 2


def test_the_same_seed_key_reproduces_the_same_files(s3: Any, tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    seed = KEY.hex()

    main(["--bucket", BUCKET, "--out", str(first), "--seed-key", seed])
    main(["--bucket", BUCKET, "--out", str(second), "--seed-key", seed])

    names = [path.name for path in sorted(first.glob("game-*.json"))]
    assert names and names == [path.name for path in sorted(second.glob("game-*.json"))]
    for name in names:
        assert (first / name).read_text() == (second / name).read_text()


def test_a_count_above_the_candidates_writes_every_candidate(s3: Any, tmp_path: Path) -> None:
    main(["--bucket", BUCKET, "--out", str(tmp_path), "--count", "10"])

    assert len(list(tmp_path.glob("game-*.json"))) == 5


def test_a_prefix_with_nothing_usable_writes_nothing(
    aws_credentials: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import boto3

    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        client.put_object(Bucket=BUCKET, Key="parsed/u/legacy.json", Body=_json(blob_v1()))

        assert main(["--bucket", BUCKET, "--out", str(tmp_path)]) == 0

    assert list(tmp_path.glob("game-*.json")) == []
    assert "candidates: 0" in capsys.readouterr().out


# ---- selection, without S3 ----


def test_selection_follows_the_priority_order() -> None:
    pool = [
        candidate(G1, seat=0, result="win", turns=14),
        candidate(G2, seat=1, result="loss", end_reason="concede", turns=9, archetype=GARDEVOIR),
        candidate(G3, seat=0, result="loss", turns=22),
        candidate(
            G4, seat=1, result="win", end_reason="opponent_concede", turns=11, archetype=MIRAIDON
        ),
        candidate(G5, seat=0, result="win", turns=5),
    ]

    selection = select_fixtures(pool, 4)

    # A new archetype, then a seat, then the concede, then the longest game, then
    # the missing result; G2 leads because it is the only concede among the ties.
    assert [pick.game_id for pick in selection.picked] == [G2, G3, G4, G1]


def test_selection_is_deterministic_whatever_the_input_order() -> None:
    pool = [
        candidate(G1, turns=14),
        candidate(G2, seat=1, result="loss", end_reason="concede", archetype=GARDEVOIR),
        candidate(G3, seat=1, turns=22, archetype=MIRAIDON),
        candidate(G4, result="loss"),
    ]

    once = select_fixtures(pool, 3)
    again = select_fixtures(list(reversed(pool)), 3)

    assert once == again
    assert select_fixtures(pool, 3) == once


def test_an_unreachable_target_is_reported_and_not_fatal() -> None:
    pool = [candidate(G1), candidate(G2), candidate(G3)]

    selection = select_fixtures(pool, 10)

    assert [pick.game_id for pick in selection.picked] == [G1, G2, G3]
    assert any("seats: 3 on seat 0 and 0 on seat 1" in item for item in selection.unmet)
    assert any("results: win" in item for item in selection.unmet)
    assert any("archetypes: 1 of 1 distinct" in item for item in selection.met)


def test_the_count_caps_the_picks_and_the_spares_fill_the_rest() -> None:
    pool = [candidate(name) for name in (G1, G2, G3, G4, G5)]

    assert len(select_fixtures(pool, 2).picked) == 2
    assert [pick.game_id for pick in select_fixtures(pool, 4).picked] == [G1, G2, G3, G4]
    assert select_fixtures([], 4).picked == []


def test_a_candidate_without_a_seat_or_an_archetype_is_still_pickable() -> None:
    pool = [candidate(G1, seat=None, archetype=None)]

    selection = select_fixtures(pool, 4)

    assert [pick.game_id for pick in selection.picked] == [G1]
    assert selection.picked[0].line(1) == (
        f"01 {G1[:8]} seat=? result=win end=prizes turns=10 opp_archetype=unknown"
    )


# ---- the scrub, without S3 ----


def test_scrub_drops_the_identifying_keys_and_tokenizes_the_user() -> None:
    anon: dict[str, Any] = {
        "summary": {
            "gameId": G1,
            "userId": RAW_USER_ID,
            "notes": "keep an eye on this one",
            "elo": {"seasonId": "s-9"},
            **{name: "something" for name in SCRUBBED_KEYS if name != "elo"},
        },
        "segments": [],
    }

    out = scrub(anon, KEY, RAW_USER_ID)

    assert [name for name in SCRUBBED_KEYS if name in out["summary"]] == []
    assert out["summary"]["gameId"] == G1
    assert out["summary"]["userId"] == user_token(RAW_USER_ID, KEY)
    assert USER_ID.match(out["summary"]["userId"])
    # The input is untouched, and a second scrub gives the same answer.
    assert anon["summary"]["userId"] == RAW_USER_ID
    assert "notes" in anon["summary"]
    assert scrub(anon, KEY, RAW_USER_ID) == out


def test_scrub_leaves_a_blob_without_a_summary_alone() -> None:
    assert scrub({"segments": []}, KEY, RAW_USER_ID) == {"segments": []}


def test_the_user_token_changes_with_the_key() -> None:
    assert user_token(RAW_USER_ID, KEY) != user_token(RAW_USER_ID, b"another-test-key")
