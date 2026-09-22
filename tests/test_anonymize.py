"""Handle anonymization: deterministic tokens, every location rewritten, no collateral edits.

The blob factory here is deliberately independent of the contract package so
this module tests the rewrite alone. Handles are invented: `Ash` is a prefix
of `Ash_K`, and `Ash K` contains a space.
"""

import copy
import re
from typing import Any

import pytest

from pipeline.anonymize import anonymize, assert_no_handles, handles_in, token_for

KEY = b"test-key-not-secret"
OTHER_KEY = b"another-test-key"
HEX16 = re.compile(r"^[0-9a-f]{16}$")


def make_blob(me: str = "Ash K", opp: str = "Ash", version: int | None = 2) -> dict[str, Any]:
    """A small parsed blob with handles in every documented location.

    `version=None` yields a v1 blob (no schemaVersion, no summary).
    """
    blob: dict[str, Any] = {
        "segments": [
            {
                "kind": "setup",
                "title": "Setup",
                "entries": [
                    {
                        "line": 1,
                        "text": f"{opp} won the coin toss",
                        "kind": "coin_toss",
                        "actor": opp,
                        "fields": {"won": True},
                        "subs": [],
                    },
                    {
                        "line": 2,
                        "text": f"{me} drew 7 cards for the opening hand.",
                        "kind": "opening_hand",
                        "actor": me,
                        "fields": {"n": 7},
                        "subs": [
                            {
                                "line": 3,
                                "text": f"{me}'s Ashes Charm was revealed",
                                "kind": "other",
                                "fields": {},
                                "details": [f"{opp} looked at Ashes Charm", "Ashes Charm"],
                            }
                        ],
                    },
                ],
            },
            {
                "kind": "turn",
                "turnNumber": 1,
                "player": opp,
                "title": f"Turn # 1 - {opp}'s Turn",
                "entries": [
                    {
                        "line": 4,
                        "text": f"{opp}'s Charmander used Scratch on {me}'s Squirtle for 10.",
                        "kind": "attack",
                        "actor": opp,
                        "fields": {
                            "pokemon": "Charmander",
                            "move": "Scratch",
                            "targetOwner": me,
                            "target": "Squirtle",
                            "damage": 10,
                        },
                        "subs": [],
                    },
                    {
                        "line": 5,
                        "text": f"{opp} conceded. {me} wins.",
                        "kind": "concede",
                        "actor": me,
                        "fields": {"who": opp, "loser": opp, "winner": me},
                        "subs": [],
                    },
                ],
            },
        ],
        "statsByPlayer": {
            me: {"cardsDrawn": 7, "pokemonPlayed": ["Squirtle"], "turnsTaken": 0},
            opp: {"cardsDrawn": 0, "pokemonPlayed": ["Charmander"], "turnsTaken": 1},
        },
        "unparsedLines": [f"{me}'s Ashes Charm was revealed"],
        "extras": {"Deck": [f"{opp} is playing Fire", "Ashes only"]},
        "myDecklist": {"cards": [{"name": "Squirtle", "count": 4}]},
        "opponentDecklist": None,
    }
    if version is not None:
        blob["schemaVersion"] = version
        blob["summary"] = {
            "gameId": "0123456789abcdef",
            "userId": "user-1",
            "players": [opp, me],
            "mySide": 1,
            "opponentName": opp,
            "winner": me,
            "result": "win",
            "turnCount": 1,
            "stats": {"me": {"cardsDrawn": 7}, "opponent": {"cardsDrawn": 0}},
        }
    return blob


def make_manual_blob(me: str = "Ash K", opp: str = "Ash") -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "summary": {
            "gameId": "manual-0001",
            "userId": "user-1",
            "players": [me, opp],
            "mySide": 0,
            "opponentName": opp,
            "winner": opp,
            "result": "loss",
            "exportVariant": "manual",
        },
        "segments": [],
        "statsByPlayer": {},
        "unparsedLines": [],
        "extras": {},
    }


def test_token_is_16_lowercase_hex_and_deterministic() -> None:
    token = token_for("Ash K", KEY)
    assert HEX16.match(token)
    assert token == token_for("Ash K", KEY)
    assert token != token_for("Ash", KEY)


def test_different_keys_give_different_tokens() -> None:
    assert token_for("Ash", KEY) != token_for("Ash", OTHER_KEY)


def test_empty_key_raises() -> None:
    with pytest.raises(ValueError):
        token_for("Ash", b"")
    with pytest.raises(ValueError):
        anonymize(make_blob(), b"")


def test_handles_in_v2_and_v1_and_manual() -> None:
    assert handles_in(make_blob()) == {"Ash K", "Ash"}
    assert handles_in(make_blob(version=None)) == {"Ash K", "Ash"}
    assert handles_in(make_manual_blob()) == {"Ash K", "Ash"}
    assert handles_in({"segments": [], "statsByPlayer": {}}) == set()


def test_anonymize_is_pure_and_deterministic() -> None:
    blob = make_blob()
    before = copy.deepcopy(blob)
    first = anonymize(blob, KEY)
    second = anonymize(blob, KEY)
    assert blob == before
    assert first == second
    assert first is not blob
    # a second blob with the same handle shares the token
    other = anonymize(make_blob(me="Misty99", opp="Ash"), KEY)
    assert other["summary"]["players"][0] == first["summary"]["players"][0]
    assert other["statsByPlayer"].keys() & first["statsByPlayer"].keys() == {token_for("Ash", KEY)}


def test_every_structured_location_is_rewritten() -> None:
    me, opp = token_for("Ash K", KEY), token_for("Ash", KEY)
    out = anonymize(make_blob(), KEY)

    summary = out["summary"]
    assert summary["players"] == [opp, me]
    assert summary["winner"] == me
    assert summary["opponentName"] == opp
    assert set(out["statsByPlayer"]) == {me, opp}
    assert out["statsByPlayer"][me]["cardsDrawn"] == 7

    setup, turn = out["segments"]
    assert turn["player"] == opp
    assert turn["title"] == f"Turn # 1 - {opp}'s Turn"
    assert [e["actor"] for e in setup["entries"]] == [opp, me]
    assert setup["entries"][1]["text"] == f"{me} drew 7 cards for the opening hand."
    attack, concede = turn["entries"]
    assert attack["fields"]["targetOwner"] == me
    assert attack["fields"]["damage"] == 10
    assert attack["text"] == f"{opp}'s Charmander used Scratch on {me}'s Squirtle for 10."
    assert concede["fields"] == {"who": opp, "loser": opp, "winner": me}
    sub = setup["entries"][1]["subs"][0]
    assert sub["text"] == f"{me}'s Ashes Charm was revealed"
    assert sub["details"] == [f"{opp} looked at Ashes Charm", "Ashes Charm"]
    assert out["unparsedLines"] == [f"{me}'s Ashes Charm was revealed"]
    assert out["extras"] == {"Deck": [f"{opp} is playing Fire", "Ashes only"]}
    # untouched parts survive as they were
    assert out["myDecklist"] == {"cards": [{"name": "Squirtle", "count": 4}]}
    assert out["opponentDecklist"] is None
    assert summary["stats"] == {"me": {"cardsDrawn": 7}, "opponent": {"cardsDrawn": 0}}


def test_guard_reports_paths_before_and_nothing_after() -> None:
    blob = make_blob()
    handles = handles_in(blob)
    before = assert_no_handles(blob, handles)
    assert "summary.players[0]" in before
    assert "summary.winner" in before
    assert "statsByPlayer.<key>" in before
    assert "segments[1].title" in before
    assert "segments[0].entries[1].subs[0].details[0]" in before
    assert "unparsedLines[0]" in before
    assert "extras.Deck[0]" in before
    assert not any("Ash" in path for path in before), "paths must not echo a handle"
    assert assert_no_handles(anonymize(blob, KEY), handles) == []


def test_prefix_handle_does_not_corrupt_longer_handle() -> None:
    out = anonymize(make_blob(me="Ash_K", opp="Ash"), KEY)
    me, opp = token_for("Ash_K", KEY), token_for("Ash", KEY)
    assert out["summary"]["players"] == [opp, me]
    assert out["segments"][0]["entries"][1]["text"] == f"{me} drew 7 cards for the opening hand."
    assert out["segments"][1]["entries"][0]["fields"]["targetOwner"] == me
    assert assert_no_handles(out, {"Ash_K", "Ash"}) == []

    spaced = anonymize(make_blob(me="Ash K", opp="Ash"), KEY)
    assert spaced["segments"][0]["entries"][1]["text"].startswith(token_for("Ash K", KEY))
    assert token_for("Ash", KEY) + " K" not in spaced["segments"][0]["entries"][1]["text"]


def test_handle_inside_unrelated_word_is_left_alone() -> None:
    out = anonymize(make_blob(me="Misty99", opp="Ash"), KEY)
    sub = out["segments"][0]["entries"][1]["subs"][0]
    assert sub["details"][1] == "Ashes Charm"
    assert out["extras"]["Deck"][1] == "Ashes only"
    assert "Ashes" in sub["text"]
    # digits and underscores are word characters too: Misty99 does not match Misty999 or Misty99_x
    assert anonymize(
        {
            "segments": [],
            "statsByPlayer": {"Misty99": {}},
            "unparsedLines": ["Misty999", "Misty99_x"],
            "extras": {},
        },
        KEY,
    )["unparsedLines"] == ["Misty999", "Misty99_x"]


def test_v1_blob_without_summary() -> None:
    blob = make_blob(version=None)
    out = anonymize(blob, KEY)
    assert "summary" not in out
    assert set(out["statsByPlayer"]) == {token_for("Ash K", KEY), token_for("Ash", KEY)}
    assert out["segments"][1]["player"] == token_for("Ash", KEY)
    assert assert_no_handles(out, handles_in(blob)) == []


def test_manual_game_without_segments() -> None:
    blob = make_manual_blob()
    out = anonymize(blob, KEY)
    assert out["segments"] == []
    assert out["summary"]["players"] == [token_for("Ash K", KEY), token_for("Ash", KEY)]
    assert out["summary"]["winner"] == token_for("Ash", KEY)
    assert assert_no_handles(out, handles_in(blob)) == []


def test_manual_game_archetype_in_players_is_not_a_handle() -> None:
    # A manual game names the opponent by archetype in players[1], opponentName and
    # winner; that string must survive untouched everywhere, including opponentArchetype.
    blob = make_manual_blob(me="Ash K", opp="Dragapult Dusknoir")
    blob["summary"]["opponentArchetype"] = "Dragapult Dusknoir"
    blob["summary"]["myArchetype"] = "Gardevoir ex"
    assert handles_in(blob) == {"Ash K"}
    out = anonymize(blob, KEY)
    assert out["summary"]["players"] == [token_for("Ash K", KEY), "Dragapult Dusknoir"]
    assert out["summary"]["opponentName"] == "Dragapult Dusknoir"
    assert out["summary"]["winner"] == "Dragapult Dusknoir"
    assert out["summary"]["opponentArchetype"] == "Dragapult Dusknoir"
    assert assert_no_handles(out, {"Ash K"}) == []
